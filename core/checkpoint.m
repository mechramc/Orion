#import "checkpoint.h"
#import "../model/configs/stories110m.h"
#import <stdio.h>
#import <string.h>

// T075: Checkpoint save
// T076: Checkpoint resume
//
// Binary format:
//   OrionCkptHdr (fixed size)
//   Per-layer weights (fp32): rms_att, wq, wk, wv, wo, rms_ffn, w1, w3, w2
//   Per-layer Adam m: same order as weights
//   Per-layer Adam v: same order as weights
//   Embedding (fp32): [vocab * dim]
//   rms_final (fp32): [dim]
//   Adam embed m,v: [vocab * dim] each
//   Adam rms_final m,v: [dim] each

#pragma mark - Helpers

static bool write_buf(FILE *f, const void *buf, size_t bytes) {
    return fwrite(buf, 1, bytes, f) == bytes;
}

static bool read_buf(FILE *f, void *buf, size_t bytes) {
    return fread(buf, 1, bytes, f) == bytes;
}

// Write all 9 weight arrays for a layer
static bool write_layer_weights(FILE *f, const OrionLayerWeights *w,
                                 int d, int h) {
    return write_buf(f, w->rms_att, d * sizeof(float)) &&
           write_buf(f, w->wq, d*d * sizeof(float)) &&
           write_buf(f, w->wk, d*d * sizeof(float)) &&
           write_buf(f, w->wv, d*d * sizeof(float)) &&
           write_buf(f, w->wo, d*d * sizeof(float)) &&
           write_buf(f, w->rms_ffn, d * sizeof(float)) &&
           write_buf(f, w->w1, h*d * sizeof(float)) &&
           write_buf(f, w->w3, h*d * sizeof(float)) &&
           write_buf(f, w->w2, d*h * sizeof(float));
}

static bool read_layer_weights(FILE *f, OrionLayerWeights *w,
                                int d, int h) {
    return read_buf(f, w->rms_att, d * sizeof(float)) &&
           read_buf(f, w->wq, d*d * sizeof(float)) &&
           read_buf(f, w->wk, d*d * sizeof(float)) &&
           read_buf(f, w->wv, d*d * sizeof(float)) &&
           read_buf(f, w->wo, d*d * sizeof(float)) &&
           read_buf(f, w->rms_ffn, d * sizeof(float)) &&
           read_buf(f, w->w1, h*d * sizeof(float)) &&
           read_buf(f, w->w3, h*d * sizeof(float)) &&
           read_buf(f, w->w2, d*h * sizeof(float));
}

// Write Adam m+v for a layer (same field order as weights)
static bool write_layer_adam(FILE *f, const OrionLayerAdam *a,
                              int d, int h) {
    // m arrays
    if (!write_buf(f, a->m_rms_att, d * sizeof(float))) return false;
    if (!write_buf(f, a->m_wq, d*d * sizeof(float))) return false;
    if (!write_buf(f, a->m_wk, d*d * sizeof(float))) return false;
    if (!write_buf(f, a->m_wv, d*d * sizeof(float))) return false;
    if (!write_buf(f, a->m_wo, d*d * sizeof(float))) return false;
    if (!write_buf(f, a->m_rms_ffn, d * sizeof(float))) return false;
    if (!write_buf(f, a->m_w1, h*d * sizeof(float))) return false;
    if (!write_buf(f, a->m_w3, h*d * sizeof(float))) return false;
    if (!write_buf(f, a->m_w2, d*h * sizeof(float))) return false;
    // v arrays
    if (!write_buf(f, a->v_rms_att, d * sizeof(float))) return false;
    if (!write_buf(f, a->v_wq, d*d * sizeof(float))) return false;
    if (!write_buf(f, a->v_wk, d*d * sizeof(float))) return false;
    if (!write_buf(f, a->v_wv, d*d * sizeof(float))) return false;
    if (!write_buf(f, a->v_wo, d*d * sizeof(float))) return false;
    if (!write_buf(f, a->v_rms_ffn, d * sizeof(float))) return false;
    if (!write_buf(f, a->v_w1, h*d * sizeof(float))) return false;
    if (!write_buf(f, a->v_w3, h*d * sizeof(float))) return false;
    if (!write_buf(f, a->v_w2, d*h * sizeof(float))) return false;
    return true;
}

static bool read_layer_adam(FILE *f, OrionLayerAdam *a,
                             int d, int h) {
    if (!read_buf(f, a->m_rms_att, d * sizeof(float))) return false;
    if (!read_buf(f, a->m_wq, d*d * sizeof(float))) return false;
    if (!read_buf(f, a->m_wk, d*d * sizeof(float))) return false;
    if (!read_buf(f, a->m_wv, d*d * sizeof(float))) return false;
    if (!read_buf(f, a->m_wo, d*d * sizeof(float))) return false;
    if (!read_buf(f, a->m_rms_ffn, d * sizeof(float))) return false;
    if (!read_buf(f, a->m_w1, h*d * sizeof(float))) return false;
    if (!read_buf(f, a->m_w3, h*d * sizeof(float))) return false;
    if (!read_buf(f, a->m_w2, d*h * sizeof(float))) return false;
    if (!read_buf(f, a->v_rms_att, d * sizeof(float))) return false;
    if (!read_buf(f, a->v_wq, d*d * sizeof(float))) return false;
    if (!read_buf(f, a->v_wk, d*d * sizeof(float))) return false;
    if (!read_buf(f, a->v_wv, d*d * sizeof(float))) return false;
    if (!read_buf(f, a->v_wo, d*d * sizeof(float))) return false;
    if (!read_buf(f, a->v_rms_ffn, d * sizeof(float))) return false;
    if (!read_buf(f, a->v_w1, h*d * sizeof(float))) return false;
    if (!read_buf(f, a->v_w3, h*d * sizeof(float))) return false;
    if (!read_buf(f, a->v_w2, d*h * sizeof(float))) return false;
    return true;
}

#pragma mark - Save

bool orion_checkpoint_save(const OrionTrainer* t, const char* path,
                           int step, float loss) {
    FILE *f = fopen(path, "wb");
    if (!f) return false;

    const OrionModelConfig *cfg = t->cfg;
    int d = cfg->d_model, h = cfg->hidden_dim, v = cfg->vocab;
    int nl = t->n_layers;

    // Write header
    OrionCkptHdr hdr = {0};
    hdr.magic = ORION_CKPT_MAGIC;
    hdr.version = ORION_CKPT_VERSION;
    hdr.step = step;
    hdr.n_layers = nl;
    hdr.vocab_size = v;
    hdr.dim = d;
    hdr.hidden_dim = h;
    hdr.n_heads = cfg->n_head;
    hdr.seq_len = cfg->max_seq;
    hdr.lr = t->lr;
    hdr.loss = loss;
    hdr.adam_t = t->adam_t;
    if (!write_buf(f, &hdr, sizeof(hdr))) { fclose(f); return false; }

    // Per-layer weights
    for (int L = 0; L < nl; L++) {
        if (!write_layer_weights(f, &t->weights[L], d, h)) { fclose(f); return false; }
    }

    // Per-layer Adam state
    for (int L = 0; L < nl; L++) {
        if (!write_layer_adam(f, &t->adam[L], d, h)) { fclose(f); return false; }
    }

    // Embedding + final RMSNorm
    if (!write_buf(f, t->embed, v * d * sizeof(float))) { fclose(f); return false; }
    if (!write_buf(f, t->rms_final, d * sizeof(float))) { fclose(f); return false; }

    // Adam state for embed + rms_final
    if (!write_buf(f, t->m_embed, v * d * sizeof(float))) { fclose(f); return false; }
    if (!write_buf(f, t->v_embed, v * d * sizeof(float))) { fclose(f); return false; }
    if (!write_buf(f, t->m_rms_final, d * sizeof(float))) { fclose(f); return false; }
    if (!write_buf(f, t->v_rms_final, d * sizeof(float))) { fclose(f); return false; }

    fclose(f);
    return true;
}

#pragma mark - Load

bool orion_checkpoint_load(OrionTrainer* t, const char* path,
                           int* out_step, float* out_loss) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;

    const OrionModelConfig *cfg = t->cfg;
    int d = cfg->d_model, h = cfg->hidden_dim, v = cfg->vocab;
    int nl = t->n_layers;

    // Read and validate header
    OrionCkptHdr hdr;
    if (!read_buf(f, &hdr, sizeof(hdr))) { fclose(f); return false; }

    if (hdr.magic != ORION_CKPT_MAGIC) {
        NSLog(@"Checkpoint: bad magic 0x%08x (expected 0x%08x)", hdr.magic, ORION_CKPT_MAGIC);
        fclose(f); return false;
    }
    if (hdr.version != ORION_CKPT_VERSION) {
        NSLog(@"Checkpoint: version %d (expected %d)", hdr.version, ORION_CKPT_VERSION);
        fclose(f); return false;
    }
    if (hdr.n_layers != nl || hdr.dim != d || hdr.hidden_dim != h || hdr.vocab_size != v) {
        NSLog(@"Checkpoint: config mismatch (layers=%d/%d, dim=%d/%d, hidden=%d/%d, vocab=%d/%d)",
              hdr.n_layers, nl, hdr.dim, d, hdr.hidden_dim, h, hdr.vocab_size, v);
        fclose(f); return false;
    }

    // Restore training state from header
    t->lr = hdr.lr;
    t->adam_t = hdr.adam_t;

    if (out_step) *out_step = hdr.step;
    if (out_loss) *out_loss = hdr.loss;

    // Per-layer weights
    for (int L = 0; L < nl; L++) {
        if (!read_layer_weights(f, &t->weights[L], d, h)) { fclose(f); return false; }
    }

    // Per-layer Adam state
    for (int L = 0; L < nl; L++) {
        if (!read_layer_adam(f, &t->adam[L], d, h)) { fclose(f); return false; }
    }

    // Embedding + final RMSNorm
    if (!read_buf(f, t->embed, v * d * sizeof(float))) { fclose(f); return false; }
    if (!read_buf(f, t->rms_final, d * sizeof(float))) { fclose(f); return false; }

    // Adam state for embed + rms_final
    if (!read_buf(f, t->m_embed, v * d * sizeof(float))) { fclose(f); return false; }
    if (!read_buf(f, t->v_embed, v * d * sizeof(float))) { fclose(f); return false; }
    if (!read_buf(f, t->m_rms_final, d * sizeof(float))) { fclose(f); return false; }
    if (!read_buf(f, t->v_rms_final, d * sizeof(float))) { fclose(f); return false; }

    fclose(f);
    return true;
}
