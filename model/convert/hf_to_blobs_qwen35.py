#!/usr/bin/env python3
"""Convert Hugging Face Qwen3.5 text weights into Orion BLOBFILE blobs.

This is a stage-1 converter for Orion Qwen porting:
  - understands Qwen3.5 text_config / layer_types
  - exports text-only weights and a manifest
  - explicitly ignores visual / mtp branches

It is intentionally conservative. The goal is to unblock loader smoke and
CPU-only inference porting, not to claim full runtime compatibility yet.
"""

import argparse
import json
import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("pip install huggingface_hub") from exc

try:
    from transformers import AutoConfig, AutoModelForCausalLM
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("pip install transformers torch") from exc

from hf_to_blobs_gpt2 import convert_tensor_to_blob, convert_tensor_transposed


TEXT_PREFIX = "model"
VISUAL_PREFIX = "visual"
MTP_PREFIX = "mtp"


def normalize_index_name(name):
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    if name.startswith("language_model.model."):
        return "model." + name[len("language_model.model."):]
    if name.startswith("language_model.lm_head."):
        return "lm_head." + name[len("language_model.lm_head."):]
    return name


def is_local_model_path(model_name):
    return Path(model_name).expanduser().exists()


def resolve_local_model_path(model_name):
    return Path(model_name).expanduser().resolve()


def output_entry(layer_idx, blob_name, hf_name, shape, transpose, alias_of=None):
    if layer_idx is None:
        rel_path = f"model/{blob_name}.bin"
    else:
        rel_path = f"layer{layer_idx}/{blob_name}.bin"
    return {
        "path": rel_path,
        "hf_name": hf_name,
        "shape": list(shape) if shape is not None else None,
        "transpose": bool(transpose),
        "alias_of": alias_of,
    }


def runtime_from_config(cfg):
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    rope_parameters = getattr(tc, "rope_parameters", {}) or {}
    return {
        "source": "text_config" if tc is not cfg else "root",
        "hidden_size": getattr(tc, "hidden_size", None),
        "num_hidden_layers": getattr(tc, "num_hidden_layers", None),
        "num_attention_heads": getattr(tc, "num_attention_heads", None),
        "num_key_value_heads": getattr(tc, "num_key_value_heads", None),
        "intermediate_size": getattr(tc, "intermediate_size", None),
        "vocab_size": getattr(tc, "vocab_size", None),
        "max_position_embeddings": getattr(tc, "max_position_embeddings", None),
        "rms_norm_eps": getattr(tc, "rms_norm_eps", None),
        "head_dim": getattr(tc, "head_dim", None),
        "tie_word_embeddings": getattr(tc, "tie_word_embeddings", None),
        "rope_parameters": rope_parameters,
        "layer_types": list(getattr(tc, "layer_types", [])),
    }


def flatten_text_config(cfg):
    tc = cfg.text_config if hasattr(cfg, "text_config") else None
    if tc is None:
        return cfg

    for key, value in tc.to_dict().items():
        if key.startswith("_"):
            continue
        setattr(cfg, key, value)
    return cfg


def build_expected_entries(cfg):
    runtime = runtime_from_config(cfg)
    layer_types = runtime["layer_types"]
    entries = [
        output_entry(None, "embed_tokens", f"{TEXT_PREFIX}.embed_tokens.weight", (runtime["vocab_size"], runtime["hidden_size"]), False),
        output_entry(None, "final_norm", f"{TEXT_PREFIX}.norm.weight", (runtime["hidden_size"],), False),
        output_entry(None, "lm_head", "lm_head.weight", (runtime["vocab_size"], runtime["hidden_size"]), False),
    ]

    for idx, layer_type in enumerate(layer_types):
        base = f"{TEXT_PREFIX}.layers.{idx}"
        entries.append(output_entry(idx, "input_layernorm", f"{base}.input_layernorm.weight", (runtime["hidden_size"],), False))
        entries.append(output_entry(idx, "post_attention_layernorm", f"{base}.post_attention_layernorm.weight", (runtime["hidden_size"],), False))
        entries.append(output_entry(idx, "mlp_gate_proj", f"{base}.mlp.gate_proj.weight", (runtime["intermediate_size"], runtime["hidden_size"]), True))
        entries.append(output_entry(idx, "mlp_up_proj", f"{base}.mlp.up_proj.weight", (runtime["intermediate_size"], runtime["hidden_size"]), True))
        entries.append(output_entry(idx, "mlp_down_proj", f"{base}.mlp.down_proj.weight", (runtime["hidden_size"], runtime["intermediate_size"]), True))

        if layer_type == "full_attention":
            entries.extend(
                [
                    output_entry(idx, "self_attn_q_proj", f"{base}.self_attn.q_proj.weight", (runtime["hidden_size"], runtime["hidden_size"]), True),
                    output_entry(idx, "self_attn_k_proj", f"{base}.self_attn.k_proj.weight", (runtime["num_key_value_heads"] * runtime["head_dim"], runtime["hidden_size"]), True),
                    output_entry(idx, "self_attn_v_proj", f"{base}.self_attn.v_proj.weight", (runtime["num_key_value_heads"] * runtime["head_dim"], runtime["hidden_size"]), True),
                    output_entry(idx, "self_attn_o_proj", f"{base}.self_attn.o_proj.weight", (runtime["hidden_size"], runtime["hidden_size"]), True),
                    output_entry(idx, "self_attn_q_norm", f"{base}.self_attn.q_norm.weight", (runtime["head_dim"],), False),
                    output_entry(idx, "self_attn_k_norm", f"{base}.self_attn.k_norm.weight", (runtime["head_dim"],), False),
                ]
            )
        elif layer_type == "linear_attention":
            entries.extend(
                [
                    output_entry(idx, "linear_attn_in_proj_qkv", f"{base}.linear_attn.in_proj_qkv.weight", None, True),
                    output_entry(idx, "linear_attn_in_proj_z", f"{base}.linear_attn.in_proj_z.weight", None, True),
                    output_entry(idx, "linear_attn_in_proj_a", f"{base}.linear_attn.in_proj_a.weight", None, True),
                    output_entry(idx, "linear_attn_in_proj_b", f"{base}.linear_attn.in_proj_b.weight", None, True),
                    output_entry(idx, "linear_attn_out_proj", f"{base}.linear_attn.out_proj.weight", None, True),
                    output_entry(idx, "linear_attn_norm", f"{base}.linear_attn.norm.weight", None, False),
                    output_entry(idx, "linear_attn_dt_bias", f"{base}.linear_attn.dt_bias", None, False),
                    output_entry(idx, "linear_attn_a_log", f"{base}.linear_attn.A_log", None, False),
                    output_entry(idx, "linear_attn_conv1d", f"{base}.linear_attn.conv1d.weight", None, False),
                ]
            )
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    return runtime, entries


def load_index(model_name):
    if is_local_model_path(model_name):
        index_path = resolve_local_model_path(model_name) / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Local model index not found: {index_path}")
    else:
        index_path = hf_hub_download(model_name, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_manifest(model_name, output_dir):
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    runtime, entries = build_expected_entries(cfg)
    index_data = load_index(model_name)
    raw_weight_map = index_data["weight_map"]
    weight_map = {normalize_index_name(k): v for k, v in raw_weight_map.items()}
    available_keys = set(weight_map.keys())
    metadata = index_data.get("metadata", {}) or {}
    total_size = metadata.get("total_size")
    missing = []
    present = []
    for entry in entries:
        if entry["hf_name"] in available_keys:
            present.append(entry)
        else:
            missing.append(entry["hf_name"])

    # Qwen3.5 ties lm_head to embed_tokens, so lm_head.weight may be absent
    # from the safetensors index even though the runtime should treat it as
    # the same tensor.
    if runtime.get("tie_word_embeddings") and "lm_head.weight" in missing:
        missing = [name for name in missing if name != "lm_head.weight"]
        present.append(
            output_entry(
                None,
                "lm_head",
                "lm_head.weight",
                (runtime["vocab_size"], runtime["hidden_size"]),
                False,
                alias_of=f"{TEXT_PREFIX}.embed_tokens.weight",
            )
        )

    manifest = {
        "status": "PASS_QWEN35_MANIFEST_ONLY" if not missing else "BLOCKED_QWEN35_MANIFEST_MISSING",
        "model": model_name,
        "runtime": runtime,
        "present_entries": present,
        "missing_hf_names": missing,
        "weight_map_shards": weight_map,
        "hf_total_size_bytes": total_size,
        "ignored_prefixes": [VISUAL_PREFIX, MTP_PREFIX],
        "notes": [
            "Stage-1 converter is text-only.",
            "Visual and mtp branches are intentionally excluded.",
            "ANE runtime support is not implied by this manifest.",
        ],
    }

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path, manifest


def apply_actual_shapes_from_state(manifest, state):
    updated_entries = []
    for entry in manifest["present_entries"]:
        tensor_name = entry["alias_of"] or entry["hf_name"]
        tensor = state[tensor_name]
        updated = dict(entry)
        updated["shape"] = list(tensor.shape)
        updated["resolved_hf_name"] = tensor_name
        updated_entries.append(updated)
    manifest["present_entries"] = updated_entries
    return manifest


def export_qwen35(model_name, output_dir):
    manifest_path, manifest = build_manifest(model_name, output_dir)
    if manifest["missing_hf_names"]:
        raise RuntimeError(f"Missing expected tensors: {manifest['missing_hf_names'][:5]}")

    print(f"Loading model {model_name} for text-only export...")
    cfg = flatten_text_config(AutoConfig.from_pretrained(model_name, trust_remote_code=True))
    model = AutoModelForCausalLM.from_pretrained(model_name, config=cfg, trust_remote_code=True)
    state = model.state_dict()
    manifest = apply_actual_shapes_from_state(manifest, state)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    total_bytes = 0
    file_count = 0
    saved = []

    for entry in manifest["present_entries"]:
        tensor_name = entry["alias_of"] or entry["hf_name"]
        tensor = state[tensor_name].detach().cpu().float().numpy()
        out_path = os.path.join(output_dir, entry["path"])
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if entry["transpose"] and tensor.ndim == 2:
            size = convert_tensor_transposed(tensor, out_path)
        else:
            size = convert_tensor_to_blob(tensor, out_path)
        saved.append({**entry, "resolved_hf_name": tensor_name, "bytes_fp16": size})
        total_bytes += size
        file_count += 1
        print(f"  {entry['path']}: {tuple(tensor.shape)} -> {size} bytes fp16")

    export_summary = {
        "status": "PASS_QWEN35_TEXT_ONLY_EXPORT",
        "model": model_name,
        "manifest_path": manifest_path,
        "file_count": file_count,
        "total_bytes_fp16": total_bytes,
        "saved_entries": saved,
    }
    summary_path = os.path.join(output_dir, "export_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(export_summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(export_summary, ensure_ascii=False, indent=2))
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3.5 text weights to Orion BLOBFILE format")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="Hugging Face model name or local export directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--manifest-only", action="store_true", help="Only write manifest.json without exporting weights")
    args = parser.parse_args()

    if args.manifest_only:
        manifest_path, manifest = build_manifest(args.model, args.output)
        print(f"MANIFEST_PATH={manifest_path}")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        export_qwen35(args.model, args.output)


if __name__ == "__main__":
    main()
