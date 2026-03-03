# Downloading Model Weights

Weight blobs are not committed to this repo. Use the scripts below to obtain and convert them.

## GPT-2 124M

```bash
# Convert from HuggingFace (downloads automatically)
python model/convert/hf_to_blobs_gpt2.py --output model/blobs/gpt2_124m/
```

## Stories110M (Llama2-style)

```bash
# Option 1: Karpathy's pre-trained TinyLlama weights
mkdir -p model/weights
curl -L -o model/weights/stories110M.bin \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin

# Option 2: Convert from HuggingFace
python model/convert/hf_to_blobs_llama.py \
  --checkpoint model/weights/stories110M.bin \
  --output model/blobs/stories110m/
```

## Training Data (TinyStories)

```bash
# Pretokenized TinyStories (~41MB, ~20M tokens)
bash scripts/download_data.sh
```
