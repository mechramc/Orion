#!/usr/bin/env python3
"""Generate golden test vectors for CPU GPT-2 forward pass.

Runs HuggingFace GPT-2 on test prompts and saves logits for validation.
"""
import json
import numpy as np

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:
    print("Error: pip install torch transformers")
    exit(1)

def main():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    # Test cases: token sequences with known outputs
    test_cases = [
        # "Hello" = [15496]
        {"name": "single_token", "tokens": [15496]},
        # "The" = [464]
        {"name": "the", "tokens": [464]},
        # "Hello, world" = roughly [15496, 11, 995]
        {"name": "hello_world", "tokens": [15496, 11, 995]},
        # "The quick brown fox" - will get exact tokens from tokenizer
    ]

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Add tokenized test case
    prompt = "The quick brown fox"
    toks = tokenizer.encode(prompt)
    test_cases.append({"name": "quick_brown_fox", "tokens": toks})

    results = []
    for tc in test_cases:
        tokens = tc["tokens"]
        with torch.no_grad():
            input_ids = torch.tensor([tokens], dtype=torch.long)
            outputs = model(input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab]

        last_logits = logits[-1].numpy()  # Last position
        top5_idx = np.argsort(last_logits)[-5:][::-1].tolist()
        top5_val = [float(last_logits[i]) for i in top5_idx]

        result = {
            "name": tc["name"],
            "tokens": tokens,
            "top5_indices": top5_idx,
            "top5_logits": top5_val,
            "argmax_token": int(np.argmax(last_logits)),
            # Save first 10 logits for detailed comparison
            "first_10_logits": [float(x) for x in last_logits[:10]],
        }

        # For single_token, also save embedding check
        if tc["name"] == "single_token":
            # Get embedding for token 15496
            wte = model.transformer.wte.weight.detach().numpy()
            wpe = model.transformer.wpe.weight.detach().numpy()
            emb = wte[15496] + wpe[0]
            result["embedding_first5"] = [float(x) for x in emb[:5]]

        results.append(result)
        print(f"{tc['name']}: tokens={tokens} → argmax={result['argmax_token']}, "
              f"top5={top5_idx}")

    with open("tests/forward_golden.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} test vectors to tests/forward_golden.json")

if __name__ == "__main__":
    main()
