#!/usr/bin/env python3
"""T029: Generate 20 golden test vectors for GPT-2 BPE tokenizer using tiktoken."""
import json

try:
    import tiktoken
except ImportError:
    print("Installing tiktoken...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken"])
    import tiktoken

enc = tiktoken.get_encoding("gpt2")

test_prompts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "GPT-2 is a transformer model.",
    " Hello",
    "Hello  ",
    "  ",
    "123456789",
    "café",
    "naïve",
    "don't",
    "I'm happy",
    "it's a test",
    "\n\n",
    "Hello\nWorld",
    "foo bar baz",
    "A",
    "The",
    "a b c d e",
    "$100.00",
    "C++ programming is fun!",
]

results = []
for prompt in test_prompts:
    tokens = enc.encode(prompt)
    decoded = enc.decode(tokens)
    results.append({
        "text": prompt,
        "tokens": tokens,
        "decoded": decoded,
    })
    print(f"  {repr(prompt):40s} → {tokens}")

with open("tests/tokenizer_golden.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} test vectors to tests/tokenizer_golden.json")
