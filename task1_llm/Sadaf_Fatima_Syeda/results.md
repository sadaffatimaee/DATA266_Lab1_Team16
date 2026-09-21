# Task 1 Results: GPT-Style LLM from Scratch

## 1. Architecture & Hyperparameters
- **Model Type:** Scratch Character-Level Transformer GPT
- **Embedding Dimension (`n_embd`):** 64
- **Number of Layers (`n_layer`):** 4
- **Number of Attention Heads (`n_head`):** 4
- **Block Size (`block_size`):** 32
- **Batch Size (`batch_size`):** 32
- **Dropout Rate:** 0.1
- **Total Parameter Count:** 209,729

## 2. Training Details
- **Dataset:** TinyStories / Character-level split (100K train tokens, 10K validation tokens)
- **Epochs:** 10
- **Optimizer:** AdamW (Learning Rate: 3e-4)

## 3. Final Evaluation Metrics Summary
- **Training Cross-Entropy Loss:** ~3.485
- **Validation Cross-Entropy Loss:** ~3.489
- **Perplexity:** ~44.96
- **Bits-Per-Character (BPC):** ~5.49
- **Generalization Gap:** ~ -0.0036
- **Top-1 Next-Character Accuracy:** ~21.00%
- **Distinct-1 / Distinct-2 / Distinct-3:** 0.30 / 0.95 / 1.00
- **Repeated 4-Gram Rate:** 0.00%