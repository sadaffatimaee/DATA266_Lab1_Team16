# Task 1: character-level GPT from scratch on TinyStories

Poushali Purkayastha, Team 16

## What was built

A decoder-only Transformer written from scratch in PyTorch: 4 pre-norm blocks, 4 attention heads, d_model 256, feed-forward width 1024, a 128-character context, learnable token and positional embeddings, and a linear language-modelling head. It is trained as a next-character predictor on 100,000 TinyStories sequences for 10 epochs with AdamW, linear warm-up, and cosine decay. The code is in src/, split into data.py, model.py, train.py, generate.py, and run.py. Metrics are computed by the shared task1_llm/evaluate_task1.py so they are comparable with Sadaf's.

Final numbers are in metrics_report.csv. Sections marked "to fill" are completed from the final GPU run.

## Data

- Source: TinyStories train split, streamed from Hugging Face (roneneldan/TinyStories). Stories are taken in dataset order until enough characters are collected.
- Tokenization: character level. The vocabulary is built from the training stories only, plus an unknown token at index 0. char_to_idx and idx_to_char are built by hand in data.py and saved to data_processed/full/vocab.json.
- Split: stories are shuffled with seed 20 and 10% of them go to validation, so no story appears on both sides. Each side is concatenated with a blank line between stories and cut into non-overlapping 129-character windows. The first 100,000 training windows and 10,000 validation windows are kept. Input is characters 1 to 128 and the target is characters 2 to 129.
- Counts, vocabulary size, and unknown-token count for the final data: data_processed/full/meta.json.

## Architecture and why

| Component | Choice | Reason |
| --- | --- | --- |
| Token embedding | learnable, vocab x 256 | required by the brief; a character vocabulary is small so the table is cheap |
| Positional embedding | learnable, 128 x 256 | required by the brief; learned positions are the simplest option for a fixed context |
| Blocks | 4 pre-norm blocks | pre-norm trains stably; 4 layers keep the 10-epoch run inside one lab slot |
| Attention | 4 heads of 64 dims, causal mask from a lower-triangular buffer, written in model.py | no prebuilt attention allowed; 64-dim heads are the width GPT-2 uses |
| Feed-forward | 256 to 1024 to 256 with GELU | the 4x expansion from the Transformer paper |
| Layer norm | own implementation with learnable scale and shift | keeps the whole block from scratch |
| Residuals | around attention and feed-forward | needed for gradient flow through the stack |
| LM head | linear 256 to vocab, no bias | projects the final hidden state to next-character logits |
| Dropout | 0.1 on embeddings, attention weights, and residual branches | 100K windows of 128 characters is small enough for a 3M-parameter model to overfit in 10 epochs |
| Init | normal(0, 0.02), residual projections scaled by 1/sqrt(2 x layers) | GPT-2 initialization, keeps the residual stream variance bounded |

Parameter count: to fill from metrics_report.csv.

## Hyperparameters and why

| Hyperparameter | Value | Reason |
| --- | --- | --- |
| Context length | 128 | holds a sentence or two; attention cost stays small |
| Batch size | 128 sequences, 16,384 tokens per step | fills a GPU well for a model this small; 782 steps per epoch |
| Epochs | 10 | minimum set by the brief |
| Optimizer | AdamW, betas 0.9 and 0.95, weight decay 0.1 on matrices only | GPT-style defaults; decay on biases and norm parameters would hurt |
| Peak learning rate | 6e-4 | a common value for models of a few million parameters |
| Warm-up | 300 steps linear | about 4% of the 7,820 total steps; protects the early Adam updates |
| Schedule | cosine decay to 6e-5 | smooth decay gives a stable final epoch |
| Gradient clipping | 1.0 | bounds the update size if a batch produces a spike |
| Mixed precision | bfloat16 if the GPU supports it, else float16 with loss scaling | faster on GPU without changing the model |
| Seed | 20 | fixed for the split, the shuffling, and the sampling |

## Training procedure

run.py prepares the data if it is not cached, trains, generates, evaluates, and writes the manifest. Every step logs loss, learning rate, and gradient norm to the raw log in reproducibility/raw_logs/. After every epoch the validation loss and top-1 accuracy are computed over all 10,000 validation windows, a checkpoint is saved, and the loss curves are re-plotted. Non-finite losses are counted and skipped. A loss spike is counted when a step loss exceeds 1.5 times its exponential moving average after warm-up.

## How each metric is computed

| Metric | Computation |
| --- | --- |
| Training cross-entropy | mean of the per-step losses in the final epoch |
| Validation cross-entropy | total cross-entropy over all validation tokens divided by the token count |
| Perplexity | exp(validation cross-entropy) |
| Bits-per-character | validation cross-entropy divided by ln 2 |
| Generalization gap | validation minus training cross-entropy |
| Top-1 next-character accuracy | fraction of validation positions where the argmax logit equals the target |
| Distinct-1/2/3 | unique word n-grams divided by total word n-grams, pooled across all generated continuations |
| Repeated 4-gram rate | per sample, 1 minus unique 4-grams over total 4-grams, averaged over samples |
| Gradient norm | global norm before clipping, recorded every step; mean, max, and last reported |
| Stability | count of loss spikes and non-finite losses |
| Parameter count | sum of parameter element counts |
| Training tokens/sec | training tokens processed divided by time spent in training steps, excluding validation |
| Generation tokens/sec | generated characters divided by generation wall time |
| Peak memory | CUDA max memory allocated on GPU, process RSS on CPU; the source is recorded |
| Training time | sum of epoch training times |

## Results

To fill from the final run: the metrics table, the loss curves in outputs/full/loss_curves.png, and the learning-rate and gradient-norm plots in outputs/full/training_dynamics.png.

## Generated samples

To fill from outputs/full/samples.txt: one greedy and one temperature sample per prompt, with a short note on each.

## Failure analysis

Three cases with snippets are in failure_analysis.md.

## Hardware and run identity

To fill from the manifest: GPU or CPU model, run_id, git commit, raw log path, manifest path.

## Checkpoints

To fill from the manifest: which checkpoint produced the reported numbers (final.pt, epoch 10) and the best validation checkpoint if different.

## How this model differs from Sadaf's

Per the team plan, Sadaf's model uses 6 layers, d_model 192, 6 heads, a 256-character context, and a linear decay schedule. Mine trades depth and context for width and a cosine schedule. The comparison table in the report sets the two side by side; the comparison paragraph is filled after both runs.
