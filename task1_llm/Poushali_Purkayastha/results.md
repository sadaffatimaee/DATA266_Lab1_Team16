# Task 1: character-level GPT from scratch on TinyStories

Poushali Purkayastha, Team 16

## What was built

A decoder-only Transformer written from scratch in PyTorch: 4 pre-norm blocks, 4 attention heads, d_model 256, feed-forward width 1024, a 128-character context, learnable token and positional embeddings, and a linear language-modelling head. It is trained as a next-character predictor on 100,000 TinyStories character sequences for 10 epochs with AdamW, linear warm-up, and cosine decay. The code is in src/, split into data.py, model.py, train.py, generate.py, and run.py. Metrics are computed by src/evaluate_task1.py, the same script used in Sadaf's folder, so the two sets of numbers are comparable.

Final numbers are in metrics_report.csv and come from run `full_20260928_180129` on an NVIDIA GeForce RTX 5090 in the GPU lab (see Hardware and run identity).

## Data

- Source: TinyStories train split, streamed from Hugging Face (roneneldan/TinyStories). The first 16,786 stories in dataset order were collected, 14.9 million characters.
- Tokenization: character level. The vocabulary is built from the training stories only, plus an unknown token at index 0, and has 93 entries. char_to_idx and idx_to_char are built by hand in data.py and saved to data_processed/full/vocab.json.
- Split: stories are shuffled with seed 20 and 10% of them, 1,678 stories, go to validation; the other 15,108 form the training pool, so no story appears on both sides. Each side is concatenated with a blank line between stories and cut into non-overlapping 129-character windows. The first 100,000 training windows and 10,000 validation windows are kept, giving 12.8 million training tokens and 1.28 million validation tokens. Input is characters 1 to 128 and the target is characters 2 to 129.
- Only 2 validation characters fall outside the training vocabulary. Counts are in data_processed/full/meta.json.

## Architecture and why

| Component | Choice | Reason |
| --- | --- | --- |
| Token embedding | learnable, 93 x 256 | required by the brief; a character vocabulary is small so the table is cheap |
| Positional embedding | learnable, 128 x 256 | required by the brief; learned positions are the simplest option for a fixed context |
| Blocks | 4 pre-norm blocks | pre-norm trains stably; 4 layers keep the 10-epoch run inside one lab slot |
| Attention | 4 heads of 64 dims, causal mask from a lower-triangular buffer, written in model.py | no prebuilt attention allowed; 64-dim heads are the width GPT-2 uses |
| Feed-forward | 256 to 1024 to 256 with GELU | the 4x expansion from the Transformer paper |
| Layer norm | own implementation with learnable scale and shift | keeps the whole block from scratch |
| Residuals | around attention and feed-forward | needed for gradient flow through the stack |
| LM head | linear 256 to 93, no bias | projects the final hidden state to next-character logits |
| Dropout | 0.1 on embeddings, attention weights, and residual branches | 100K windows of 128 characters is small enough for a 3M-parameter model to overfit in 10 epochs |
| Init | normal(0, 0.02), residual projections scaled by 1/sqrt(2 x layers) | GPT-2 initialization, keeps the residual stream variance bounded |

Parameter count: 3,239,936.

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
| Mixed precision | bfloat16 autocast, which the RTX 5090 supports natively | faster training without changing the model |
| Seed | 20 | fixed for the split, the shuffling, and the sampling |

## Training procedure

run.py prepares the data if it is not cached, trains, generates, evaluates, and writes the manifest. Every step logs loss, learning rate, and gradient norm to the raw log in reproducibility/raw_logs/. After every epoch the validation loss and top-1 accuracy are computed over all 10,000 validation windows, a checkpoint is saved, and the loss curves are re-plotted. Non-finite losses are counted and skipped. A loss spike is counted when a step loss exceeds 1.5 times its exponential moving average after warm-up.

## How each metric is computed

| Metric | Computation |
| --- | --- |
| Training cross-entropy | mean of the per-step losses in the final epoch, measured with dropout active |
| Validation cross-entropy | total cross-entropy over all 1.28 million validation tokens divided by the token count, dropout off |
| Perplexity | exp(validation cross-entropy), per character |
| Bits-per-character | validation cross-entropy divided by ln 2 |
| Generalization gap | validation minus training cross-entropy |
| Top-1 next-character accuracy | fraction of validation positions where the argmax logit equals the target |
| Distinct-1/2/3 | unique word n-grams divided by total word n-grams, pooled across all 15 generated continuations |
| Repeated 4-gram rate | per sample, 1 minus unique 4-grams over total 4-grams, averaged over the 15 samples |
| Gradient norm | global norm before clipping, recorded every step; mean, max, and last reported |
| Stability | count of loss spikes and non-finite losses |
| Parameter count | sum of parameter element counts |
| Training tokens/sec | training tokens processed divided by time spent in training steps, excluding validation |
| Generation tokens/sec | generated characters divided by generation wall time, 15 samples of 400 characters in batches of 1 or 2 |
| Peak memory | CUDA max memory allocated during training |
| Training time | sum of epoch training times |

## Results

All values are from metrics_report.csv.

| Metric | Value |
| --- | --- |
| Training cross-entropy, final epoch | 0.7687 |
| Validation cross-entropy, final epoch | 0.7475 |
| Best validation cross-entropy | 0.7475, epoch 10 |
| Perplexity per character | 2.112 |
| Bits-per-character | 1.078 |
| Generalization gap | -0.021 |
| Top-1 next-character accuracy | 76.30% |
| Distinct-1 / Distinct-2 / Distinct-3 | 0.331 / 0.687 / 0.810 |
| Repeated 4-gram rate | 0.051 |
| Gradient norm mean / max / last | 0.489 / 10.45 / 0.767 |
| Loss spikes / NaN losses | 0 / 0 |
| Parameter count | 3,239,936 |
| Training tokens/sec | 921,856 |
| Generation tokens/sec | 460.7 |
| Peak memory | 1,285 MB, CUDA max memory allocated |
| Total training time | 138.9 s for 10 epochs, 7,820 optimizer steps |

Validation loss fell every epoch and was still improving slightly at epoch 10, so the best checkpoint is the final one:

| Epoch | Train loss | Val loss | Val top-1 |
| --- | --- | --- | --- |
| 1 | 1.779 | 1.094 | 65.8% |
| 2 | 1.046 | 0.919 | 71.1% |
| 3 | 0.934 | 0.860 | 72.8% |
| 5 | 0.849 | 0.802 | 74.6% |
| 7 | 0.803 | 0.769 | 75.6% |
| 10 | 0.769 | 0.748 | 76.3% |

Evidence: outputs/full/loss_curves.png (loss per epoch and per step), outputs/full/training_dynamics.png (learning-rate schedule and gradient norm per step), outputs/full/history.json (every step).

Reading the numbers:

- The generalization gap is slightly negative because the training loss is measured with dropout active and averaged over the whole final epoch, while validation runs on the finished weights without dropout. The two curves track each other closely, so there is no overfitting after 10 epochs; a longer run would still improve.
- The gradient norm maximum of 10.45 occurs at step 1, before warm-up has done anything; after warm-up it stays around 0.5 with no spikes, which is what the warm-up plus cosine schedule was chosen for.
- Distinct-2 of 0.69 and a repeated 4-gram rate of 5% show that most samples do not loop, but the greedy samples do repeat, see the failure analysis.

## Generated samples

All 15 samples are in outputs/full/samples.txt, three prompts, one greedy and two each at temperature 0.7 and 1.0, 400 characters each.

- Greedy, prompt "Once upon a time": "there was a little girl named Lily. She loved to play outside in the sunshine. One day, she went to the park with her mommy. They saw a big box with a big smile on her face." Fluent TinyStories style, correct punctuation and dialogue form, but greedy decoding falls back to the most common opening and re-starts the same story after the first paragraph.
- Temperature 0.7, prompt "The cat": "The cat was so happy that it had to be safe and sound. The cat made him feel better and he had never seen anything special." Grammatical sentences with the dataset's moral-of-the-story ending ("He learned that it's important to listen to his mom.") but characters and pronouns drift.
- Temperature 1.0, prompt "One day, a little girl": "They both laughed and clapped again. They went inside, hoping that they shared the reindeer again." More varied vocabulary, occasional invented words and dropped verbs.

Temperature 0.7 gives the best trade-off between fluency and repetition for this model; greedy repeats, temperature 1.0 breaks grammar more often.

## Failure analysis

Three cases with verbatim snippets are in failure_analysis.md: a greedy repetition loop, invented and truncated words at temperature 1.0, and a self-contradicting narrative.

## Hardware and run identity

- Device: NVIDIA GeForce RTX 5090, 32 GB, driver 610.60, CUDA 12.8, torch 2.11.0+cu128, bfloat16 autocast. Machine: Windows 11, Intel Core 24 cores, 68 GB RAM, GPU lab.
- Run id: full_20260928_180129, executed through src/task1_char_gpt.ipynb with configs/full.yaml on 2026-09-28.
- Git commit at run time: ec1788f.
- Raw log: reproducibility/raw_logs/Poushali_Purkayastha_task1_llm_full_20260928_180129.log
- Manifest: reproducibility/manifests/Poushali_Purkayastha_task1_llm_full_20260928_180129.json, which records the config, the pip freeze, and the checkpoint-to-metric mapping.

## Checkpoints

checkpoints/full/final.pt (epoch 10) produced every number in metrics_report.csv and every sample. checkpoints/full/best.pt is the same epoch, because validation loss was lowest at the end. Both hold the weights, the vocabulary, and the config, and load with src/generate.py.

## How this model differs from Sadaf's

Sadaf's model uses 6 layers, d_model 192, 6 heads, a 256-character context, ReLU, dropout 0.2, and a linear-decay schedule, trained on a Tesla T4 in Colab with batch size 32. Mine trades depth and context for width, uses GELU, dropout 0.1, and a cosine schedule with batch size 128. The largest difference is the data: I train on 100,000 windows of 128 characters, 12.8 million characters, while her run used a 100,000-character training set, so her model overfits after epoch 2 (best validation loss 1.314, final 2.609) where mine is still improving at epoch 10 (0.748). The report's comparison table sets the two side by side.
