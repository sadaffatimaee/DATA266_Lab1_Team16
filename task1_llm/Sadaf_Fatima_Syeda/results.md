# Task 1 Results

## Data

- Dataset: TinyStories, character level
- Vocabulary size: 91
- Training data: 100K characters
- Validation data: 10K characters
- Sequence length: 256 characters

## Model

| Setting | Value |
|---|---:|
| Layers / Heads / Embedding | 6 / 6 / 192 |
| Context length | 256 |
| Activation | ReLU |
| Dropout | 0.2 |
| Parameters | 2.75M |

## Why

I chose 6 layers and 192 dimensions because they provide enough capacity while keeping the model small and efficient. I used dropout 0.2 because it helps reduce overfitting.

## Training

- Optimizer: AdamW
- Learning rate: 6e-4
- Schedule: Warm-up followed by linear decay
- Batch size: 32
- Training duration: 10 epochs
- Gradient clipping: 1.0
- Hardware: Tesla T4 GPU on Colab
- Training time: 11 minutes

## Results

| Metric | Value |
|---|---:|
| Best validation loss | 1.314 at epoch 2 |
| Final train / validation loss | 0.055 / 2.609 |
| Perplexity | 13.58 |
| Bits per character | 3.76 |
| Top-1 accuracy | Not reported |
| Distinct-1 / 2 / 3 | Not reported |
| Repeated 4-gram rate | Not reported |
| Loss spikes / NaNs | 0 / 0 |
| Throughput | 95,654 tokens/sec |
| Peak memory | 1,273 MB |

## What I Found

Validation loss was lowest at epoch 2, and after that it increased. This shows the model is overfitting because training loss continued to decrease while validation loss worsened. So I use the best model checkpoint from epoch 2 as my final model.

## Files

- Best model: `checkpoints/best_model.pt`
- Log: `reproducibility/raw_logs/task1_sadaf_20260926_050656.log`
