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

Metric,Value
Training Cross-Entropy Loss,0.0553982596939955
Validation Cross-Entropy Loss,2.6086549758911133
Best Validation Loss,1.3140264749526978
Perplexity,13.580772096016522
Bits-Per-Character (BPC),3.7634935971084276
Generalization Gap (Val - Train),2.553256716197118
Top-1 Next-Character Accuracy,0.6420272435897436
Distinct-1 (temp sampling),0.07666666666666666
Distinct-2 (temp sampling),0.36740146960587844
Distinct-3 (temp sampling),0.6298527443105756
Repeated 4-gram Rate (temp sampling),0.2293762575452716
Repeated 4-gram Rate (greedy),0.2857142857142857
Gradient Norm Mean,2.3091898074478676
Gradient Norm Max,6.027122497558594
Gradient Norm Last,2.321993350982666
Loss Spikes,0
NaN/Inf Losses,0
Parameter Count,2750299
Training Throughput (tokens/sec),95654.1843507542
Generation Speed (tokens/sec),59.73162349732671
Peak Memory Usage (MB),1273.18994140625
Total Training Time (sec),668.0063233375549
Epochs,10
Optimizer Steps,7800
Device,Tesla T4

## What I Found

Validation loss was lowest at epoch 2, and after that it increased. This shows the model is overfitting because training loss continued to decrease while validation loss worsened. So I use the best model checkpoint from epoch 2 as my final model.

## Files

- Best model: `checkpoints/best_model.pt`
- Log: `reproducibility/raw_logs/task1_sadaf_20260926_050656.log`
