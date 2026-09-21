# Task 2: Yelp polarity sentiment classification

Poushali Purkayastha, Team 16

## What was built

Three binary sentiment classifiers trained from scratch on a 100,000-review subset of the Yelp polarity training split and evaluated on the official 38,000-review test split. No pretrained embeddings or language models are used anywhere; every model learns its own 128-dimensional word embeddings. The baseline is a mean-pooled embedding bag with an MLP head. The two experimental models change the architecture: a TextCNN with kernel widths 3, 4, and 5, and a bidirectional LSTM. The code is in src/, split into preprocess.py, models.py, train.py, evaluate_task2.py, and run.py.

Final numbers are in metrics_report.csv (long format, one row per model and metric) and outputs/full/metrics_wide.csv (models side by side). Sections marked "to fill" are completed from the final GPU run.

## Data and preprocessing

- Source: Yelp polarity on Hugging Face (fancyzhx/yelp_polarity), 560,000 training and 38,000 test reviews, labelled 0 negative and 1 positive.
- EDA on the full official splits: class counts and balance, review length in characters and words with percentiles, length by class, malformed-row checks (non-string, empty, labels outside 0 and 1, duplicate texts). Plots and numbers are in outputs/full/eda/.
- Subset: 110,000 rows drawn from the training split with seed 20, of which 100,000 are used for training and 10,000 for validation. The test split is used whole and unchanged.
- Cleaning, in order: literal "\n" and escaped quotes left over from the original CSV are replaced, text is lowercased, punctuation and special characters are removed while apostrophes inside words are kept so contractions such as "don't" survive, whitespace tokenization, stopword removal with the NLTK English list minus every negation word, Snowball stemming.
- Negations are kept on purpose: "not good" and "good" must stay distinguishable, and the standard stopword list would drop "not", "no", "never", and every "n't" form.
- Vocabulary: the 20,000 most frequent stemmed tokens in the training subset with at least 2 occurrences, plus padding at index 0 and unknown at index 1. Reviews are truncated to 256 tokens. The unknown-token rate and truncation fraction per split are in data_processed/full/meta.json.
- Data slices for robustness, computed on the raw test text: length bucket (short under 50 words, medium 50 to 150, long over 150), presence of a negation word, presence of "but", presence of an exclamation mark.

## Models and why

| Model | Architecture | Embedding choice | Why |
| --- | --- | --- | --- |
| baseline | learned embedding, masked mean pooling over tokens, Linear 128 to 128, ReLU, dropout 0.3, Linear 128 to 1 | 128-d, learned from scratch, padding index zeroed | the simplest neural model with learned embeddings; order-free bag of embeddings sets the floor and trains in seconds |
| textcnn | learned embedding, Conv1d filters of width 3, 4, 5 with 100 channels each, ReLU, max over valid positions, dropout 0.5, Linear 300 to 1 | same 128-d layer, learned separately | captures local phrases such as "not worth" or "highly recommend" that mean pooling blurs; padded positions are masked before the max so review length cannot leak in |
| bilstm | learned embedding, bidirectional LSTM with 128 hidden units per direction, final states concatenated, dropout 0.3, Linear 256 to 1 | same 128-d layer, learned separately | reads the whole review in both directions so negation scope and contrast ("great food but terrible service") can be modelled; sequences are packed so padding is never read |

Each experimental model changes the architecture while keeping the embedding size, optimizer, batch size, and epoch budget fixed, so the comparison isolates the effect of the architecture.

## Hyperparameters and why

| Hyperparameter | Value | Why |
| --- | --- | --- |
| Embedding dimension | 128 | large enough for a 20,000-word vocabulary, small enough that the embedding table stays about 2.6M parameters |
| Max tokens per review | 256 | above the 95th percentile of preprocessed review length, so few reviews are cut |
| Optimizer | Adam, learning rate 1e-3, no weight decay | the standard choice for small text models; dropout does the regularizing |
| Batch size | 256 | 391 steps per epoch on 100,000 reviews |
| Epochs | 4, best epoch kept by validation macro-F1 | these models converge within a few passes; model selection on validation keeps the test set untouched |
| Loss | binary cross-entropy on a single logit | one output is enough for two classes and gives a calibrated probability for the calibration metrics |
| Gradient clipping | 1.0 | protects the LSTM from occasional large gradients |
| Mixed precision | on GPU, bfloat16 if supported else float16 with loss scaling | faster training without changing the models |
| Seed | 20 | fixed for the subset, the shuffling, the initialization, and the bootstrap |

## How each metric is computed

| Metric | Computation |
| --- | --- |
| Accuracy, precision, recall, F1 | on the 38,000 test reviews with threshold 0.5; precision, recall, and F1 reported with macro, micro, and weighted averaging |
| Confusion matrix | TN, FP, FN, TP counts, plus a plot per model |
| ROC-AUC, PR-AUC | from the predicted positive probability; PR-AUC is average precision |
| MCC | Matthews correlation coefficient on the thresholded predictions |
| Brier score | mean squared error between the positive probability and the label |
| ECE | expected calibration error with 10 equal-width confidence bins, confidence being the probability of the predicted class |
| 95% bootstrap CIs | 1,000 resamples of the test set with replacement; 2.5th and 97.5th percentiles of accuracy, macro-F1, and MCC |
| McNemar test | paired on the test set, baseline against each experimental model, chi-square with continuity correction on the two discordant counts |
| Slice metrics | macro-F1, error rate, and support for every slice value listed above |
| Parameter count, training time, examples/sec | from the training run; training time excludes validation passes |
| Peak memory | CUDA max memory allocated on GPU, process RSS on CPU; the source is recorded |

## Error review method

evaluate_task2.py selects 20 candidates from the reviewed model's test errors: the 5 false positives with the highest positive probability, the 5 false negatives with the lowest, the 5 errors closest to the 0.5 threshold, and the 5 most confident errors inside the slice with the highest error rate among slices with at least 200 examples. The candidates with their full text are written to outputs/full/<model>/error_review_candidates.md. The manual review with error types and the proposed fix is in failure_analysis.md.

## Results

To fill from the final run: the metrics table from outputs/full/metrics_wide.csv, the training curves, confusion matrices, ROC and PR curves, and reliability diagrams from outputs/full/<model>/.

## Comparison of my three models

To fill after the final run: which architecture won on which metric, what the McNemar tests say about whether the differences are significant, where calibration differs, and which slices each model struggles with.

## Comparison with Sadaf's models

To fill after both members' runs, using the report comparison table.

## Strengths, weaknesses, limitations, future work

To fill after the final run.

## Hardware and run identity

To fill from the manifest: GPU or CPU model per model, run_id, git commit, raw log path, manifest path.

## Checkpoints

To fill from the manifest: checkpoints/full/<model>/best.pt with the epoch it was saved at.
