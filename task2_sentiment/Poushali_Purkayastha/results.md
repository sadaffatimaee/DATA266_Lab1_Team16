# Task 2: Yelp polarity sentiment classification

Poushali Purkayastha, Team 16

## What was built

Three binary sentiment classifiers trained from scratch on a 100,000-review subset of the Yelp polarity training split and evaluated on the official 38,000-review test split. No pretrained embeddings or language models are used anywhere; every model learns its own 128-dimensional word embeddings. The baseline is a mean-pooled embedding bag with an MLP head. The two experimental models change the architecture: a TextCNN with kernel widths 3, 4, and 5, and a bidirectional LSTM. The code is in src/, split into preprocess.py, models.py, train.py, evaluate_task2.py, and run.py.

Final numbers are in metrics_report.csv (long format, one row per model and metric) and outputs/full/metrics_wide.csv (models side by side). They come from run `full_20260928_180234` on an NVIDIA GeForce RTX 5090 in the GPU lab, where all three models were trained back to back so their timing is comparable.

## Data and preprocessing

- Source: Yelp polarity on Hugging Face (fancyzhx/yelp_polarity), 560,000 training and 38,000 test reviews, labelled 0 negative and 1 positive.
- EDA on the full official splits (outputs/full/eda/): both splits are exactly balanced, 280,000 per class in train and 19,000 per class in test. Review length is right-skewed: median 97 words, mean 133, 5th percentile 18, 95th percentile 372, 99th percentile 608, maximum 1,052. Negative reviews are longer than positive ones, median 112 words against 84. There are no missing, non-string, or empty texts, no labels outside {0, 1}, and no duplicate texts in either split. Plots: eda_class_distribution.png, eda_length_distribution.png, eda_length_by_class.png.
- Subset: 110,000 rows drawn from the training split with seed 20, of which 100,000 are used for training (49,555 negative, 50,445 positive) and 10,000 for validation. The test split is used whole and unchanged.
- Cleaning, in order: literal "\n" and escaped quotes left over from the original CSV are replaced, text is lowercased, punctuation and special characters are removed while apostrophes inside words are kept so contractions such as "don't" survive, whitespace tokenization, stopword removal with the NLTK English list minus every negation word, Snowball stemming.
- Negations are kept on purpose: "not good" and "good" must stay distinguishable, and the standard stopword list would drop "not", "no", "never", and every "n't" form.
- Vocabulary: the 20,000 most frequent stemmed tokens in the training subset with at least 2 occurrences, plus padding at index 0 and unknown at index 1. Reviews are truncated to 256 tokens. After cleaning a review has 69 tokens on average, 1.0% of training tokens and 1.1% of test tokens map to the unknown index, and 2.2% of reviews are truncated.
- Data slices for robustness, computed on the raw test text: length bucket (short under 50 words, 9,051 reviews; medium 50 to 150, 17,146; long over 150, 11,803), presence of a negation word (28,388 with, 9,612 without), presence of "but" (19,913 with), presence of an exclamation mark (17,636 with).

## Models and why

| Model | Architecture | Embedding choice | Why |
| --- | --- | --- | --- |
| baseline | learned embedding, masked mean pooling over tokens, Linear 128 to 128, ReLU, dropout 0.3, Linear 128 to 1 | 128-d, learned from scratch, padding index zeroed | the simplest neural model with learned embeddings; order-free bag of embeddings sets the floor and trains in seconds |
| textcnn | learned embedding, Conv1d filters of width 3, 4, 5 with 100 channels each, ReLU, max over valid positions, dropout 0.5, Linear 300 to 1 | same 128-d layer, learned separately | captures local phrases such as "not worth" or "highly recommend" that mean pooling blurs; padded positions are masked before the max so review length cannot leak in |
| bilstm | learned embedding, bidirectional LSTM with 128 hidden units per direction, final states concatenated, dropout 0.3, Linear 256 to 1 | same 128-d layer, learned separately | reads the whole review in both directions so negation scope and contrast ("great food but terrible service") can be modelled; sequences are packed so padding is never read |

Each experimental model changes the architecture while keeping the embedding size, optimizer, batch size, and epoch budget fixed, so the comparison isolates the effect of the architecture. Parameter counts: 2,576,641, 2,714,201, and 2,824,449; the 2.56M-entry embedding table dominates all three.

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
| Mixed precision | bfloat16 autocast on the RTX 5090 | faster training without changing the models |
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
| Peak memory | CUDA max memory allocated during training |

## Error review method

evaluate_task2.py selects 20 candidates from the BiLSTM's test errors: the 5 false positives with the highest positive probability, the 5 false negatives with the lowest, the 5 errors closest to the 0.5 threshold, and the 5 most confident errors inside the slice with the highest error rate among slices with at least 200 examples, which was reviews without an exclamation mark. The candidates with their full text are in outputs/full/bilstm/error_review_candidates.md. The manual review with error types and the proposed fix is in failure_analysis.md.

## Results

All values are from metrics_report.csv; the slice rows are in outputs/full/metrics_wide.csv.

| Metric | baseline | textcnn | bilstm |
| --- | --- | --- | --- |
| Accuracy | 0.9232 | 0.9251 | 0.9306 |
| Accuracy 95% CI | 0.9205 to 0.9258 | 0.9225 to 0.9279 | 0.9281 to 0.9330 |
| Precision / recall / F1, macro | 0.9232 / 0.9232 / 0.9232 | 0.9252 / 0.9251 / 0.9251 | 0.9307 / 0.9306 / 0.9305 |
| Precision / recall / F1, micro and weighted | 0.9232 | 0.9251 | 0.9306 |
| Macro-F1 95% CI | 0.9205 to 0.9258 | 0.9225 to 0.9279 | 0.9280 to 0.9330 |
| Confusion TN / FP / FN / TP | 17,477 / 1,523 / 1,395 / 17,605 | 17,724 / 1,276 / 1,571 / 17,429 | 17,494 / 1,506 / 1,133 / 17,867 |
| ROC-AUC | 0.9746 | 0.9787 | 0.9812 |
| PR-AUC | 0.9748 | 0.9791 | 0.9810 |
| MCC | 0.8464 | 0.8503 | 0.8613 |
| MCC 95% CI | 0.8409 to 0.8516 | 0.8451 to 0.8559 | 0.8563 to 0.8661 |
| Brier score | 0.0585 | 0.0551 | 0.0528 |
| ECE | 0.0064 | 0.0033 | 0.0241 |
| McNemar vs baseline, b / c / statistic / p | | 980 / 1,051 / 2.41 / 0.120 | 906 / 1,185 / 36.96 / 1.2e-9 |
| Parameter count | 2,576,641 | 2,714,201 | 2,824,449 |
| Best epoch | 3 | 4 | 4 |
| Training time | 6.8 s | 10.7 s | 93.9 s |
| Training examples/sec | 58,708 | 37,378 | 4,258 |
| Inference examples/sec | 503,211 | 204,576 | 46,287 |
| Peak memory | 317 MB | 533 MB | 568 MB |
| Device | RTX 5090 | RTX 5090 | RTX 5090 |

Per-slice macro-F1 and error rate:

| Slice | Support | baseline | textcnn | bilstm |
| --- | --- | --- | --- | --- |
| Short, under 50 words | 9,051 | 0.920 / 7.7% | 0.916 / 8.0% | 0.927 / 7.0% |
| Medium, 50 to 150 words | 17,146 | 0.924 / 7.6% | 0.928 / 7.2% | 0.934 / 6.6% |
| Long, over 150 words | 11,803 | 0.919 / 7.8% | 0.922 / 7.5% | 0.923 / 7.5% |
| Contains negation | 28,388 | 0.916 / 8.1% | 0.919 / 7.8% | 0.926 / 7.2% |
| No negation | 9,612 | 0.911 / 6.3% | 0.908 / 6.6% | 0.912 / 6.3% |
| Contains "but" | 19,913 | 0.913 / 8.6% | 0.916 / 8.3% | 0.922 / 7.8% |
| No "but" | 18,087 | 0.933 / 6.6% | 0.933 / 6.6% | 0.939 / 6.0% |
| Exclamation mark | 17,636 | 0.933 / 6.4% | 0.933 / 6.5% | 0.941 / 5.7% |
| No exclamation mark | 20,364 | 0.910 / 8.8% | 0.914 / 8.4% | 0.918 / 8.0% |

Training curves (outputs/full/<model>/curves.png, history.json): validation macro-F1 by epoch was 0.907, 0.919, 0.920, 0.919 for the baseline, 0.897, 0.914, 0.923, 0.925 for the TextCNN, and 0.914, 0.917, 0.926, 0.928 for the BiLSTM. The baseline peaked at epoch 3 and the other two at epoch 4. The BiLSTM's validation loss rose from 0.187 at epoch 3 to 0.197 at epoch 4 while its F1 still improved, which is where its higher ECE comes from. Gradient norms stayed small and stable: mean 0.10, 1.00, and 0.48, no NaN or Inf losses in any run.

Plots per model: confusion_matrix.png, roc_pr_curves.png, calibration.png in outputs/full/<model>/.

## Comparison of my three models

- Accuracy rises from 92.3% for the baseline to 92.5% for the TextCNN and 93.1% for the BiLSTM, with the same ordering in macro-F1, ROC-AUC, PR-AUC, MCC, and Brier score. The bootstrap intervals of the baseline and the BiLSTM do not overlap; those of the baseline and the TextCNN do.
- The McNemar tests confirm this. The TextCNN corrects 1,051 baseline errors but introduces 980 new ones, p 0.12, so its gain is not significant. The BiLSTM corrects 1,185 and introduces 906, p about 1e-9, a significant improvement.
- The confusion matrices show different error profiles: the TextCNN is the most conservative (fewest false positives, most false negatives), the BiLSTM cuts false negatives the most, the baseline sits in between.
- Calibration is good everywhere. The baseline and TextCNN have ECE under 0.01; the BiLSTM is a little overconfident at 0.024 because its training loss kept falling after its validation loss had bottomed out. Its Brier score is still the best, so its probabilities rank examples better even though the extremes are too confident.
- The slices explain where the BiLSTM earns its lead: reviews containing negation or "but" are the hardest for every model, and the BiLSTM gains the most on exactly those, 0.9 and 0.9 points of macro-F1 over the baseline, because it reads word order and can scope a negation or a contrast. Reviews with an exclamation mark are the easiest for all three, presumably because emphatic reviews are rarely mixed. The TextCNN is worst on short reviews, where its 3 to 5 word filters see very little.
- The cost is speed: the BiLSTM trains 14 times slower than the baseline and infers 11 times slower, for 0.7 points of accuracy.

## Comparison with Sadaf's models

Sadaf trained a max-pooled embedding baseline, a BiGRU with attention pooling, and a from-scratch Transformer encoder, all with 128-d embeddings, batch size 64, 3 epochs, lemmatization instead of stemming, and seed 266, on a Tesla T4. Her results: 90.4%, 93.4%, and 92.2% accuracy. Read together:

- Her BiGRU with attention is the best model in the team at 93.4%, 0.3 points above my BiLSTM, with a similar MCC (0.869 against 0.861) and ROC-AUC (0.984 against 0.981). Attention pooling over all positions seems to help slightly over using only the final states.
- My mean-pooling baseline with an MLP head (92.3%) is clearly stronger than her max-pooling linear baseline (90.4%), so the choice of pooling and the hidden layer matter more than expected for a bag-of-embeddings model.
- Her Transformer encoder (92.2%) lands below both recurrent models, consistent with the usual finding that small from-scratch Transformers need more data than 100K reviews to beat RNNs.
- Both members see the same pattern on slices: negation and long reviews are hardest, and the recurrent models close the gap there. Sadaf also measured a truncated-review slice, where her Transformer wins, which suggests the recurrent models lose information when the end of a long review is cut off.

## Strengths, weaknesses, limitations, future work

- Strengths: a simple stemmed bag of words already reaches 92.3%, all models are well calibrated, and the full metric suite with bootstrap intervals and paired tests makes the ranking defensible rather than anecdotal.
- Weaknesses: the error review shows that the remaining errors are mostly mixed reviews whose verdict comes late, sarcasm, and a visible share of label noise; none of the three architectures handles a late reversal such as "the food was good, but I will not come back" reliably.
- Limitations: 100K of the 560K training reviews were used to fit the lab slot; stopword removal discards intensifiers such as "very" and "too"; 256-token truncation cuts 2% of reviews; the models see stems, not words, so slang and puns are out of vocabulary.
- Future work: train on the full 560K reviews, add attention pooling to the BiLSTM as Sadaf's result suggests, keep intensifiers in the vocabulary, and test the recency-weighted pooling proposed in failure_analysis.md on the "but" slice.

## Hardware and run identity

- Device: NVIDIA GeForce RTX 5090, 32 GB, CUDA 12.8, torch 2.11.0+cu128, bfloat16 autocast. Machine: Windows 11, 24-core Intel CPU, 68 GB RAM, GPU lab. All three models trained in the same run, back to back, so their times are directly comparable.
- Run id: full_20260928_180234, executed through src/task2_sentiment.ipynb with configs/full.yaml on 2026-09-28.
- Git commit at run time: ec1788f.
- Raw log: reproducibility/raw_logs/Poushali_Purkayastha_task2_sentiment_full_20260928_180234.log
- Manifest: reproducibility/manifests/Poushali_Purkayastha_task2_sentiment_full_20260928_180234.json
- An earlier CPU run of the baseline alone on 2026-09-21 (log Poushali_Purkayastha_task2_sentiment_full_20260921_161720.log) was a pipeline check on my laptop and is superseded by the GPU run.

## Checkpoints

checkpoints/full/baseline/best.pt (epoch 3), checkpoints/full/textcnn/best.pt (epoch 4), and checkpoints/full/bilstm/best.pt (epoch 4) produced the test probabilities in outputs/full/<model>/test_probs.npy from which every metric is computed.
