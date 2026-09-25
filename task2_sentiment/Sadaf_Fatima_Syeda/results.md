## 1. Overview

We trained and compared three binary text classification models from scratch on 100k Yelp reviews (tested on 10k reviews) using a T4 GPU. No pretrained models or weights were used.

## 2. Models

1. **Baseline:** Word embeddings + Max Pooling + Linear Classifier.
2. **BiGRU + Attention:** Bidirectional RNN + Custom Attention Layer.
3. **Transformer Encoder:** Self-Attention Layers + Mean Pooling.

## 3. Results

| Model | Accuracy | Macro-F1 | ROC-AUC | Brier Score | McNemar vs. Baseline |
| --- | --- | --- | --- | --- | --- |
| **Baseline** | 89.63% | 0.8954 | 0.9619 | 0.0768 | — |
| **BiGRU + Attention** | **93.72%** | **0.9369** | **0.9840** | **0.0475** | $p = 1.02 \times 10^{-39}$ (Significant) |
| **Transformer Encoder** | 89.94% | 0.8988 | 0.9640 | 0.0738 | $p = 0.2812$ (Not significant) |

## 4. Key Findings

* **The Winner:** The **BiGRU + Attention** model performed the best (93.72% accuracy). Statistical tests prove this improvement is real and significant.
* **Transformer Challenge:** Training a Transformer from scratch without pretraining yielded results almost identical to the simple baseline (89.94%), showing they need massive pre-existing data to excel.
* **Errors:** Most mistakes happened with reviews that had mixed feelings (e.g., a polite intro hiding a complaint) or were unusually short/long.

## 5. Next Steps

* Use subword tokenization to better handle slang and typos.
* Test chunked attention to better handle very long reviews.
