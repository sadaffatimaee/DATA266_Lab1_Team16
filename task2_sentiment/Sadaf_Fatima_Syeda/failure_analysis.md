## 1. False Positives (Guessed Positive, Actually Negative)

Polite intros or nice staff mentions tricked the model into high-positive scores ($p > 0.9$).

1. **Index 35** ($p = 0.9177$): Meandering polite start. *Fix:* Add end-of-text attention.
2. **Index 168** ($p = 0.9223$): Hooked on "On the plus side." *Fix:* Penalize over-reliance on transition words.
3. **Index 290** ($p = 0.9672$): Nice staff overshadowed dirty rooms. *Fix:* Give hygiene words more weight.
4. **Index 408** ($p = 0.9974$): Mentioned multiple restaurants. *Fix:* Filter out competing names.
5. **Index 415** ($p = 0.9116$): Lukewarm review mixed with great staff notes. *Fix:* Add a neutral class.

## 2. False Negatives (Guessed Negative, Actually Positive)

An initial complaint panicked the model into low-negative scores ($p < 0.1$).

1. **Index 264** ($p = 0.0010$): Ranted about delivery, but loved the restaurant. *Fix:* Weight final sentences more.
2. **Index 336** ($p = 0.0017$): One bad recent experience ruined past praise. *Fix:* Balance past vs. final feedback.
3. **Index 649** ($p = 0.0126$): Long backstory sounded like a complaint. *Fix:* Trim unnecessary backstory.
4. **Index 734** ($p = 0.0044$): Wild emotional swing from past dislike to current love. *Fix:* Use focal loss.
5. **Index 863** ($p = 0.0388$): Slang term ("credz") confused embeddings. *Fix:* Expand subword n-grams.

## 3. Near-Threshold Errors (Stuck at 50%)

A tight mix of pros and cons left the model right at 50% probability.

1. **Index 707** ($p = 0.4889$): Great food, but too far away. *Fix:* Separate food quality from distance.
2. **Index 940** ($p = 0.5020$): Educational, emotionless tone. *Fix:* Add sentiment lexicon features.
3. **Index 1938** ($p = 0.4933$): Bad windshield broke, but voucher fixed it. *Fix:* Use hierarchical layers.
4. **Index 2455** ($p = 0.5060$): Lots of exclamation points over a mild gripe. *Fix:* Normalize punctuation.
5. **Index 2593** ($p = 0.4851$): Nostalgic and hypothetical. *Fix:* Filter out speculative comments.

## 4. Length-Based Failures

* **Short Reviews (< 10 words):** Not enough context for the RNN. *(Fix: Add TF-IDF rules).*
* **Long Reviews (> 200 words):** Meaning got washed out across too many steps. *(Fix: Use chunked attention).*
