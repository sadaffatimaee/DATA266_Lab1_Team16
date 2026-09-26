# Task 2 Failure Analysis (Sadaf Fatima Syeda)

Model reviewed: BiGRU + attention (my best model). Twenty errors from the 38K test set were taken from `outputs/error_review_BiGRU_Attention.csv`. Label 1 = positive, 0 = negative. `p` = predicted probability of positive.

## 1. Confident False Positives (true negative, predicted positive)

| # | Snippet | p | Error type | Proposed fix |
|---:|---|---:|---|---|
| 1 | "Wow love the place and everything is very clean and new! Great place to come and relax worth a try!" | 1.000 | Positive lexical cues or possible label noise | Use more balanced examples and inspect similar training reviews. |
| 2 | "The service was awesome; space impressive. But... that NYC gem looses something and just isn't the same." | 0.9998 | Contrast and negation: positive words are followed by a negative conclusion | Add contrastive examples using words such as "but," "however," and "isn't." |
| 3 | "20 years for me and sad to see them go... consistently great fresh fish and great service... END OF AN ERA." | 0.9998 | Mixed sentiment and nostalgic wording | Train on more examples where praise and disappointment occur together. |
| 4 | "The food is a tad better than okay... The manager is awesome... five different music styles... nerve wrecking." | 0.9996 | Mixed sentiment with several positive aspects and a negative final judgment | Use attention or pooling that gives more weight to the overall conclusion. |
| 5 | "I absolutely love how it is decorated... The gel manicure itself wasn't done well at all." | 0.9989 | Aspect-level sentiment: positive comments about the decoration but negative comments about the manicure | Add aspect-aware sentiment examples that separate opinions about different features. |

## 2. Confident False Negatives (true positive, predicted negative)

| # | Snippet | p | Error type | Proposed fix |
|---:|---|---:|---|---|
| 6 | "The beer is Delicious and so is the food - However I unfortunately, can not say the same about the HELP. The service was terrible" | 0.0001 | Negation and contrast: positive food comments are followed by strongly negative service comments | Add more training examples containing "however," "not," and "terrible." |
| 7 | "This review is for the pharmacy only... Price difference is unbelievable!! ...No wonder premiums are high" | 0.0001 | Negative-sounding words ("unbelievable," "premiums are high") in a positive review about low prices | Include more examples of positive reviews that use negative-sounding or domain-specific language. |
| 8 | "Pizza is great. However, the service is not... UPDATE: service has improved drastically." | 0.0002 | Temporal update: the review describes an earlier problem followed by later improvement | Add examples with updates and temporal changes in sentiment. |
| 9 | "The children were running in and out of the bar... One father even started swearing at her... the bartender was very cordial" | 0.0002 | Long mixed-sentiment review where the negative events dominate the meaning | Use longer context or hierarchical modeling to capture the main complaint. |
| 10 | "This place is so much better since they changed owners... it was terrible... Now its much better." | 0.0006 | Temporal contrast between past negative and current positive experiences | Add examples that distinguish past and present sentiment. |

## 3. Near-Threshold Errors (true negative, p close to 0.5)

| # | Snippet | p | Error type | Proposed fix |
|---:|---|---:|---|---|
| 11 | "This place was horrible... Insipid, over-battered... On the bright side, service was very fast." | 0.5006 | Mixed sentiment: strong negative criticism combined with positive service comments | Train with more mixed-review examples and use aspect-level attention. |
| 12 | "It was total bootleg up in this location... So glad there are other locations of Ross stores near by offering the same great deals." | 0.5006 | Sarcasm and indirect sentiment | Add sarcastic and humorous reviews to the training data. |
| 13 | "The first time... was a 3-star experience... Last week's experience was a real let down... the setting is very pretty and the service is attentive." | 0.5007 | Conflicting evidence across multiple visits and review details | Use temporal and aspect-aware examples during training. |
| 14 | "Walmart makes me cringe... But... The pharmacy staff have always been pleasant and efficient." | 0.5008 | Mixed sentiment across different parts of the business | Separate sentiment by aspect, such as the store and pharmacy. |
| 15 | "Fast no wait seating on a Saturday night ...but be prepared to pay for substandard food!" | 0.5012 | Contrast between quick service and poor food quality | Add contrastive examples where one positive aspect does not outweigh the main complaint. |

## 4. Slice-Specific Errors: Truncated Reviews (> 256 tokens)

| # | Snippet | True / Pred | p | Error type | Proposed fix |
|---:|---|---|---:|---|---|
| 16 | Hotel review: long list of problems, ends "The entire property is nice, which I would stay there again" | 1 / 0 | 0.0033 | Important negative details are followed by a positive ending that was cut off by truncation | Increase the context length or use hierarchical processing. |
| 17 | Long humorous casino story with very little about the venue itself | 1 / 0 | 0.0063 | Long humorous story with weak direct sentiment cues | Add longer humorous reviews and use document-level context. |
| 18 | Waste facility visit: "the place stinks... bins were full", ends "Can't beat that!" | 1 / 0 | 0.0072 | Negative details are mixed with a positive or humorous ending | Preserve the full review or add examples with mixed conclusions. |
| 19 | Show review: "worst show you will have ever seen" jokes, ends "top 2-3 shows in town... Totally would recommend it" | 1 / 0 | 0.0076 | Sarcasm and joke-negative wording ("worst show") hide the positive meaning | Add sarcastic reviews and use a longer context window. |
| 20 | Restaurant review: first visit great, then "concept change" disappointed, "it will probably be the last" | 0 / 1 | 0.9899 | Long review contains both positive and negative visits, and the detailed positive descriptions outweigh the negative judgment | Increase the maximum sequence length and include temporal review examples. |

## Summary

The most common error types were mixed sentiment, negation or contrast, temporal updates, sarcasm, and loss of context in long reviews. The most useful fix would be to increase the context length or use hierarchical modeling so the model can consider the entire review. I would test this by training the BiGRU with a longer context window and comparing overall accuracy and the truncated-review slice metrics against the current model.
