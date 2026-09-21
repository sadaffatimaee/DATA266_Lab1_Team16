# Task 2.2: manual error review

Poushali Purkayastha, Team 16

Model reviewed: to fill (checkpoint path and epoch). Candidates were selected by src/evaluate_task2.py from the test-set errors and are listed with their full text in outputs/full/<model>/error_review_candidates.md. Each entry below quotes the review snippet, gives the label, the prediction, the positive probability, and the error type I assigned.

Error types used: mixed sentiment, sarcasm or irony, negation scope, comparison to another business, neutral text with an extreme label, likely label noise, key sentiment after the 256-token cut, cue lost in preprocessing, rare or out-of-vocabulary cue, too short to judge.

## Confident false positives (label negative, predicted positive)

1. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.
2. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.
3. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.
4. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.
5. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.

## Confident false negatives (label positive, predicted negative)

1. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.
2. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.
3. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.
4. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.
5. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.

## Near-threshold errors (probability closest to 0.5)

1. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.
2. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.
3. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.
4. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.
5. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.

## Slice-specific failures

Slice: to fill (the slice with the highest error rate, named in the candidates file).

1. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.
2. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.
3. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.
4. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.
5. Test index: to fill. p(positive): to fill. Snippet: to fill. Error type: to fill. Note: to fill.

## Error type summary

To fill: a count of each error type across the 20 cases.

## One testable fix

To fill: the single change I would make, the error type it targets, and how I would test it (which metric on which slice should move, and by roughly how much).
