# Task 2.2: manual error review

Poushali Purkayastha, Team 16

Model reviewed: the BiLSTM, checkpoints/full/bilstm/best.pt (epoch 4), from run full_20260928_180234. Candidates were selected by src/evaluate_task2.py from its 2,639 test-set errors and are listed with their full text in outputs/full/bilstm/error_review_candidates.md. Each entry below quotes the review, gives the label, the prediction, the positive probability, and the error type I assigned after reading the whole review.

Error types used: mixed sentiment (praise and complaint in one review, verdict decided by weight or by the last sentence), sarcasm or irony, negation or contrast scope, temporal reversal (past bad, now good), likely label noise (the text does not support the label), figurative language or slang, numeric cue lost in preprocessing, narrative with weak sentiment cues.

## Confident false positives (label negative, predicted positive)

1. Test index 25815. p(positive) 1.000. Snippet: "The TrimTini I had was really delicious... Everything he brought was delicious... Urban 7 isn't the type of place I'm attracted to, so it's not likely I will return any time soon. But it was a nice event, and I enjoyed myself." Error type: mixed sentiment. Note: almost every sentence is positive; the only negative content is "not likely I will return", and the review ends on "enjoyed myself". The label is defensible but the text is not.
2. Test index 17756. p(positive) 1.000. Snippet: "I absolutely love how it is decorated... The gel manicure itself wasn't done well at all. You could see through my nails... I'm sure ill be back... But I can find better service and product for cheaper." Error type: mixed sentiment. Note: the decisive complaint is one sentence in the middle, surrounded by praise for decor, staff, and a promise to return.
3. Test index 9609. p(positive) 1.000. Snippet: "Sure it has the Eiffel Tower... one of my favorite restaurants... a terrific buffet... But the hotel and casino is not good for one reason only -- SMOKERS... If you are a SMOKER, then it's great. Highly recommended." Error type: sarcasm or irony. Note: the closing "Highly recommended" is ironic and the concessive "Sure it has..." structure front-loads positives that the model takes at face value.
4. Test index 8945. p(positive) 1.000. Snippet: "Bianco's is the best marketing ploy in the city... I love Bianco's pizza. It truly reminds me of eating in Naples... Do I have to remind you that you have to arrive at Bianco's an hour before the restaurant even opens to wait in line..." Error type: sarcasm or irony. Note: a 434-word essay whose complaint is the 3-hour wait, wrapped in praise and rhetorical questions; the stemmed bag of tokens is dominated by positive words.
5. Test index 30899. p(positive) 1.000. Snippet: "the room was exactly what we needed, bed, clean bathroom... our neighbors were literally partying until 8am, so as drunk as we were, we did not get any sleep... Besides that the room was nice and would've been perfect if we were able to get sleep." Error type: mixed sentiment. Note: the reviewer rates the night negatively for noise but describes the room positively three times; the model weighs the description.

## Confident false negatives (label positive, predicted negative)

6. Test index 22807. p(positive) 0.000. Snippet: "EDIT: They really did change the service up since I last posted this. Horrible service... it doesn't give you any excuse to disrespect your paying customers like that." Error type: likely label noise. Note: the edited text is unambiguously negative; the star rating was not updated when the review was edited. The model is right about the text.
7. Test index 30793. p(positive) 0.000. Snippet: "This place is so much better since they changed owners... it was terrible. We waited forever and the food never came... It was horrible. Now its much better. The staff are very friendly..." Error type: temporal reversal. Note: half the review describes the old owners in strong negative words; the verdict depends on "since" and "now", which stopword removal deletes.
8. Test index 11401. p(positive) 0.000. Snippet: "Perhaps my expectations were too high... I was a little disappointed... waited forever... wrong food and then cold food... pretty rockin. My suggestion is head there for drinks but not for food." Error type: mixed sentiment. Note: a three-star style review with more complaint than praise; the positive label is generous, but the reviewer does recommend the drinks.
9. Test index 19296. p(positive) 0.000. Snippet: "Tofu, tofu and more tofu... ALL you can eat!... the sweet and sour beef looked like fried spam and didn't taste much better... green tea ice cream... is soooo ewwwwww!!... This place is good for groups, but service is slow." Error type: mixed sentiment. Note: an itemized list where the negatives are vivid and the positives are flat ("is great", "good for groups"); the vivid negatives win in the model.
10. Test index 30366. p(positive) 0.000. Snippet: "there was no acknowledgement of a birthday... Called the manager this morning who apologized but absolutely no offer of a gift card... That was not good business." Error type: likely label noise. Note: the entire review is a complaint that ends "That was not good business"; the positive label does not fit the text.

## Near-threshold errors (probability closest to 0.5)

11. Test index 32036. p(positive) 0.500. Snippet: "This place is not only not open during business hours but looks like it's shuttered it's doors for good..." Error type: negation or contrast scope. Note: 19 words, a double negation, and no sentiment vocabulary at all; after stopword removal little is left except "open", "business", "shuttered", "doors".
12. Test index 37885. p(positive) 0.499, label positive. Snippet: "Compare to the rest of the nordy rack stores, this one by far has no good items. Couldn't find anything good here." Error type: likely label noise. Note: the text is negative and the model leaned negative; the positive label is wrong.
13. Test index 20958. p(positive) 0.501. Snippet: "we realized there was a fire... a lady took our number and said she will take care of everything... we never got anything in the mail. It does not mean i stop eating at Chipotle, we love it. But till the time our issue is resolved we will not go to This Chipotle." Error type: narrative with weak sentiment cues. Note: 315 words of incident narration with almost no evaluative words; "we love it" and "we will not go" cancel out.
14. Test index 10954. p(positive) 0.502. Snippet: "I would like to give this place 1 star only but the fact that the bathroom smells like PURE FECES makes me not love this place more... Love that it's 24 hours... He should be FIRED!... If they decrease my gym membership then I'll give them one more star :)" Error type: sarcasm or irony. Note: the review mixes genuine praise with mock praise and a smiley; even a human needs the whole text to settle on negative.
15. Test index 11883. p(positive) 0.498, label positive. Snippet: "This museum is a BLAST! It's THE BOMB!... how can this place have 19 reviews and I am the first one to use those lame puns?... you will enjoy this place." Error type: figurative language or slang. Note: "blast", "bomb", and "lame" are negative in their literal senses; the pun is the point and the model has no way to see it.

## Slice-specific failures

Slice: reviews without an exclamation mark, the slice with the highest BiLSTM error rate among slices with at least 200 examples, 8.0% against 5.7% for reviews with one. These are the calmer, more measured reviews, and their verdicts hinge on qualifiers rather than emphasis.

16. Test index 25701. p(positive) 1.000, label negative. Snippet: "the gorditas were filling and good.....but not great. Good for a quick stop, but there are much better options... The service was very friendly and the restaurant spotlessly clean." Error type: mixed sentiment. Note: the negative verdict lives entirely in "not great" and "much better options"; the rest is praise.
17. Test index 15988. p(positive) 1.000, label negative. Snippet: "I find their meats to be generally good, and reasonably generous... the conclusion I've always had was of an enjoyable adventure. Prices are downright cheap." Error type: likely label noise. Note: a measured, positive essay about a Pittsburgh institution with one reservation about fries on a sandwich; the negative label is hard to justify from the text.
18. Test index 28036. p(positive) 1.000, label negative. Snippet: "I tried out Tavern Grille and loved it... A patron makes a special request for pickleS and you provide one pickle on a plate? Amazing... Better than freshly made sliders without pickles. Ahhhhhhhhhhhhhhhhh." Error type: sarcasm or irony. Note: the reviewer's anger is expressed through "Amazing", rhetorical questions, and the long "Ahhhh", none of which the token model can read as negative.
19. Test index 4488. p(positive) 1.000, label negative. Snippet: "It's good. The rolls are better than the sashimi... Good atmosphere and my girl likes the martinis. Little Tokyo in Mt. Lebanon and Kiku in Station Square are better... 4.5 of 10 relative to sushi quality alone." Error type: numeric cue lost in preprocessing. Note: the explicit score "4.5 of 10" is the clearest negative signal and survives cleaning only as the tokens "4", "5", "10", which carry no learned polarity; the comparison to two better restaurants is also lost.
20. Test index 4806. p(positive) 0.000, label positive. Snippet: "I usually associate anything with the government or DMV as a huge pain is the @$$, but this was extremely fast and easy... There is no doubt that I will go back here for my next inspection." Error type: negation or contrast scope. Note: "huge pain", "government", "DMV" set a negative tone that the "but" reverses, and "no doubt" reads as a negation of "doubt"; the model scores the negatives.

## Error type summary

| Error type | Cases | Count |
| --- | --- | --- |
| Mixed sentiment with a late or buried verdict | 1, 2, 5, 8, 9, 16 | 6 |
| Sarcasm or irony | 3, 4, 14, 18 | 4 |
| Likely label noise | 6, 10, 12, 17 | 4 |
| Negation or contrast scope | 11, 20 | 2 |
| Temporal reversal | 7 | 1 |
| Figurative language or slang | 15 | 1 |
| Numeric cue lost in preprocessing | 19 | 1 |
| Narrative with weak sentiment cues | 13 | 1 |

Two things stand out. First, 4 of the 20 reviewed errors look like wrong labels, which means a share of the remaining 7% error rate is not recoverable by any model and the practical ceiling is below 100%. Second, 9 of the 20 involve a contrast or a reversal ("but", "since", "now", a late verdict), which matches the slice results: the "but" slice has the highest error rate among the content slices for all three models.

## One testable fix

Fix: give the BiLSTM an explicit view of the end of the review, where the verdict usually sits. Concatenate its two final hidden states with a max-pool over the outputs of the last 32 valid tokens, so the classifier sees both the whole-review summary and the closing sentences, and retrain with the same seed, data, and 4-epoch budget. Keep "but", "since", and "now" out of the stopword list at the same time, because two of the reviewed cases lost their pivot word in preprocessing.

Test: compare the new model against the current BiLSTM on the same 38,000 test reviews. Success means macro-F1 on the "contains but" slice rises by at least 0.5 points from 0.922, macro-F1 on the "no but" slice does not fall, and the paired McNemar test between the two models is significant at p below 0.05. If the gain appears only on the "but" slice, the fix worked for the reason claimed; if it appears everywhere, the extra capacity rather than the recency view is responsible, and the ablation with the stopword change alone separates the two.
