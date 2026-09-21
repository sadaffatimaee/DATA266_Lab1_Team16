# Task 2: shared Yelp polarity dataset

The dataset is the official Yelp polarity release on Hugging Face: [fancyzhx/yelp_polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity). It has 560,000 training reviews and 38,000 test reviews, labelled 0 for negative and 1 for positive, balanced across the two classes.

The raw files are not committed. Each member's preprocessing script downloads the dataset with the `datasets` library on first use (about 200 MB, cached by Hugging Face) and writes its own processed tensors under the member's data_processed folder. The official test split is used unchanged by both members so results are comparable.
