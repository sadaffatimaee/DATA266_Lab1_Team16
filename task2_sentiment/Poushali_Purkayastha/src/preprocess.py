import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem.snowball import SnowballStemmer

PAD_IDX = 0
UNK_IDX = 1
SLICE_COLUMNS = ["length_bucket", "has_negation", "has_but", "has_exclamation"]

STOPWORDS = set(
    "i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself yourselves he him his "
    "himself she she's her hers herself it it's its itself they them their theirs themselves what which who whom this "
    "that that'll these those am is are was were be been being have has had having do does did doing a an the and but "
    "if or because as until while of at by for with about against between into through during before after above below "
    "to from up down in out on off over under again further then once here there when where why how all any both each "
    "few more most other some such no nor not only own same so than too very s t can will just don don't should "
    "should've now d ll m o re ve y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't "
    "haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't shouldn shouldn't wasn wasn't "
    "weren weren't won won't wouldn wouldn't".split()
)
NEGATION_WORDS = {"no", "nor", "not", "never", "nothing", "nobody", "none", "neither", "nowhere", "cannot"}
CLEAN_RE = re.compile(r"[^a-z0-9'\s]+")
NEGATION_RE = re.compile(r"\b(?:no|nor|not|never|nothing|nobody|none|neither|nowhere|cannot)\b|n't\b")


def is_negation(token):
    return token in NEGATION_WORDS or token.endswith("n't")


def normalize(text):
    text = text.replace("\\n", " ").replace('\\"', '"').lower()
    return CLEAN_RE.sub(" ", text)


class Preprocessor:
    def __init__(self, remove_stopwords, keep_negations, stem):
        self.remove_stopwords = remove_stopwords
        self.stemmer = SnowballStemmer("english") if stem else None
        self.cache = {}
        drop = set(STOPWORDS)
        if keep_negations:
            drop = {w for w in drop if not is_negation(w)}
        self.drop = drop

    def stem(self, token):
        out = self.cache.get(token)
        if out is None:
            out = self.stemmer.stem(token)
            self.cache[token] = out
        return out

    def __call__(self, text):
        tokens = []
        for tok in normalize(text).split():
            tok = tok.strip("'")
            if not tok:
                continue
            if self.remove_stopwords and tok in self.drop:
                continue
            if self.stemmer is not None:
                tok = self.stem(tok)
            tokens.append(tok)
        return tokens


def slice_features(text):
    raw = text.replace("\\n", " ")
    low = raw.lower()
    n_words = len(raw.split())
    if n_words < 50:
        bucket = "short"
    elif n_words <= 150:
        bucket = "medium"
    else:
        bucket = "long"
    return {
        "n_words": n_words,
        "length_bucket": bucket,
        "has_negation": int(bool(NEGATION_RE.search(low))),
        "has_but": int(" but " in f" {low} "),
        "has_exclamation": int("!" in raw),
    }


def percentiles(values):
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": int(values.min()),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
    }


def split_stats(split, rows):
    n = len(split) if rows is None else min(int(rows), len(split))
    sub = split.select(range(n))
    labels = np.asarray(sub["label"], dtype=np.int64)
    texts = list(sub["text"])
    malformed = {"missing_or_non_string": 0, "empty_or_whitespace": 0}
    chars = np.zeros(n, dtype=np.int64)
    words = np.zeros(n, dtype=np.int64)
    seen = set()
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            malformed["missing_or_non_string"] += 1
            continue
        if not t.strip():
            malformed["empty_or_whitespace"] += 1
        chars[i] = len(t)
        words[i] = len(t.split())
        seen.add(hash(t))
    malformed["labels_outside_0_1"] = int(((labels != 0) & (labels != 1)).sum())
    malformed["duplicate_texts"] = int(n - len(seen))
    stats = {
        "rows": int(n),
        "class_counts": {"negative": int((labels == 0).sum()), "positive": int((labels == 1).sum())},
        "positive_fraction": float((labels == 1).mean()),
        "chars": percentiles(chars),
        "words": percentiles(words),
        "words_by_class": {
            "negative": percentiles(words[labels == 0]),
            "positive": percentiles(words[labels == 1]),
        },
        "malformed": malformed,
    }
    return stats, words, labels


def eda_plots(train_words, train_labels, test_words, test_labels, eda_dir):
    classes = ["negative", "positive"]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, [(train_labels == 0).sum(), (train_labels == 1).sum()], width=0.4, label="train")
    ax.bar(x + 0.2, [(test_labels == 0).sum(), (test_labels == 1).sum()], width=0.4, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("reviews")
    ax.set_title("class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(eda_dir / "eda_class_distribution.png", dpi=120)
    plt.close(fig)

    cap = int(np.percentile(train_words, 99))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(train_words, 0, cap), bins=60, alpha=0.7, label="train")
    ax.hist(np.clip(test_words, 0, cap), bins=60, alpha=0.7, label="test")
    ax.set_xlabel(f"words per review (clipped at p99 = {cap})")
    ax.set_ylabel("reviews")
    ax.set_title("review length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(eda_dir / "eda_length_distribution.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for c, name in enumerate(classes):
        ax.hist(np.clip(train_words[train_labels == c], 0, cap), bins=60, alpha=0.6, label=name)
    ax.set_xlabel(f"words per review (clipped at p99 = {cap})")
    ax.set_ylabel("reviews")
    ax.set_title("review length by class, train")
    ax.legend()
    fig.tight_layout()
    fig.savefig(eda_dir / "eda_length_by_class.png", dpi=120)
    plt.close(fig)


def prepare(cfg, data_dir, eda_dir, logger):
    from datasets import load_dataset

    d = cfg["data"]
    rng = np.random.default_rng(cfg["seed"])
    data_dir, eda_dir = Path(data_dir), Path(eda_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    eda_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(d["hf_dataset"])
    train, test = ds["train"], ds["test"]
    logger.info("official splits: train %d test %d columns %s", len(train), len(test), train.column_names)

    train_stats, train_words, train_labels = split_stats(train, d.get("eda_rows"))
    test_stats, test_words, test_labels = split_stats(test, d.get("eda_rows"))
    eda_plots(train_words, train_labels, test_words, test_labels, eda_dir)
    (eda_dir / "eda.json").write_text(json.dumps({"train": train_stats, "test": test_stats}, indent=2), encoding="utf-8")
    for name, stats in (("train", train_stats), ("test", test_stats)):
        logger.info(
            "eda %s rows %d positive fraction %.4f words median %.0f p95 %.0f p99 %.0f malformed %s",
            name, stats["rows"], stats["positive_fraction"], stats["words"]["median"],
            stats["words"]["p95"], stats["words"]["p99"], json.dumps(stats["malformed"]),
        )

    n_train, n_val = int(d["n_train"]), int(d["n_val"])
    perm = rng.permutation(len(train))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val : n_val + n_train])
    if d.get("n_test") is None:
        test_idx = np.arange(len(test))
    else:
        test_idx = np.sort(rng.permutation(len(test))[: int(d["n_test"])])

    pre = Preprocessor(bool(d["remove_stopwords"]), bool(d["keep_negations"]), bool(d["stem"]))

    def tokenize(split, idx):
        sub = split.select(idx.tolist())
        texts = list(sub["text"])
        labels = np.asarray(sub["label"], dtype=np.int64)
        return [pre(t) for t in texts], labels, texts

    train_tok, train_y, _ = tokenize(train, train_idx)
    val_tok, val_y, _ = tokenize(train, val_idx)
    test_tok, test_y, test_texts = tokenize(test, test_idx)
    logger.info("tokenized train %d val %d test %d, stem cache %d words", len(train_tok), len(val_tok), len(test_tok), len(pre.cache))

    counter = Counter()
    for toks in train_tok:
        counter.update(toks)
    itos = ["<pad>", "<unk>"] + [w for w, c in counter.most_common(int(d["vocab_size"]) - 2) if c >= int(d["min_freq"])]
    stoi = {w: i for i, w in enumerate(itos)}
    max_len = int(d["max_len"])
    dtype = np.int16 if len(itos) < 32768 else np.int32

    def encode(token_lists):
        ids = np.zeros((len(token_lists), max_len), dtype=dtype)
        lengths = np.zeros(len(token_lists), dtype=np.int32)
        unk = total = truncated = 0
        for i, toks in enumerate(token_lists):
            if len(toks) > max_len:
                truncated += 1
            enc = [stoi.get(t, UNK_IDX) for t in toks[:max_len]] or [UNK_IDX]
            ids[i, : len(enc)] = enc
            lengths[i] = len(enc)
            unk += sum(1 for e in enc if e == UNK_IDX)
            total += len(enc)
        stats = {
            "unk_rate": unk / max(1, total),
            "truncated_fraction": truncated / max(1, len(token_lists)),
            "mean_tokens": total / max(1, len(token_lists)),
        }
        return ids, lengths, stats

    train_ids, train_len, train_enc = encode(train_tok)
    val_ids, val_len, val_enc = encode(val_tok)
    test_ids, test_len, test_enc = encode(test_tok)
    np.savez_compressed(
        data_dir / "tokens.npz",
        train_ids=train_ids, train_len=train_len, train_y=train_y,
        val_ids=val_ids, val_len=val_len, val_y=val_y,
        test_ids=test_ids, test_len=test_len, test_y=test_y,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )
    (data_dir / "vocab.json").write_text(json.dumps({"itos": itos, "stoi": stoi}, ensure_ascii=False), encoding="utf-8")
    with gzip.open(data_dir / "test_texts.json.gz", "wt", encoding="utf-8") as f:
        json.dump(test_texts, f, ensure_ascii=False)
    with open(data_dir / "test_slices.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "label", "n_words"] + SLICE_COLUMNS)
        for i, (t, y) in enumerate(zip(test_texts, test_y)):
            s = slice_features(t)
            w.writerow([i, int(y), s["n_words"]] + [s[c] for c in SLICE_COLUMNS])

    def class_counts(y):
        return {"negative": int((y == 0).sum()), "positive": int((y == 1).sum())}

    meta = {
        "source": d["hf_dataset"],
        "seed": cfg["seed"],
        "official_train_rows": len(train),
        "official_test_rows": len(test),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "train_class_counts": class_counts(train_y),
        "val_class_counts": class_counts(val_y),
        "test_class_counts": class_counts(test_y),
        "preprocessing": {
            "lowercase": True,
            "literal_newlines_replaced": True,
            "punctuation_and_special_characters_removed": True,
            "apostrophes_kept_inside_words": True,
            "remove_stopwords": bool(d["remove_stopwords"]),
            "keep_negations": bool(d["keep_negations"]),
            "stemmer": "snowball_english" if d["stem"] else None,
        },
        "max_len": max_len,
        "vocab_size": len(itos),
        "min_freq": int(d["min_freq"]),
        "train_encoding": train_enc,
        "val_encoding": val_enc,
        "test_encoding": test_enc,
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("prepared data %s", json.dumps(meta))
    return meta


def load_processed(data_dir):
    data_dir = Path(data_dir)
    z = np.load(data_dir / "tokens.npz")
    data = {k: z[k] for k in z.files}
    vocab = json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))
    data["itos"] = vocab["itos"]
    data["meta"] = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    columns = ["n_words"] + SLICE_COLUMNS
    slices = {c: [] for c in columns}
    with open(data_dir / "test_slices.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for c in columns:
                slices[c].append(row[c])
    data["slices"] = {c: np.asarray(v) for c, v in slices.items()}
    return data


def load_test_texts(data_dir):
    with gzip.open(Path(data_dir) / "test_texts.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)
