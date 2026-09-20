import json
from pathlib import Path

import numpy as np

UNK = "<unk>"
END = "<|endoftext|>"
STORY_SEP = "\n\n"


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.unk_idx = self.char_to_idx[UNK]
        lut = np.full(0x110000, self.unk_idx, dtype=np.uint32)
        for c, i in self.char_to_idx.items():
            if len(c) == 1:
                lut[ord(c)] = i
        self._lut = lut

    @classmethod
    def from_text(cls, text):
        return cls([UNK] + sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return [self.char_to_idx.get(c, self.unk_idx) for c in text]

    def encode_np(self, text):
        codes = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        return self._lut[codes]

    def decode(self, ids):
        return "".join(self.idx_to_char.get(int(i), UNK) for i in ids)

    def save(self, path):
        payload = {
            "chars": self.chars,
            "char_to_idx": self.char_to_idx,
            "idx_to_char": {str(k): v for k, v in self.idx_to_char.items()},
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["chars"])


def iter_stories(source, hf_dataset):
    if source != "hf":
        text = Path(source).read_text(encoding="utf-8")
        for story in text.split(END):
            story = story.strip()
            if story:
                yield story
        return
    from datasets import load_dataset

    ds = load_dataset(hf_dataset, split="train", streaming=True)
    for row in ds:
        story = row["text"].strip()
        if story:
            yield story


def to_sequences(ids, seq_len):
    n = len(ids) // seq_len
    return ids[: n * seq_len].reshape(n, seq_len)


def prepare(cfg, out_dir, logger):
    d = cfg["data"]
    seq_len = d["block_size"] + 1
    n_train, n_val = d["n_train"], d["n_val"]
    need_chars = int((n_train + n_val) * seq_len * d.get("char_margin", 1.05))
    rng = np.random.default_rng(cfg["seed"])

    stories, total = [], 0
    for story in iter_stories(d["source"], d["hf_dataset"]):
        stories.append(story)
        total += len(story) + len(STORY_SEP)
        if total >= need_chars:
            break
    logger.info("collected %d stories, %d chars from %s", len(stories), total, d["source"])

    order = rng.permutation(len(stories))
    n_val_stories = max(1, int(len(stories) * d["val_story_fraction"]))
    val_stories = [stories[i] for i in order[:n_val_stories]]
    train_stories = [stories[i] for i in order[n_val_stories:]]
    train_text = STORY_SEP.join(train_stories)
    val_text = STORY_SEP.join(val_stories)

    tok = CharTokenizer.from_text(train_text)
    train_seqs = to_sequences(tok.encode_np(train_text), seq_len)
    val_seqs = to_sequences(tok.encode_np(val_text), seq_len)
    if len(train_seqs) < n_train or len(val_seqs) < n_val:
        raise RuntimeError(
            f"not enough sequences: train {len(train_seqs)}/{n_train}, val {len(val_seqs)}/{n_val}; raise char_margin"
        )
    train_seqs = train_seqs[:n_train]
    val_seqs = val_seqs[:n_val]

    dtype = np.uint8 if tok.vocab_size <= 256 else np.uint16
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train.npy", train_seqs.astype(dtype))
    np.save(out_dir / "val.npy", val_seqs.astype(dtype))
    tok.save(out_dir / "vocab.json")
    meta = {
        "source": d["hf_dataset"] if d["source"] == "hf" else d["source"],
        "seed": cfg["seed"],
        "block_size": d["block_size"],
        "n_train_sequences": int(n_train),
        "n_val_sequences": int(n_val),
        "train_tokens": int(n_train * d["block_size"]),
        "val_tokens": int(n_val * d["block_size"]),
        "vocab_size": tok.vocab_size,
        "n_stories_total": len(stories),
        "n_train_stories": len(train_stories),
        "n_val_stories": len(val_stories),
        "val_unk_tokens": int((val_seqs == tok.unk_idx).sum()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("prepared data %s", json.dumps(meta))
    return meta


def load_arrays(data_dir):
    data_dir = Path(data_dir)
    tok = CharTokenizer.load(data_dir / "vocab.json")
    train = np.load(data_dir / "train.npy")
    val = np.load(data_dir / "val.npy")
    return tok, train, val
