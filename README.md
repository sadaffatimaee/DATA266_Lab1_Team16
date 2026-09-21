# DATA266 Lab 1, Team 16

LLM pretraining from scratch, Yelp polarity sentiment classification, and CycleGAN style transfer.

Members: Poushali Purkayastha, Sadaf Fatima Syeda. Each member builds, trains, and documents their own model for every task under their own named folder.

## Layout

The repo follows the structure given in the lab brief.

```
README.md
task1_llm/
  data/                      shared raw TinyStories pointer
  Poushali_Purkayastha/
  Sadaf_Fatima_Syeda/
task2_sentiment/
  data/                      shared raw Yelp polarity pointer
  Poushali_Purkayastha/
  Sadaf_Fatima_Syeda/
task3_gan/
  data/                      shared Monet and photo images pointer
  Poushali_Purkayastha/
  Sadaf_Fatima_Syeda/
reproducibility/
  manifests/                 one JSON per run: config, hardware, package versions, checkpoint to metric mapping
  raw_logs/                  one untouched log per run
report/
  DATA266_Lab1_Report_Team_16.pdf
```

Inside each member folder: `src/` (code and the notebook with outputs), `data_processed/`, `checkpoints/`, `outputs/`, `metrics_report.csv`, `failure_analysis.md`, `results.md`. Task 3 folders also hold `outputs/pred_A2B/`, `outputs/pred_B2A/`, `evaluate_local.py`, `submission.csv`, and `full_metrics_report.csv`.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows activate with `.venv\Scripts\activate`. Python 3.12 was used. On Linux with a GPU the pinned torch installs with CUDA support.

## Smoke test (one command)

Runs Poushali's Task 1 pipeline end to end on a tiny config, on CPU, in about a minute. It streams a few hundred TinyStories from Hugging Face, trains a 2-layer model for one epoch, generates text, and writes metrics, plots, a raw log, and a manifest.

```
python task1_llm/Poushali_Purkayastha/src/run.py --config configs/smoke.yaml
```

Outputs land in `task1_llm/Poushali_Purkayastha/outputs/smoke/`, the log in `reproducibility/raw_logs/`, and the manifest in `reproducibility/manifests/`.

The Task 2 equivalent trains the three sentiment models on 3,000 reviews and evaluates on 2,000, in under a minute after a one-time dataset download:

```
python task2_sentiment/Poushali_Purkayastha/src/run.py --config configs/smoke.yaml
```

## Task 1, Poushali Purkayastha

Full run, 10 epochs on 100K training sequences, meant for a GPU:

```
python task1_llm/Poushali_Purkayastha/src/run.py --config configs/full.yaml
```

The same run through the notebook, so its outputs are saved in place:

```
bash task1_llm/Poushali_Purkayastha/src/run_task1.sh
```

Throughput benchmark, 30 steps, prints the estimated full training time on the current hardware:

```
python task1_llm/Poushali_Purkayastha/src/run.py --config configs/full.yaml --bench-steps 30
```

Where results live:

| File | Content |
| --- | --- |
| task1_llm/Poushali_Purkayastha/metrics_report.csv | every required Task 1 metric from the final run |
| task1_llm/Poushali_Purkayastha/outputs/full/ | loss_curves.png, training_dynamics.png, samples.txt, run_summary.json, history.json |
| task1_llm/Poushali_Purkayastha/checkpoints/full/final.pt and best.pt | trained weights with vocabulary and config |
| task1_llm/Poushali_Purkayastha/data_processed/full/ | the 100K/10K split, vocab.json with char_to_idx and idx_to_char, meta.json |
| task1_llm/Poushali_Purkayastha/results.md | architecture, hyperparameters, and results |
| task1_llm/Poushali_Purkayastha/failure_analysis.md | three failure cases |

Per-epoch checkpoints and last.pt are not committed. They are backed up to Drive after each run.

## Task 1 evaluation script

`task1_llm/Poushali_Purkayastha/src/evaluate_task1.py` computes every Task 1 metric from two files. Each member's src folder carries the same script so both members' numbers are computed the same way.

```
python task1_llm/Poushali_Purkayastha/src/evaluate_task1.py --summary <run_summary.json> --samples <samples.json> --out metrics_report.csv
```

`run_summary.json` needs these keys: final_train_loss, final_val_loss, best_val_loss, val_top1_accuracy, grad_norm_mean, grad_norm_max, grad_norm_last, loss_spikes, nan_losses, param_count, train_tokens_per_sec, generation_tokens_per_sec, peak_memory_mb, peak_memory_source, train_time_sec, epochs, optimizer_steps, device. `samples.json` is a list of objects with a `continuation` field holding the generated text without the prompt. Distinct-n and the repeated 4-gram rate are computed on whitespace-split words of the continuations; the repeated 4-gram rate is averaged per sample.

## Conventions

- Every run is driven by a YAML config. No hard-coded paths, credentials, or API keys.
- Every run writes one raw log to `reproducibility/raw_logs/` and one manifest to `reproducibility/manifests/`. Neither is edited after the run.
- Raw datasets are not committed. Each task's `data/README.md` points to the shared copy.

## Task 2, Poushali Purkayastha

Full run, three models on 100K training reviews, evaluated on the official 38K test split, meant for a GPU:

```
python task2_sentiment/Poushali_Purkayastha/src/run.py --config configs/full.yaml
```

Train a subset of the models, for example only the two experimental ones:

```
python task2_sentiment/Poushali_Purkayastha/src/run.py --config configs/full.yaml --models textcnn,bilstm
```

The same run through the notebook, so its outputs are saved in place:

```
bash task2_sentiment/Poushali_Purkayastha/src/run_task2.sh
```

Throughput benchmark, 30 steps per model:

```
python task2_sentiment/Poushali_Purkayastha/src/run.py --config configs/full.yaml --bench-steps 30
```

Re-run only the evaluation on saved predictions, for example after adding a model:

```
python task2_sentiment/Poushali_Purkayastha/src/evaluate_task2.py --config configs/full.yaml
```

Where results live:

| File | Content |
| --- | --- |
| task2_sentiment/Poushali_Purkayastha/metrics_report.csv | every required Task 2 metric for every model, long format |
| task2_sentiment/Poushali_Purkayastha/outputs/full/metrics_wide.csv | the same metrics with models side by side |
| task2_sentiment/Poushali_Purkayastha/outputs/full/eda/ | class distribution, length distributions, eda.json with malformed-row checks |
| task2_sentiment/Poushali_Purkayastha/outputs/full/<model>/ | curves, confusion matrix, ROC and PR curves, reliability diagram, test probabilities, error_review_candidates.md |
| task2_sentiment/Poushali_Purkayastha/checkpoints/full/<model>/best.pt | best-epoch weights per model |
| task2_sentiment/Poushali_Purkayastha/data_processed/full/ | tokenized splits, vocabulary, test slices, test texts, meta.json |
| task2_sentiment/Poushali_Purkayastha/results.md | preprocessing, model and hyperparameter justification, results |
| task2_sentiment/Poushali_Purkayastha/failure_analysis.md | the 20-error manual review and the proposed fix |

## Task 3

Added as the work lands.
