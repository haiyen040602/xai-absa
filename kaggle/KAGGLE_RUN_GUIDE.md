# Kaggle Run Guide

This guide helps you reproduce the transformer experiment from this repository on Kaggle.

## 1) Upload dataset to Kaggle

Create a Kaggle Dataset that includes:

- `FINAL_CLEANED_CORRECTED_SHUFFLED_DATASET_NO_DUPLICATE.jsonl`

You can keep the exact filename. The runner auto-detects it under `/kaggle/input/*/`.

## 2) Start a Kaggle Notebook and pull the GitHub code

In a notebook cell:

```bash
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>
```

## 3) Install dependencies

```bash
!pip install -q -r kaggle/requirements-kaggle.txt
```

## 4) Run the experiment

Default run (DistilBERT, 3 epochs):

```bash
!python kaggle/run_absa_repro.py
```

Run full comparison table (Transformer + LSTM + BiLSTM + RNN):

```bash
!python kaggle/run_absa_repro.py \
  --experiment all \
  --model_name roberta-base \
  --epochs 3 \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --max_length 256 \
  --baseline_epochs 10 \
  --baseline_batch_size 8 \
  --baseline_learning_rate 1e-3 \
  --baseline_max_length 50 \
  --output_dir /kaggle/working/absa_outputs
```

Run with custom hyperparameters (example RoBERTa):

```bash
!python kaggle/run_absa_repro.py \
  --model_name roberta-base \
  --epochs 15 \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --max_length 256 \
  --output_dir /kaggle/working/roberta_run
```

## 5) Outputs

The script writes artifacts to the output directory:

- Saved model + tokenizer
- `metrics.json`
- `test_predictions.csv`
- `comparison_table.csv`
- `comparison_table.json`
- `run_summary.json`

Default output:

- `/kaggle/working/absa_outputs`

## Notes

- Separator token is auto-selected by model type:
  - RoBERTa: `</s></s>`
  - XLNet: `<sep>`
  - Others (BERT, DistilBERT, ALBERT): `[SEP]`
- If auto-detection fails, pass dataset path manually:

```bash
!python kaggle/run_absa_repro.py --dataset_path /kaggle/input/<dataset-name>/FINAL_CLEANED_CORRECTED_SHUFFLED_DATASET_NO_DUPLICATE.jsonl
```

## 6) Notebook template

Use the notebook template in this repo:

- `kaggle/KAGGLE_RUN_ALL.ipynb`

It is pre-wired for Run All on Kaggle and reads the comparison table at the end.
