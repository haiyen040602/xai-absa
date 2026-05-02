import argparse
import csv
import glob
import json
import os
import pickle
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import jsonlines
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import matplotlib.pyplot as plt


LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


@dataclass
class Example:
    sentence: str
    aspect: str
    label: int


@dataclass
class SplitExamples:
    train: List[Example]
    val: List[Example]
    test: List[Example]


class TransformerDataset(Dataset):
    def __init__(self, examples: Sequence[Example], tokenizer, max_length: int, separator: str):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.separator = separator

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        text = f"{ex.sentence} {self.separator} {ex.aspect}"
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(ex.label, dtype=torch.long)
        return item


class BaselineDataset(Dataset):
    def __init__(self, examples: Sequence[Example], vocab: Dict[str, int], max_length: int):
        self.examples = list(examples)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        text = f"{ex.sentence} [SEP] {ex.aspect}"
        token_ids = encode_text(text, self.vocab, self.max_length)
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(ex.label, dtype=torch.long)


class RNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
        model_type: str,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.model_type = model_type

        if model_type in {"lstm", "bilstm"}:
            self.encoder = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=(model_type == "bilstm"),
                dropout=dropout if n_layers > 1 else 0.0,
            )
            out_dim = hidden_dim * (2 if model_type == "bilstm" else 1)
        elif model_type == "rnn":
            self.encoder = nn.RNN(
                embedding_dim,
                hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            out_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported baseline model_type: {model_type}")

        self.fc = nn.Linear(out_dim, output_dim)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        if self.model_type in {"lstm", "bilstm"}:
            _, (hidden, _) = self.encoder(packed)
            if self.model_type == "bilstm":
                hidden_out = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                hidden_out = hidden[-1]
        else:
            _, hidden = self.encoder(packed)
            hidden_out = hidden[-1]

        hidden_out = self.dropout(hidden_out)
        return self.fc(hidden_out)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_separator(model_name: str) -> str:
    name = model_name.lower()
    if "roberta" in name:
        return "</s></s>"
    if "xlnet" in name:
        return "<sep>"
    return "[SEP]"


def resolve_dataset_path(user_path: str) -> str:
    if user_path and os.path.exists(user_path):
        return user_path

    local_candidate = os.path.join(
        "Datasets", "FINAL_CLEANED_CORRECTED_SHUFFLED_DATASET_NO_DUPLICATE.jsonl"
    )
    if os.path.exists(local_candidate):
        return local_candidate

    kaggle_candidates = glob.glob(
        "/kaggle/input/*/FINAL_CLEANED_CORRECTED_SHUFFLED_DATASET_NO_DUPLICATE.jsonl"
    )
    if kaggle_candidates:
        return kaggle_candidates[0]

    raise FileNotFoundError(
        "Could not find dataset. Pass --dataset_path explicitly or place the JSONL file in Datasets/."
    )


def load_examples(dataset_path: str) -> List[Example]:
    examples: List[Example] = []
    with jsonlines.open(dataset_path) as reader:
        for row in reader:
            sentence = row["text"]
            for item in row["labels"]:
                aspect = item["aspect"]
                polarity = item["polarity"].lower().strip()
                if polarity not in LABEL_MAP:
                    continue
                examples.append(Example(sentence=sentence, aspect=aspect, label=LABEL_MAP[polarity]))

    if not examples:
        raise ValueError("No valid examples loaded from dataset.")
    return examples


def split_examples(examples: Sequence[Example], seed: int) -> SplitExamples:
    labels = [ex.label for ex in examples]
    train_set, rem_set = train_test_split(
        list(examples),
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )

    rem_labels = [ex.label for ex in rem_set]
    val_set, test_set = train_test_split(
        rem_set,
        test_size=0.50,
        random_state=seed,
        stratify=rem_labels,
    )
    return SplitExamples(train=train_set, val=val_set, test=test_set)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def save_predictions(path: str, labels: List[int], preds: List[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["true_label", "pred_label"])
        writer.writeheader()
        for y_true, y_pred in zip(labels, preds):
            writer.writerow(
                {
                    "true_label": INV_LABEL_MAP.get(y_true, str(y_true)),
                    "pred_label": INV_LABEL_MAP.get(y_pred, str(y_pred)),
                }
            )


def save_confusion_matrix_artifacts(exp_dir: str, labels: List[int], preds: List[int]) -> Dict[str, str]:
    labels_order = [0, 1, 2]
    class_names = [INV_LABEL_MAP[idx] for idx in labels_order]
    matrix = confusion_matrix(labels, preds, labels=labels_order)

    csv_path = os.path.join(exp_dir, "confusion_matrix.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["true\\pred"] + class_names)
        for i, row in enumerate(matrix.tolist()):
            writer.writerow([class_names[i]] + row)

    json_path = os.path.join(exp_dir, "confusion_matrix.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "labels_order": labels_order,
                "class_names": class_names,
                "matrix": matrix.tolist(),
            },
            fp,
            indent=2,
        )

    png_path = os.path.join(exp_dir, "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)

    threshold = matrix.max() / 2.0 if matrix.size > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    return {
        "confusion_matrix_csv": csv_path,
        "confusion_matrix_json": json_path,
        "confusion_matrix_png": png_path,
    }


def simple_tokenizer(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_vocab(examples: Sequence[Example]) -> Dict[str, int]:
    all_tokens: List[str] = []
    for ex in examples:
        text = f"{ex.sentence} [SEP] {ex.aspect}"
        all_tokens.extend(simple_tokenizer(text))

    counts = Counter(all_tokens)
    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in counts.items():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_length: int) -> List[int]:
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in simple_tokenizer(text)]
    return ids[:max_length]


def baseline_collate(batch):
    input_ids, labels = zip(*batch)
    lengths = torch.tensor([len(x) if len(x) > 0 else 1 for x in input_ids], dtype=torch.long)

    safe_input_ids = []
    for x in input_ids:
        if len(x) == 0:
            safe_input_ids.append(torch.tensor([1], dtype=torch.long))
        else:
            safe_input_ids.append(x)

    padded = pad_sequence(safe_input_ids, batch_first=True, padding_value=0)
    labels_tensor = torch.stack(labels)
    return padded, labels_tensor, lengths


def train_epoch_transformer(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == batch["labels"]).sum().item())
        total_seen += int(batch["labels"].size(0))

    avg_loss = total_loss / max(len(loader), 1)
    avg_acc = total_correct / max(total_seen, 1)
    return avg_loss, avg_acc


def evaluate_transformer(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            preds = torch.argmax(outputs.logits, dim=1)
            all_labels.extend(batch["labels"].cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(len(loader), 1)
    metrics["labels"] = all_labels
    metrics["preds"] = all_preds
    return metrics


def train_epoch_baseline(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for input_ids, labels, lengths in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == labels).sum().item())
        total_seen += int(labels.size(0))

    avg_loss = total_loss / max(len(loader), 1)
    avg_acc = total_correct / max(total_seen, 1)
    return avg_loss, avg_acc


def evaluate_baseline(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for input_ids, labels, lengths in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())

            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(len(loader), 1)
    metrics["labels"] = all_labels
    metrics["preds"] = all_preds
    return metrics


def run_transformer(args, split: SplitExamples, device, root_output_dir: str) -> Dict:
    exp_name = "transformer"
    exp_dir = os.path.join(root_output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    separator = get_separator(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
    model.to(device)

    train_ds = TransformerDataset(split.train, tokenizer, args.max_length, separator)
    val_ds = TransformerDataset(split.val, tokenizer, args.max_length, separator)
    test_ds = TransformerDataset(split.test, tokenizer, args.max_length, separator)

    train_loader = make_loader(train_ds, args.batch_size, True)
    val_loader = make_loader(val_ds, args.batch_size, False)
    test_loader = make_loader(test_ds, args.batch_size, False)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    best_val_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch_transformer(model, train_loader, optimizer, device)
        val_metrics = evaluate_transformer(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)

        print(
            f"[{exp_name}] Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            model.save_pretrained(exp_dir)
            tokenizer.save_pretrained(exp_dir)

    test_metrics = evaluate_transformer(model, test_loader, device)
    save_predictions(
        os.path.join(exp_dir, "test_predictions.csv"), test_metrics["labels"], test_metrics["preds"]
    )
    cm_paths = save_confusion_matrix_artifacts(exp_dir, test_metrics["labels"], test_metrics["preds"])

    metrics_payload = {
        "experiment": exp_name,
        "model_name": args.model_name,
        "separator": separator,
        "history": history,
        "artifacts": cm_paths,
        "test": {k: v for k, v in test_metrics.items() if k not in {"labels", "preds"}},
    }
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)

    print(f"[{exp_name}] Test metrics: {json.dumps(metrics_payload['test'], indent=2)}")
    return metrics_payload


def run_baseline(args, split: SplitExamples, device, root_output_dir: str, baseline_name: str) -> Dict:
    exp_name = baseline_name
    exp_dir = os.path.join(root_output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    vocab_source = split.train + split.val + split.test
    vocab = build_vocab(vocab_source)

    train_ds = BaselineDataset(split.train, vocab, args.baseline_max_length)
    val_ds = BaselineDataset(split.val, vocab, args.baseline_max_length)
    test_ds = BaselineDataset(split.test, vocab, args.baseline_max_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.baseline_batch_size, shuffle=True, collate_fn=baseline_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.baseline_batch_size, shuffle=False, collate_fn=baseline_collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.baseline_batch_size, shuffle=False, collate_fn=baseline_collate
    )

    model = RNNLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=3,
        n_layers=args.n_layers,
        dropout=args.dropout,
        model_type=baseline_name,
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.baseline_learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_f1 = -1.0
    history = []

    for epoch in range(1, args.baseline_epochs + 1):
        train_loss, train_acc = train_epoch_baseline(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_baseline(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)

        print(
            f"[{exp_name}] Epoch {epoch}/{args.baseline_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), os.path.join(exp_dir, "model.pth"))
            with open(os.path.join(exp_dir, "vocab.pkl"), "wb") as fp:
                pickle.dump(vocab, fp)

    test_metrics = evaluate_baseline(model, test_loader, criterion, device)
    save_predictions(
        os.path.join(exp_dir, "test_predictions.csv"), test_metrics["labels"], test_metrics["preds"]
    )
    cm_paths = save_confusion_matrix_artifacts(exp_dir, test_metrics["labels"], test_metrics["preds"])

    metrics_payload = {
        "experiment": exp_name,
        "history": history,
        "artifacts": cm_paths,
        "config": {
            "baseline_epochs": args.baseline_epochs,
            "baseline_batch_size": args.baseline_batch_size,
            "baseline_learning_rate": args.baseline_learning_rate,
            "baseline_max_length": args.baseline_max_length,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
        },
        "test": {k: v for k, v in test_metrics.items() if k not in {"labels", "preds"}},
    }
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)

    print(f"[{exp_name}] Test metrics: {json.dumps(metrics_payload['test'], indent=2)}")
    return metrics_payload


def save_comparison_table(root_output_dir: str, rows: List[Dict]) -> None:
    table_path = os.path.join(root_output_dir, "comparison_table.csv")
    fields = ["experiment", "accuracy", "macro_precision", "macro_recall", "macro_f1", "loss"]

    with open(table_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    with open(os.path.join(root_output_dir, "comparison_table.json"), "w", encoding="utf-8") as fp:
        json.dump(rows, fp, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Kaggle reproducibility runner for ABSA.")
    parser.add_argument("--dataset_path", type=str, default="", help="Path to JSONL dataset.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="transformer",
        choices=["transformer", "lstm", "bilstm", "rnn", "all"],
        help="Experiment to run.",
    )

    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--baseline_epochs", type=int, default=10)
    parser.add_argument("--baseline_batch_size", type=int, default=8)
    parser.add_argument("--baseline_learning_rate", type=float, default=1e-3)
    parser.add_argument("--baseline_max_length", type=int, default=50)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/absa_outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_path = resolve_dataset_path(args.dataset_path)
    print(f"Using dataset: {dataset_path}")
    print(f"Selected experiment: {args.experiment}")

    examples = load_examples(dataset_path)
    split = split_examples(examples, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    plan = [args.experiment]
    if args.experiment == "all":
        plan = ["transformer", "lstm", "bilstm", "rnn"]

    comparison_rows: List[Dict] = []
    run_summaries: Dict[str, Dict] = {}

    for item in plan:
        if item == "transformer":
            payload = run_transformer(args, split, device, args.output_dir)
        else:
            payload = run_baseline(args, split, device, args.output_dir, item)

        test_row = dict(payload["test"])
        test_row["experiment"] = item
        comparison_rows.append(test_row)
        run_summaries[item] = payload

    save_comparison_table(args.output_dir, comparison_rows)

    master = {
        "config": vars(args),
        "dataset_path": dataset_path,
        "runs": run_summaries,
        "comparison_table": comparison_rows,
    }
    with open(os.path.join(args.output_dir, "run_summary.json"), "w", encoding="utf-8") as fp:
        json.dump(master, fp, indent=2)

    print("All done. Summary table:")
    print(json.dumps(comparison_rows, indent=2))
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
