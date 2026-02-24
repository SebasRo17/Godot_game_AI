"""Train and save a lightweight AI model for a speedrun game.

This script relies only on the Python standard library so it can run in
restricted environments. It trains a tiny neural network on either a
user-provided CSV dataset or synthetic gameplay samples and saves the weights to
an ``.json`` file for later reuse.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


Vector = List[float]
Matrix = List[Vector]


def _zeros(length: int) -> Vector:
    return [0.0 for _ in range(length)]


def _relu(values: Sequence[float]) -> Vector:
    return [max(0.0, v) for v in values]


def _relu_grad(values: Sequence[float]) -> Vector:
    return [1.0 if v > 0 else 0.0 for v in values]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _add_in_place(target: Vector, updates: Iterable[float]) -> None:
    for idx, val in enumerate(updates):
        target[idx] += val


@dataclass
class Batch:
    features: List[Vector]
    targets: Vector


class SpeedrunDataset:
    """Dataset wrapper that can load data from CSV or synthesize examples."""

    def __init__(self, input_dim: int, synthetic_samples: int, seed: int) -> None:
        self.input_dim = input_dim
        self.synthetic_samples = synthetic_samples
        self.seed = seed

    def load(self, path: Path | None) -> Tuple[List[Vector], Vector]:
        if path is None:
            return self._generate_synthetic()
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return self._load_csv(path)

    def _load_csv(self, path: Path) -> Tuple[List[Vector], Vector]:
        features: List[Vector] = []
        labels: Vector = []

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("CSV file is missing headers.")
            expected_fields = [f"state_{i}" for i in range(self.input_dim)] + [
                "best_time"
            ]
            missing = [field for field in expected_fields if field not in reader.fieldnames]
            if missing:
                raise ValueError(
                    "CSV headers missing required fields: " + ", ".join(missing)
                )

            for row in reader:
                features.append([float(row[f"state_{i}"]) for i in range(self.input_dim)])
                labels.append(float(row["best_time"]))

        if not features:
            raise ValueError("Dataset is empty.")

        return features, labels

    def _generate_synthetic(self) -> Tuple[List[Vector], Vector]:
        rng = random.Random(self.seed)
        features: List[Vector] = []
        targets: Vector = []

        for _ in range(self.synthetic_samples):
            state = [rng.gauss(0.0, 1.0) for _ in range(self.input_dim)]
            base_time = 200.0 - (sum(state) / self.input_dim) * 30.0
            skill_bonus = math.tanh(sum(state)) * 10.0
            noise = rng.gauss(0.0, 2.0)
            targets.append(base_time - skill_bonus + noise)
            features.append(state)

        return features, targets


class SpeedrunModel:
    """A minimal two-layer neural network trained with gradient descent."""

    def __init__(self, input_dim: int, hidden_dim: int, seed: int) -> None:
        rng = random.Random(seed)
        self.W1: Matrix = [
            [rng.uniform(-0.1, 0.1) for _ in range(hidden_dim)]
            for _ in range(input_dim)
        ]
        self.b1: Vector = _zeros(hidden_dim)
        self.W2: Vector = [rng.uniform(-0.1, 0.1) for _ in range(hidden_dim)]
        self.b2: float = 0.0

    def forward(self, x: Vector) -> Tuple[float, Tuple[Vector, Vector]]:
        hidden_dim = len(self.b1)
        z1 = []
        for j in range(hidden_dim):
            weighted = sum(self.W1[i][j] * x[i] for i in range(len(x))) + self.b1[j]
            z1.append(weighted)
        a1 = _relu(z1)
        out = _dot(a1, self.W2) + self.b2
        return out, (z1, a1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mean_squared_error(predictions: Sequence[float], targets: Sequence[float]) -> float:
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(targets)


def _batch_iter(features: List[Vector], targets: Vector, batch_size: int, seed: int) -> Iterable[Batch]:
    rng = random.Random(seed)
    indices = list(range(len(features)))
    rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield Batch([features[i] for i in batch_idx], [targets[i] for i in batch_idx])


def train(
    model: SpeedrunModel,
    features: List[Vector],
    targets: Vector,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[float]:
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        for batch in _batch_iter(features, targets, batch_size, seed + epoch):
            hidden_dim = len(model.W2)
            grad_W2 = _zeros(hidden_dim)
            grad_b2 = 0.0
            grad_W1 = [
                _zeros(hidden_dim) for _ in range(len(model.W1))
            ]
            grad_b1 = _zeros(hidden_dim)

            for x, target in zip(batch.features, batch.targets):
                pred, (z1, a1) = model.forward(x)
                error = pred - target
                grad_pred = 2 * error / len(batch.targets)

                for j, a1_val in enumerate(a1):
                    grad_W2[j] += grad_pred * a1_val
                grad_b2 += grad_pred

                relu_grad = _relu_grad(z1)
                for i, x_i in enumerate(x):
                    for j in range(hidden_dim):
                        grad = grad_pred * model.W2[j] * relu_grad[j]
                        grad_W1[i][j] += grad * x_i
                        grad_b1[j] += grad

            for j in range(hidden_dim):
                model.W2[j] -= learning_rate * grad_W2[j]
            model.b2 -= learning_rate * grad_b2

            for i in range(len(model.W1)):
                for j in range(hidden_dim):
                    model.W1[i][j] -= learning_rate * grad_W1[i][j]
            _add_in_place(model.b1, (-learning_rate * g for g in grad_b1))

        preds = [model.forward(x)[0] for x in features]
        loss = mean_squared_error(preds, targets)
        history.append(loss)
        print(f"Epoch {epoch:03d}: loss={loss:.4f}")

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="CSV file containing state_0..state_N-1 columns and best_time target."
        " When omitted, synthetic data is generated.",
    )
    parser.add_argument("--input-dim", type=int, default=8, help="Number of state features.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden layer width.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--model-in",
        type=Path,
        help="Existing JSON model to resume training from.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/speedrun_model.json"),
        help="Where to save the trained model weights.",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=512,
        help="How many synthetic gameplay samples to generate when no dataset is provided.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    dataset = SpeedrunDataset(args.input_dim, synthetic_samples=args.synthetic_samples, seed=args.seed)
    features, targets = dataset.load(args.dataset)

    model = SpeedrunModel(args.input_dim, args.hidden_dim, seed=args.seed)

    if args.model_in:
        payload = json.loads(args.model_in.read_text(encoding="utf-8"))

        # Minimal validation keeps older checkpoints compatible while ensuring shapes match CLI flags.
        if "W1" not in payload or "b1" not in payload or "W2" not in payload or "b2" not in payload:
            raise ValueError("Checkpoint missing required weight tensors.")
        if len(payload["W1"]) != args.input_dim:
            raise ValueError("Checkpoint input_dim does not match provided --input-dim.")
        hidden_dim = len(payload["b1"])
        if hidden_dim != args.hidden_dim:
            raise ValueError("Checkpoint hidden_dim does not match provided --hidden-dim.")
        if any(len(row) != hidden_dim for row in payload["W1"]):
            raise ValueError("Checkpoint W1 rows do not match hidden_dim.")
        if len(payload["W2"]) != hidden_dim:
            raise ValueError("Checkpoint W2 does not match hidden_dim.")

        # Load saved weights so training continues instead of starting from scratch.
        model.W1 = payload["W1"]
        model.b1 = payload["b1"]
        model.W2 = payload["W2"]
        model.b2 = payload["b2"]

    train(
        model,
        features,
        targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    model.save(args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()