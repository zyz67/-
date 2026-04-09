import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


V0 = 3.5
XI = np.array([1.0, -1.0, 0.2], dtype=np.float32)
YI = np.array([0.5, -0.8, 1.2], dtype=np.float32)
AI = np.array([2.0, 1.5, 1.0], dtype=np.float32)
SIGMA_X = np.array([0.5, 0.7, 1.0], dtype=np.float32)
SIGMA_Y = np.array([0.8, 0.4, 0.5], dtype=np.float32)
CONTOUR_INFER_BATCH_SIZE = 4096
COLORBAR_SHRINK = 0.85
MIN_SAMPLING_BATCH_SIZE = 1024
EXTRAPOLATION_OVERSAMPLE_FACTOR = 2
CONTOUR_LEVELS = 30


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def potential_and_force_np(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    dx = x - XI[None, :]
    dy = y - YI[None, :]
    exp_term = np.exp(-(dx ** 2) / (2 * (SIGMA_X[None, :] ** 2)) - (dy ** 2) / (2 * (SIGMA_Y[None, :] ** 2)))
    v = V0 - np.sum(AI[None, :] * exp_term, axis=1)
    dv_dx = np.sum(AI[None, :] * exp_term * (dx / (SIGMA_X[None, :] ** 2)), axis=1)
    dv_dy = np.sum(AI[None, :] * exp_term * (dy / (SIGMA_Y[None, :] ** 2)), axis=1)
    force = np.stack([-dv_dx, -dv_dy], axis=1)
    return v.astype(np.float32), force.astype(np.float32)


def sample_uniform(n: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(low, high, size=(n, 2)).astype(np.float32)


def sample_grid(n: int, low: float, high: float) -> np.ndarray:
    side = int(math.ceil(math.sqrt(n)))
    coords = np.linspace(low, high, side, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    grid = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    return grid[:n]


def sample_lhs(n: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    points = np.zeros((n, 2), dtype=np.float32)
    for dim in range(2):
        perm = rng.permutation(n)
        points[:, dim] = (perm + rng.random(n)) / n
    return (low + points * (high - low)).astype(np.float32)


def sample_gradient_importance(n: int, low: float, high: float, candidates: int, rng: np.random.Generator) -> np.ndarray:
    if n > candidates:
        raise ValueError(f"train_size ({n}) must be <= candidate_size ({candidates}) for grad_importance sampling")
    pool = sample_uniform(candidates, low, high, rng)
    _, f = potential_and_force_np(pool)
    weight = np.linalg.norm(f, axis=1) + 1e-6
    prob = weight / weight.sum()
    idx = rng.choice(candidates, size=n, replace=False, p=prob)
    return pool[idx]


def sample_extrapolation_ring(
    n: int,
    inner_low: float,
    inner_high: float,
    outer_low: float,
    outer_high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    collected_points: List[np.ndarray] = []
    collected = 0
    while collected < n:
        batch = max((n - collected) * EXTRAPOLATION_OVERSAMPLE_FACTOR, MIN_SAMPLING_BATCH_SIZE)
        cand = rng.uniform(outer_low, outer_high, size=(batch, 2)).astype(np.float32)
        in_inner = (
            (cand[:, 0] >= inner_low)
            & (cand[:, 0] <= inner_high)
            & (cand[:, 1] >= inner_low)
            & (cand[:, 1] <= inner_high)
        )
        selected = cand[~in_inner]
        if selected.shape[0] > 0:
            collected_points.append(selected)
            collected += selected.shape[0]
    return np.concatenate(collected_points, axis=0)[:n]


def sample_points(method: str, n: int, low: float, high: float, rng: np.random.Generator, candidates: int) -> np.ndarray:
    if method == "uniform":
        return sample_uniform(n, low, high, rng)
    if method == "grid":
        return sample_grid(n, low, high)
    if method == "lhs":
        return sample_lhs(n, low, high, rng)
    if method == "grad_importance":
        return sample_gradient_importance(n, low, high, candidates, rng)
    raise ValueError(f"Unknown sampling method: {method}")


class MLP(nn.Module):
    def __init__(self, hidden_dims: List[int], activation: str, output_dim: int, dropout_prob: float = 0.0):
        super().__init__()
        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "sigmoid": nn.Sigmoid,
        }
        if activation not in act_map:
            raise ValueError(f"Unsupported activation: {activation}")
        if not (0.0 <= dropout_prob < 1.0):
            raise ValueError("dropout_prob must be >= 0.0 and < 1.0")
        layers: List[nn.Module] = []
        prev = 2
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), act_map[activation]()])
            if dropout_prob > 0:
                layers.append(nn.Dropout(p=dropout_prob))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainResult:
    model: nn.Module
    train_loss: List[float]
    val_loss: List[float]
    test_energy_mae: float
    test_force_mae: float


def make_optimizer(name: str, params: Iterable[torch.nn.Parameter], lr: float):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def predict(model: nn.Module, x: torch.Tensor, mode: str, create_graph: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if mode == "direct":
        out = model(x)
        return out[:, 0], out[:, 1:3]
    if mode == "autograd":
        x_in = x.clone().detach().requires_grad_(True)
        v = model(x_in).squeeze(-1)
        grad = torch.autograd.grad(v.sum(), x_in, create_graph=create_graph)[0]
        return v, -grad
    raise ValueError(f"Unknown force mode: {mode}")


def evaluate_mae(model: nn.Module, loader: DataLoader, mode: str, device: torch.device) -> Tuple[float, float]:
    model.eval()
    e_abs = 0.0
    f_abs = 0.0
    count = 0
    for x, y_e, y_f in loader:
        x = x.to(device)
        y_e = y_e.to(device)
        y_f = y_f.to(device)
        with torch.set_grad_enabled(mode == "autograd"):
            p_e, p_f = predict(model, x, mode, create_graph=False)
        e_abs += torch.sum(torch.abs(p_e - y_e)).item()
        f_abs += torch.sum(torch.abs(p_f - y_f)).item()
        count += x.size(0)
    return e_abs / count, f_abs / (count * 2)


def predict_energy_only(model: nn.Module, x: torch.Tensor, mode: str) -> torch.Tensor:
    out = model(x)
    if mode == "direct":
        return out[:, 0]
    if mode == "autograd":
        return out.squeeze(-1)
    raise ValueError(f"Unknown force mode: {mode}")


def plot_contour_comparison(
    model: nn.Module,
    mode: str,
    device: torch.device,
    out_path: str,
    low: float,
    high: float,
    grid_size: int = 161,
) -> None:
    coords = np.linspace(low, high, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    true_v, _ = potential_and_force_np(points)

    model.eval()
    pred_v_list: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, points.shape[0], CONTOUR_INFER_BATCH_SIZE):
            end = min(i + CONTOUR_INFER_BATCH_SIZE, points.shape[0])
            x = torch.from_numpy(points[i:end]).to(device)
            pred_v_list.append(predict_energy_only(model, x, mode).detach().cpu().numpy())
    pred_v = np.concatenate(pred_v_list, axis=0)

    true_map = true_v.reshape(grid_size, grid_size)
    pred_map = pred_v.reshape(grid_size, grid_size)
    vmin = float(min(true_map.min(), pred_map.min()))
    vmax = float(max(true_map.max(), pred_map.max()))
    levels = np.linspace(vmin, vmax, CONTOUR_LEVELS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    c0 = axes[0].contourf(xx, yy, true_map, levels=levels)
    axes[0].set_title("True Potential Energy Surface (PES)")
    axes[0].set_xlim(low, high)
    axes[0].set_ylim(low, high)
    c1 = axes[1].contourf(xx, yy, pred_map, levels=levels)
    axes[1].set_title("Predicted Potential Energy Surface (PES)")
    axes[1].set_xlim(low, high)
    axes[1].set_ylim(low, high)
    fig.colorbar(c0, ax=axes[0], shrink=COLORBAR_SHRINK)
    fig.colorbar(c1, ax=axes[1], shrink=COLORBAR_SHRINK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_once(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    test_dataset: TensorDataset,
    hidden_dims: List[int],
    activation: str,
    dropout: float,
    force_mode: str,
    optimizer_name: str,
    lr: float,
    alpha: float,
    lambda_force: float,
    epochs: int,
    batch_size: int,
    patience: int,
    device: torch.device,
) -> TrainResult:
    output_dim = 3 if force_mode == "direct" else 1
    model = MLP(hidden_dims=hidden_dims, activation=activation, output_dim=output_dim, dropout_prob=dropout).to(device)
    optimizer = make_optimizer(optimizer_name, model.parameters(), lr)
    mse = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []
    best_val = float("inf")
    best_state = None
    stale = 0

    for _ in range(epochs):
        model.train()
        total_train = 0.0
        total_count = 0
        for x, y_e, y_f in train_loader:
            x = x.to(device)
            y_e = y_e.to(device)
            y_f = y_f.to(device)
            optimizer.zero_grad()
            p_e, p_f = predict(model, x, force_mode, create_graph=(force_mode == "autograd"))
            loss_e = mse(p_e, y_e)
            loss_f = mse(p_f, y_f)
            loss = alpha * loss_e + lambda_force * loss_f
            loss.backward()
            optimizer.step()
            bsz = x.size(0)
            total_train += loss.item() * bsz
            total_count += bsz

        train_loss_hist.append(total_train / total_count)

        model.eval()
        total_val = 0.0
        val_count = 0
        for x, y_e, y_f in val_loader:
            x = x.to(device)
            y_e = y_e.to(device)
            y_f = y_f.to(device)
            with torch.set_grad_enabled(force_mode == "autograd"):
                p_e, p_f = predict(model, x, force_mode, create_graph=False)
                v_loss = alpha * mse(p_e, y_e) + lambda_force * mse(p_f, y_f)
            bsz = x.size(0)
            total_val += v_loss.item() * bsz
            val_count += bsz
        cur_val = total_val / val_count
        val_loss_hist.append(cur_val)

        if cur_val < best_val:
            best_val = cur_val
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    e_mae, f_mae = evaluate_mae(model, test_loader, force_mode, device)
    return TrainResult(model=model, train_loss=train_loss_hist, val_loss=val_loss_hist, test_energy_mae=e_mae, test_force_mae=f_mae)


def to_dataset(points: np.ndarray) -> TensorDataset:
    e, f = potential_and_force_np(points)
    x_t = torch.from_numpy(points)
    e_t = torch.from_numpy(e)
    f_t = torch.from_numpy(f)
    return TensorDataset(x_t, e_t, f_t)


def plot_sampling(points_map: Dict[str, np.ndarray], out_path: str) -> None:
    keys = list(points_map.keys())
    cols = 2
    rows = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8, 4 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for i, key in enumerate(keys):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        pts = points_map[key]
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.7)
        ax.set_title(key)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    for j in range(len(keys), rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss(train_loss: List[float], val_loss: List[float], out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_hidden_dims(text: str) -> List[int]:
    dims = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not dims:
        raise ValueError('hidden_dims must be a comma-separated list of integers, e.g., "64,64"')
    return dims


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal runnable PES experiment framework")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--sampling_method", type=str, default="uniform", choices=["uniform", "grid", "lhs", "grad_importance"])
    parser.add_argument("--compare_sampling", action="store_true")
    parser.add_argument("--train_size", type=int, default=512)
    parser.add_argument("--val_size", type=int, default=256)
    parser.add_argument("--test_size", type=int, default=1024)
    parser.add_argument("--candidate_size", type=int, default=5000)
    parser.add_argument("--force_mode", type=str, default="autograd", choices=["direct", "autograd"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda_force", type=float, default=1.0)
    parser.add_argument("--hidden_dims", type=str, default="64,64")
    parser.add_argument("--activation", type=str, default="tanh", choices=["relu", "tanh", "gelu", "sigmoid"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam", "adamw", "rmsprop"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.train_size = 256
        args.val_size = 128
        args.test_size = 256
        args.epochs = 60
        args.candidate_size = 2000

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_points = sample_uniform(args.val_size, -3.0, 3.0, rng)
    test_points = sample_uniform(args.test_size, -3.0, 3.0, rng)
    extrap_points = sample_extrapolation_ring(
        n=args.test_size,
        inner_low=-3.0,
        inner_high=3.0,
        outer_low=-4.0,
        outer_high=4.0,
        rng=rng,
    )
    val_dataset = to_dataset(val_points)
    test_dataset = to_dataset(test_points)
    extrap_dataset = to_dataset(extrap_points)

    methods = ["uniform", "grid", "lhs", "grad_importance"] if args.compare_sampling else [args.sampling_method]
    results = {}
    sampling_points = {}

    for method in methods:
        train_points = sample_points(
            method=method,
            n=args.train_size,
            low=-3.0,
            high=3.0,
            rng=rng,
            candidates=args.candidate_size,
        )
        sampling_points[method] = train_points
        train_dataset = to_dataset(train_points)

        run = train_once(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            hidden_dims=hidden_dims,
            activation=args.activation,
            dropout=args.dropout,
            force_mode=args.force_mode,
            optimizer_name=args.optimizer,
            lr=args.lr,
            alpha=args.alpha,
            lambda_force=args.lambda_force,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            device=device,
        )

        results[method] = {
            "interpolation_energy_mae": run.test_energy_mae,
            "interpolation_force_mae": run.test_force_mae,
            "epochs_ran": len(run.train_loss),
        }
        extrap_loader = DataLoader(extrap_dataset, batch_size=args.batch_size, shuffle=False)
        ext_e_mae, ext_f_mae = evaluate_mae(run.model, extrap_loader, args.force_mode, device)
        results[method]["extrapolation_energy_mae"] = ext_e_mae
        results[method]["extrapolation_force_mae"] = ext_f_mae
        model_path = os.path.join(args.output_dir, f"model_{method}.pth")
        torch.save(run.model.state_dict(), model_path)
        plot_loss(run.train_loss, run.val_loss, os.path.join(args.output_dir, f"loss_{method}.png"))
        plot_contour_comparison(
            model=run.model,
            mode=args.force_mode,
            device=device,
            out_path=os.path.join(args.output_dir, f"contour_interp_{method}.png"),
            low=-3.0,
            high=3.0,
        )
        plot_contour_comparison(
            model=run.model,
            mode=args.force_mode,
            device=device,
            out_path=os.path.join(args.output_dir, f"contour_extrap_{method}.png"),
            low=-4.0,
            high=4.0,
        )

    plot_sampling(sampling_points, os.path.join(args.output_dir, "sampling_points.png"))
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
