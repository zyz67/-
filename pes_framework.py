import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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


def evaluate_weighted_loss(
    model: nn.Module,
    loader: DataLoader,
    mode: str,
    device: torch.device,
    alpha: float,
    lambda_force: float,
) -> float:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_count = 0
    for x, y_e, y_f in loader:
        x = x.to(device)
        y_e = y_e.to(device)
        y_f = y_f.to(device)
        with torch.set_grad_enabled(mode == "autograd"):
            p_e, p_f = predict(model, x, mode, create_graph=False)
            b_loss = alpha * mse(p_e, y_e) + lambda_force * mse(p_f, y_f)
        total_loss += b_loss.item()
        total_count += x.size(0)
    return total_loss / total_count


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


def plot_hidden_depth_vs_test_loss(depths: List[int], losses: List[float], out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(depths, losses, marker="o")
    plt.xlabel("hidden layer count")
    plt.ylabel("test loss")
    plt.xticks(depths)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_optimizer_train_loss(curves: Dict[str, List[float]], out_path: str) -> None:
    plt.figure(figsize=(7, 4))
    for name, values in curves.items():
        plt.plot(values, label=name)
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_multi_strategy_loss_curves(
    train_curves: Dict[str, List[float]],
    val_curves: Dict[str, List[float]],
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, values in train_curves.items():
        axes[0].plot(values, label=name)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("train loss")
    axes[0].legend()
    for name, values in val_curves.items():
        axes[1].plot(values, label=name)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].set_title("val loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def first_epoch_reaching_threshold(values: List[float], threshold: float) -> Optional[int]:
    for i, v in enumerate(values):
        if v <= threshold:
            return i + 1
    return None


def parse_hidden_dims(text: str) -> List[int]:
    dims = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not dims:
        raise ValueError('hidden_dims must be a comma-separated list of integers, e.g., "64,64"')
    return dims


def run_assignment_tasks(
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
    val_dataset: TensorDataset,
    test_dataset: TensorDataset,
    extrap_dataset: TensorDataset,
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    assignment_results: Dict[str, object] = {}
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    extrap_loader = DataLoader(extrap_dataset, batch_size=args.batch_size, shuffle=False)

    # 统一采用 LHS 采样，便于实验可比
    train_points_lhs = sample_points(
        method="lhs",
        n=args.train_size,
        low=-3.0,
        high=3.0,
        rng=rng,
        candidates=args.candidate_size,
    )
    train_dataset_lhs = to_dataset(train_points_lhs)

    # Task 1: 隐藏层数量(1~5) vs 测试 loss
    depth_test_losses: List[float] = []
    depth_results: Dict[str, Dict[str, float]] = {}
    for depth in [1, 2, 3, 4, 5]:
        hidden_dims = [64] * depth
        run = train_once(
            train_dataset=train_dataset_lhs,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            hidden_dims=hidden_dims,
            activation=args.activation,
            dropout=0.0,
            force_mode="autograd",
            optimizer_name="rmsprop",
            lr=args.lr,
            alpha=1.0,
            lambda_force=5.0,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            device=device,
        )
        test_loss = evaluate_weighted_loss(
            model=run.model,
            loader=test_loader,
            mode="autograd",
            device=device,
            alpha=1.0,
            lambda_force=5.0,
        )
        depth_test_losses.append(test_loss)
        depth_results[str(depth)] = {
            "test_loss": test_loss,
            "test_energy_mae": run.test_energy_mae,
            "test_force_mae": run.test_force_mae,
            "epochs_ran": len(run.train_loss),
        }
    plot_hidden_depth_vs_test_loss(
        depths=[1, 2, 3, 4, 5],
        losses=depth_test_losses,
        out_path=os.path.join(args.output_dir, "task1_hidden_depth_vs_test_loss.png"),
    )
    assignment_results["task1_hidden_depth"] = depth_results

    # Task 2: 四种优化器训练 loss 曲线 + 同精度收敛 epoch
    optimizer_runs: Dict[str, TrainResult] = {}
    optimizer_train_curves: Dict[str, List[float]] = {}
    for opt_name in ["sgd", "adam", "adamw", "rmsprop"]:
        run = train_once(
            train_dataset=train_dataset_lhs,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            hidden_dims=parse_hidden_dims(args.hidden_dims),
            activation=args.activation,
            dropout=args.dropout,
            force_mode=args.force_mode,
            optimizer_name=opt_name,
            lr=args.lr,
            alpha=args.alpha,
            lambda_force=args.lambda_force,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            device=device,
        )
        optimizer_runs[opt_name] = run
        optimizer_train_curves[opt_name] = run.train_loss
    plot_optimizer_train_loss(
        curves=optimizer_train_curves,
        out_path=os.path.join(args.output_dir, "task2_optimizers_train_loss.png"),
    )
    common_target = min(run.val_loss[-1] for run in optimizer_runs.values())
    optimizer_metrics: Dict[str, Dict[str, object]] = {}
    for name, run in optimizer_runs.items():
        optimizer_metrics[name] = {
            "final_train_loss": run.train_loss[-1],
            "final_val_loss": run.val_loss[-1],
            "epochs_ran": len(run.train_loss),
            "epoch_reach_common_precision": first_epoch_reaching_threshold(run.val_loss, common_target),
            "test_energy_mae": run.test_energy_mae,
            "test_force_mae": run.test_force_mae,
        }
    best_optimizer = min(optimizer_metrics.items(), key=lambda kv: kv[1]["final_val_loss"])[0]
    assignment_results["task2_optimizer_compare"] = {
        "common_precision_target_val_loss": common_target,
        "metrics": optimizer_metrics,
        "best_optimizer_by_final_val_loss": best_optimizer,
        "discussion": {
            "sgd": "适合对泛化稳定性要求高、可接受较慢收敛、并愿意精细调学习率与动量的场景。",
            "adam": "适合多数默认场景，通常收敛较快，对学习率不敏感，调参成本较低。",
            "adamw": "适合需要权重衰减正则化的场景，通常在泛化与收敛速度间折中较好。",
            "rmsprop": "适合损失地形非平稳或梯度尺度变化较大的问题，常有较快前期下降。",
            "best_optimizer_reason": f"本次实验中 {best_optimizer} 的最终验证 loss 最低，因此表现最好；其自适应学习率机制在当前任务上更有效。",
        },
    }

    # Task 5: 固定参数下在 3~5 层中选最优，作为最终模型
    final_candidates: Dict[str, Dict[str, float]] = {}
    final_model: Optional[nn.Module] = None
    final_depth: Optional[int] = None
    best_val_for_final = float("inf")
    final_run: Optional[TrainResult] = None
    for depth in [3, 4, 5]:
        hidden_dims = [64] * depth
        run = train_once(
            train_dataset=train_dataset_lhs,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            hidden_dims=hidden_dims,
            activation=args.activation,
            dropout=0.0,
            force_mode="autograd",
            optimizer_name="rmsprop",
            lr=args.lr,
            alpha=1.0,
            lambda_force=5.0,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            device=device,
        )
        cur_val = run.val_loss[-1]
        final_candidates[str(depth)] = {
            "final_val_loss": cur_val,
            "test_energy_mae": run.test_energy_mae,
            "test_force_mae": run.test_force_mae,
            "epochs_ran": len(run.train_loss),
        }
        if cur_val < best_val_for_final:
            best_val_for_final = cur_val
            final_depth = depth
            final_model = run.model
            final_run = run
    if final_model is None or final_depth is None or final_run is None:
        raise RuntimeError("Failed to train final model candidates")
    torch.save(final_model.state_dict(), os.path.join(args.output_dir, "model_final_best.pth"))
    plot_loss(
        final_run.train_loss,
        final_run.val_loss,
        os.path.join(args.output_dir, "task5_final_train_val_loss.png"),
    )
    plot_contour_comparison(
        model=final_model,
        mode="autograd",
        device=device,
        out_path=os.path.join(args.output_dir, "task5_contour_interp_final.png"),
        low=-3.0,
        high=3.0,
    )
    plot_contour_comparison(
        model=final_model,
        mode="autograd",
        device=device,
        out_path=os.path.join(args.output_dir, "task5_contour_extrap_final.png"),
        low=-4.0,
        high=4.0,
    )
    final_interp_e_mae, final_interp_f_mae = evaluate_mae(final_model, test_loader, "autograd", device)
    final_extrap_e_mae, final_extrap_f_mae = evaluate_mae(final_model, extrap_loader, "autograd", device)
    assignment_results["task5_final_model"] = {
        "fixed_config": {
            "sampling_method": "lhs",
            "alpha": 1.0,
            "lambda_force": 5.0,
            "force_mode": "autograd",
            "optimizer": "rmsprop",
            "searched_hidden_depth": [3, 4, 5],
            "hidden_units_per_layer": 64,
        },
        "searched_depth_results": final_candidates,
        "best_depth": final_depth,
        "test_metrics": {
            "interpolation_energy_mae": final_interp_e_mae,
            "interpolation_force_mae": final_interp_f_mae,
            "extrapolation_energy_mae": final_extrap_e_mae,
            "extrapolation_force_mae": final_extrap_f_mae,
        },
    }

    # Task 4: 插值/外推 MAE + 等高线图（沿用最终模型）
    assignment_results["task4_interp_extrap_report"] = {
        "interpolation_energy_mae": final_interp_e_mae,
        "interpolation_force_mae": final_interp_f_mae,
        "extrapolation_energy_mae": final_extrap_e_mae,
        "extrapolation_force_mae": final_extrap_f_mae,
        "figures": {
            "interp_contour": "task5_contour_interp_final.png",
            "extrap_contour": "task5_contour_extrap_final.png",
        },
    }

    # Task 3 / q6: 训练-验证曲线 + 过拟合处理 (Dropout / Early Stopping)
    overfit_train_points = sample_points(
        method="lhs",
        n=max(64, args.train_size // 4),
        low=-3.0,
        high=3.0,
        rng=rng,
        candidates=args.candidate_size,
    )
    overfit_train_dataset = to_dataset(overfit_train_points)
    overfit_hidden = [128, 128, 128, 128]
    baseline_run = train_once(
        train_dataset=overfit_train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        hidden_dims=overfit_hidden,
        activation=args.activation,
        dropout=0.0,
        force_mode=args.force_mode,
        optimizer_name=args.optimizer,
        lr=args.lr,
        alpha=args.alpha,
        lambda_force=args.lambda_force,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.epochs + 1,
        device=device,
    )
    dropout_run = train_once(
        train_dataset=overfit_train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        hidden_dims=overfit_hidden,
        activation=args.activation,
        dropout=0.3,
        force_mode=args.force_mode,
        optimizer_name=args.optimizer,
        lr=args.lr,
        alpha=args.alpha,
        lambda_force=args.lambda_force,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.epochs + 1,
        device=device,
    )
    early_stop_run = train_once(
        train_dataset=overfit_train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        hidden_dims=overfit_hidden,
        activation=args.activation,
        dropout=0.0,
        force_mode=args.force_mode,
        optimizer_name=args.optimizer,
        lr=args.lr,
        alpha=args.alpha,
        lambda_force=args.lambda_force,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=max(8, args.patience // 2),
        device=device,
    )
    plot_multi_strategy_loss_curves(
        train_curves={
            "baseline": baseline_run.train_loss,
            "dropout": dropout_run.train_loss,
            "early_stopping": early_stop_run.train_loss,
        },
        val_curves={
            "baseline": baseline_run.val_loss,
            "dropout": dropout_run.val_loss,
            "early_stopping": early_stop_run.val_loss,
        },
        out_path=os.path.join(args.output_dir, "task3_q6_overfit_strategies_loss.png"),
    )
    assignment_results["task3_q6"] = {
        "main_train_val_curve_file": "task5_final_train_val_loss.png",
        "overfit_experiment_curve_file": "task3_q6_overfit_strategies_loss.png",
        "overfit_setting": {
            "train_size": max(64, args.train_size // 4),
            "hidden_dims": overfit_hidden,
        },
        "strategy_metrics": {
            "baseline": {
                "test_energy_mae": baseline_run.test_energy_mae,
                "test_force_mae": baseline_run.test_force_mae,
                "epochs_ran": len(baseline_run.train_loss),
            },
            "dropout": {
                "test_energy_mae": dropout_run.test_energy_mae,
                "test_force_mae": dropout_run.test_force_mae,
                "epochs_ran": len(dropout_run.train_loss),
            },
            "early_stopping": {
                "test_energy_mae": early_stop_run.test_energy_mae,
                "test_force_mae": early_stop_run.test_force_mae,
                "epochs_ran": len(early_stop_run.train_loss),
            },
        },
        "qa": {
            "q1_validation_set_role": "除早停外，验证集还用于模型选择、超参数选择、学习率/正则强度对比、监控训练稳定性以及发现数据分布漂移。",
            "q2_training_state_diagnosis": "训练与验证 loss 都高且下降慢通常是欠拟合；两者都低且接近通常是适当拟合；训练 loss 持续下降但验证 loss 回升通常是过拟合。本实验可结合 `task3_q6_overfit_strategies_loss.png` 观察。",
            "q3_overfit_mitigation": "Dropout 通过抑制共适应降低过拟合；早停通过在验证集不再改进时停止训练减少过拟合累积。请结合 strategy_metrics 比较其对测试误差的影响。",
        },
    }

    # 生成汇总文件
    result_json_path = os.path.join(args.output_dir, "assignment_results.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(assignment_results, f, ensure_ascii=False, indent=2)
    report_lines = [
        "# Assignment Auto Report",
        "",
        "## 1) Hidden depth vs test loss",
        f"- Figure: `task1_hidden_depth_vs_test_loss.png`",
        "",
        "## 2) Optimizer comparison",
        "- Figure: `task2_optimizers_train_loss.png`",
        f"- Common precision target (val loss): {assignment_results['task2_optimizer_compare']['common_precision_target_val_loss']}",
        f"- Best optimizer: {assignment_results['task2_optimizer_compare']['best_optimizer_by_final_val_loss']}",
        "",
        "## 3) q6 train/val curves and overfitting handling",
        "- Main curve: `task5_final_train_val_loss.png`",
        "- Overfitting strategies curve: `task3_q6_overfit_strategies_loss.png`",
        "",
        "## 4) Interpolation/Extrapolation MAE + contour",
        "- Figures: `task5_contour_interp_final.png`, `task5_contour_extrap_final.png`",
        "",
        "## 5) Final model",
        f"- Best depth in [3,4,5]: {assignment_results['task5_final_model']['best_depth']}",
        f"- Test metrics: {assignment_results['task5_final_model']['test_metrics']}",
    ]
    with open(os.path.join(args.output_dir, "assignment_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(json.dumps(assignment_results, ensure_ascii=False, indent=2))


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
    parser.add_argument("--run_assignment", action="store_true")
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

    if args.run_assignment:
        run_assignment_tasks(
            args=args,
            rng=rng,
            device=device,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            extrap_dataset=extrap_dataset,
        )
        return

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
