"""Training and evaluation loops for patch-based HSI classification."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch

from .profiling import sync_device


@torch.no_grad()
def evaluate_accuracy(data_iter, net, loss_fn, device) -> Tuple[float, float]:
    """Evaluate patch accuracy and average loss on a DataLoader."""
    net.eval()
    correct = 0.0
    total = 0.0
    loss_sum = 0.0
    batch_count = 0

    for patches, labels, coords in data_iter:
        del coords
        patches = patches.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, aux = net(patches)
        loss = loss_fn(logits, aux, labels)
        pred = logits.argmax(dim=1)
        correct += pred.eq(labels.long()).float().sum().item()
        total += labels.numel()
        loss_sum += float(loss.detach().cpu())
        batch_count += 1

    return correct / max(total, 1.0), loss_sum / max(batch_count, 1)


def train(
    net,
    train_iter,
    valida_iter,
    loss_fn,
    optimizer,
    device,
    epochs: int = 300,
    early_stopping: bool = True,
    early_num: int = 50,
    scheduler_tmax: int = 50,
) -> Dict[str, List[float]]:
    """Train patch-based LST-Mamba.

    Each mini-batch contains local HSI patches and the corresponding center-pixel
    labels.  Timing is synchronized on CUDA so the reported epoch and total
    training times are suitable for experiment tables.
    """
    net = net.to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(scheduler_tmax), 1), eta_min=1e-6
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epoch_time_sec": [],
        "lr": [],
    }
    best_val_loss = float("inf")
    bad_epochs = 0

    print("training on", device)
    sync_device(device)
    start = time.perf_counter()

    for epoch in range(int(epochs)):
        net.train()
        sync_device(device)
        epoch_start = time.perf_counter()
        train_loss_sum = 0.0
        correct = 0.0
        total = 0.0
        batch_count = 0

        for patches, labels, coords in train_iter:
            del coords
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, aux = net(patches)
            loss = loss_fn(logits, aux, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                correct += pred.eq(labels.long()).float().sum().item()
                total += labels.numel()
                train_loss_sum += float(loss.detach().cpu())
                batch_count += 1

        sync_device(device)
        epoch_time = time.perf_counter() - epoch_start
        scheduler.step()

        train_loss = train_loss_sum / max(batch_count, 1)
        train_acc = correct / max(total, 1.0)
        val_acc, val_loss = evaluate_accuracy(valida_iter, net, loss_fn, device)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["epoch_time_sec"].append(float(epoch_time))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"epoch {epoch + 1:03d}, train loss {train_loss:.6f}, "
            f"train acc {train_acc:.4f}, val loss {val_loss:.6f}, "
            f"val acc {val_acc:.4f}, lr {optimizer.param_groups[0]['lr']:.2e}, "
            f"epoch time {epoch_time:.2f}s"
        )

        if early_stopping:
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = float(val_loss)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(early_num):
                    print(f"Early stopping at epoch {epoch + 1}; best val loss {best_val_loss:.6f}.")
                    break

    sync_device(device)
    total_time = time.perf_counter() - start
    history["total_time_sec"] = float(total_time)
    history["num_epochs_ran"] = int(len(history["epoch_time_sec"]))
    history["avg_epoch_time_sec"] = float(
        sum(history["epoch_time_sec"]) / max(len(history["epoch_time_sec"]), 1)
    )
    history["best_val_loss"] = float(best_val_loss)
    history["best_epoch"] = (
        int(min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i]) + 1)
        if history["val_loss"]
        else 0
    )
    print(
        f"training finished, total time {total_time:.2f}s, "
        f"avg epoch time {history['avg_epoch_time_sec']:.2f}s, "
        f"epochs {history['num_epochs_ran']}"
    )
    return history
