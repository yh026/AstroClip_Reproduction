import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast

from data import DataConfig, SpectrumDataModule
from model import SpecFormer, SpecFormerConfig
from scheduler import CosineAnnealingWithWarmupLR


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(self.count, 1)

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


@torch.no_grad()
def evaluate(model, loader, device, max_batches: Optional[int] = None):
    model.eval()
    loss_meter = AverageMeter()
    masked_meter = AverageMeter()
    cached_batch = None

    for i, batch in enumerate(loader):
        spectrum = batch["spectrum"].to(device, non_blocking=True)
        loss, stats, cache = model.compute_loss(spectrum)
        bsz = spectrum.size(0)
        loss_meter.update(stats["loss"], bsz)
        masked_meter.update(stats["masked_fraction"], bsz)
        if cached_batch is None:
            cached_batch = cache
        if max_batches is not None and i + 1 >= max_batches:
            break

    return {
        "loss": loss_meter.avg,
        "masked_fraction": masked_meter.avg,
        "cache": cached_batch,
    }


def flatten_for_plot(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:, 2:].reshape(x.size(0), -1)


def save_reconstruction_plot(cache: Dict[str, torch.Tensor], out_path: Path, sample_id: int = 0, win: int = 20):
    target = flatten_for_plot(cache["target"]).cpu()[sample_id]
    masked = flatten_for_plot(cache["masked_input"]).cpu()[sample_id]
    recon = flatten_for_plot(cache["reconstructions"]).cpu()[sample_id]

    def moving_avg(x: torch.Tensor, win: int):
        return [x[i : i + win].mean().item() for i in range(0, len(x), win)]

    plt.figure(figsize=(10, 4))
    plt.plot(moving_avg(target, win), label="original")
    plt.plot(moving_avg(masked, win), label="masked")
    plt.plot(moving_avg(recon, win), label="reconstructed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_loss_curves(history: Dict[str, List[float]], out_path: Path):
    plt.figure(figsize=(8, 5))
    epochs = list(range(1, len(history["train_loss_epoch"]) + 1))
    if history["train_loss_epoch"]:
        plt.plot(epochs, history["train_loss_epoch"], label="train")
    if history["val_loss_epoch"]:
        plt.plot(epochs[: len(history["val_loss_epoch"])], history["val_loss_epoch"], label="val")
    if history.get("test_loss"):
        plt.axhline(history["test_loss"][-1], linestyle="--", label="test_final")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


@torch.no_grad()
def export_layer_embeddings(
    model: SpecFormer,
    loader,
    device,
    out_dir: Path,
    num_batches: int = 1,
    sample_limit: int = 8,
):
    model.eval()
    batch = next(iter(loader))
    spectrum = batch["spectrum"].to(device, non_blocking=True)
    spectrum = spectrum[:sample_limit]

    preprocessed = model.preprocess(spectrum)
    outputs = model.encode(preprocessed, preprocessed=True, return_all_layers=True)

    export = {
        "preprocessed_input": preprocessed.detach().cpu(),
        "token_embedding": outputs["token_embedding"].detach().cpu(),
        "final_embedding": outputs["embedding"].detach().cpu(),
        "all_layer_embeddings": [x.detach().cpu() for x in outputs["all_layer_embeddings"]],
    }
    torch.save(export, out_dir / "layer_embeddings.pt")

    rows = []
    rows.append(["token_embedding", *export["token_embedding"].shape, float(export["token_embedding"].norm(dim=-1).mean())])
    for idx, emb in enumerate(export["all_layer_embeddings"], start=1):
        rows.append([f"block_{idx}", *emb.shape, float(emb.norm(dim=-1).mean())])
    rows.append(["final_embedding", *export["final_embedding"].shape, float(export["final_embedding"].norm(dim=-1).mean())])

    with open(out_dir / "layer_embedding_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "batch", "tokens", "dim", "mean_token_l2_norm"])
        writer.writerows(rows)


@torch.no_grad()
def export_parameter_stats(model: SpecFormer, out_dir: Path):
    params_dir = out_dir / "parameters"
    params_dir.mkdir(parents=True, exist_ok=True)

    state = model.state_dict()
    torch.save(state, out_dir / "model_state_dict.pt")

    with open(out_dir / "parameter_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "shape",
                "numel",
                "mean",
                "std",
                "min",
                "max",
                "abs_mean",
                "l2_norm",
                "has_nan",
                "has_inf",
            ]
        )
        for name, tensor in state.items():
            t = tensor.detach().float().cpu()
            writer.writerow(
                [
                    name,
                    list(t.shape),
                    t.numel(),
                    float(t.mean()),
                    float(t.std(unbiased=False)),
                    float(t.min()),
                    float(t.max()),
                    float(t.abs().mean()),
                    float(t.norm()),
                    bool(torch.isnan(t).any()),
                    bool(torch.isinf(t).any()),
                ]
            )
            safe_name = name.replace(".", "__")
            torch.save(t, params_dir / f"{safe_name}.pt")


@torch.no_grad()
def run_test_and_export(model: SpecFormer, loader, device, out_dir: Path, cfg: Dict, history: Dict[str, List[float]]):
    test_dir = out_dir / "test_results"
    test_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate(model, loader, device, max_batches=None)
    history.setdefault("test_loss", []).append(metrics["loss"])
    history.setdefault("test_masked_fraction", []).append(metrics["masked_fraction"])

    if metrics["cache"] is not None:
        save_reconstruction_plot(metrics["cache"], test_dir / "test_reconstruction.png")

    save_json(
        {
            "test_loss": metrics["loss"],
            "test_masked_fraction": metrics["masked_fraction"],
        },
        test_dir / "test_metrics.json",
    )

    export_layer_embeddings(
        model,
        loader,
        device,
        test_dir,
        num_batches=cfg["analysis"]["embedding_batches"],
        sample_limit=cfg["analysis"]["embedding_sample_limit"],
    )
    export_parameter_stats(model, test_dir)


def save_checkpoint(
    path: Path,
    model: SpecFormer,
    optimizer,
    scheduler,
    scaler: GradScaler,
    cfg: Dict,
    history: Dict,
    epoch: int,
    global_step: int,
    best_val: float,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg,
            "history": history,
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
        },
        path,
    )


def try_resume(checkpoint_path: Optional[Path], model, optimizer, scheduler, scaler):
    if checkpoint_path is None or not checkpoint_path.exists():
        return 0, 0, math.inf, {
            "train_loss_epoch": [],
            "val_loss_epoch": [],
            "lr_epoch_end": [],
            "train_masked_fraction_epoch": [],
            "val_masked_fraction_epoch": [],
            "test_loss": [],
            "test_masked_fraction": [],
        }

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    history = ckpt.get("history", {})
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    global_step = int(ckpt.get("global_step", 0))
    best_val = float(ckpt.get("best_val", math.inf))
    print(f"Resumed from {checkpoint_path} at epoch={start_epoch}, global_step={global_step}")
    return start_epoch, global_step, best_val, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_cfg = DataConfig(**cfg["data"])
    dm = SpectrumDataModule(data_cfg)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = val_loader

    model_cfg = SpecFormerConfig(**cfg["model"])
    model = SpecFormer(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )

    max_steps = cfg["schedule"]["max_steps"]
    eta_min = cfg["optim"]["lr"] * cfg["schedule"]["eta_min_ratio"]
    scheduler = CosineAnnealingWithWarmupLR(
        optimizer,
        T_max=max_steps,
        T_warmup=cfg["schedule"]["warmup_steps"],
        eta_min=eta_min,
    )

    use_amp = bool(cfg["train"].get("use_amp", True) and device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    resume_path = Path(args.resume) if args.resume is not None else (output_dir / "last.pt")
    start_epoch, global_step, best_val, history = try_resume(resume_path, model, optimizer, scheduler, scaler)

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        train_loss_meter = AverageMeter()
        train_masked_meter = AverageMeter()

        for batch_idx, batch in enumerate(train_loader):
            spectrum = batch["spectrum"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                loss, stats, _ = model.compute_loss(spectrum)

            scaler.scale(loss).backward()
            if cfg["optim"]["grad_clip"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            train_loss_meter.update(stats["loss"], spectrum.size(0))
            train_masked_meter.update(stats["masked_fraction"], spectrum.size(0))

            if global_step % cfg["train"]["log_every"] == 0:
                print(
                    f"epoch={epoch} step={global_step} batch={batch_idx} "
                    f"train_loss={train_loss_meter.avg:.6f} "
                    f"masked_fraction={train_masked_meter.avg:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.8e}"
                )

            if global_step >= max_steps:
                print("Reached max_steps, stopping training.")
                break

        val_metrics = evaluate(model, val_loader, device, max_batches=None)
        history.setdefault("train_loss_epoch", []).append(train_loss_meter.avg)
        history.setdefault("val_loss_epoch", []).append(val_metrics["loss"])
        history.setdefault("lr_epoch_end", []).append(float(optimizer.param_groups[0]["lr"]))
        history.setdefault("train_masked_fraction_epoch", []).append(train_masked_meter.avg)
        history.setdefault("val_masked_fraction_epoch", []).append(val_metrics["masked_fraction"])

        print(
            f"[EPOCH {epoch}] train_loss={train_loss_meter.avg:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"train_masked={train_masked_meter.avg:.4f} "
            f"val_masked={val_metrics['masked_fraction']:.4f}"
        )

        if val_metrics["cache"] is not None:
            save_reconstruction_plot(val_metrics["cache"], output_dir / f"val_reconstruction_epoch_{epoch:03d}.png")
        save_loss_curves(history, output_dir / "loss_curves.png")
        save_json(history, output_dir / "history.json")

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            cfg,
            history,
            epoch,
            global_step,
            best_val,
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                cfg,
                history,
                epoch,
                global_step,
                best_val,
            )

        if (epoch + 1) % cfg["train"]["save_every_epochs"] == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                cfg,
                history,
                epoch,
                global_step,
                best_val,
            )

        if global_step >= max_steps:
            break

    best_ckpt_path = output_dir / "best.pt"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
        model.load_state_dict(best_ckpt["model"])
        print(f"Loaded best checkpoint from {best_ckpt_path} for final testing.")

    run_test_and_export(model, test_loader, device, output_dir, cfg, history)
    save_loss_curves(history, output_dir / "loss_curves.png")
    save_json(history, output_dir / "history.json")
    print(f"Training + final test complete. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
