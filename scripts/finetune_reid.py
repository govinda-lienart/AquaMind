"""
Stage 6.4 — FINE-TUNED re-ID: unfreeze DINOv2 + train it end-to-end with the head.

Unlike train_reid (FROZEN backbone), here the backbone LEARNS — so it runs INSIDE the training loop
over raw crops and gradients flow through it. Goal: teach the features to separate look-alike
individuals (e.g. stretch02 fish3/fish5) that frozen generic features can't.

Per configured stretch: crop dataloader (augmented train / plain val, temporal split) ->
unfreeze last N blocks -> train (backbone + head together) -> eval (per-fish + confusion) ->
log to MLflow (mode=finetune, same experiment as train_reid so frozen vs finetune compare). Never pooled.

Usage: python -m scripts.finetune_reid --video_name IMG_1839
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
import matplotlib
matplotlib.use("Agg")            # headless — save figures, never open a window
import matplotlib.pyplot as plt
import re
import os
import logging
import yaml
import argparse
import subprocess
import datetime
from scripts.reid_features import FishCropDataset, load_backbone, transform as EVAL_TF
from scripts.console import banner, banner_sub

logger = logging.getLogger(__name__)

CONFIG_PATH = 'config.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)['finetune_reid']

# augmentation for TRAINING crops only (val uses the plain EVAL_TF from reid_features).
# brightness only — NO hue/saturation jitter (that erases the pigment signal that separates same-morph fish).
TRAIN_TF = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#============================================================
# CONFIG LOADER
#============================================================

def grab_video_name(video_name):
    "pull this video's fine-tune params from config.yaml"
    v = cfg['videos'][video_name]
    banner('LOADING CONFIGURATION')
    logger.info(f"loaded cfg: {v}")
    return (v['crops_run'], v['stretches'], v['backbone'], v['num_epochs'],
            v['head_lr'], v['backbone_lr'], v['batch_size'], v['unfreeze_blocks'])

#============================================================
# EVAL HELPER
#============================================================

def evaluate(backbone, head, loader, device):
    """Run a loader through backbone+head (no grad), return (preds, labels) on CPU."""
    backbone.eval(); head.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for tensors, labels, _frames in loader:
            preds = head(backbone(tensors.to(device))).argmax(dim=1).cpu()
            preds_all.append(preds); labels_all.append(labels)
    return torch.cat(preds_all), torch.cat(labels_all)

#============================================================
# MLFLOW LOGGING
#============================================================

def log_to_mlflow(params, metrics, sweep_id, train_losses, val_accs, fig):
    """log run to MLflow in the SAME experiment as train_reid, so frozen vs finetune sit in one table.
       Also logs the per-epoch curves (step metrics) + the overfitting figure.
       NOTE: near-duplicate of train_reid.log_to_mlflow — extract to a shared mlflow_utils when convenient."""
    import mlflow
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlflow.set_tracking_uri(os.path.join(repo_root, "mlruns"))
    mlflow.set_experiment("aquamind_reid")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    dirty  = bool(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip())
    with mlflow.start_run(run_name=f"{sweep_id}__finetune__stretch{params['stretch']}"):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for e, (tl, va) in enumerate(zip(train_losses, val_accs)):   # per-epoch curves (view in the MLflow UI)
            mlflow.log_metric("epoch_train_loss", tl, step=e)
            mlflow.log_metric("epoch_val_acc",   va, step=e)
        mlflow.log_figure(fig, f"overfit_curve_stretch{params['stretch']}.png")   # the figure as an artifact
        mlflow.set_tag("sweep_id", sweep_id)
        mlflow.set_tag("git_commit", commit)
        mlflow.set_tag("git_dirty", dirty)
    plt.close(fig)
    logger.info(f"logged to MLflow (aquamind_reid, sweep={sweep_id}, stretch={params['stretch']}, commit={commit[:8]}, dirty={dirty})")

#============================================================
# ONE STRETCH — FINE-TUNE
#============================================================

def run_one_stretch(crops_run, stretch, backbone_name, num_epochs, head_lr, backbone_lr, batch_size, unfreeze_blocks, sweep_id):
    banner(f"FINETUNE RE-ID — STRETCH {stretch}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"device: {device}")

    # ---------- data: two views over the SAME crops (augmented train / plain val), TEMPORAL split ----------
    globs = [f"{crops_run}/curated_crops/stretch{int(stretch):02d}_fish*/*.jpg"]
    train_source = FishCropDataset(globs, tf=TRAIN_TF)   # augmented
    val_source   = FishCropDataset(globs, tf=EVAL_TF)    # plain (identical paths/order to train_source)
    label_map = train_source.label_map
    n_classes = len(label_map)
    logger.info(f"total crops: {len(train_source)}  |  label map: {label_map}")

    frames  = torch.tensor([int(re.search(r'frame_(\d+)', p).group(1)) for p in train_source.paths])
    order   = torch.argsort(frames)                      # early -> late
    n_train = int(len(order) * 0.7)
    train_idx = order[:n_train].tolist()
    val_idx   = order[n_train:].tolist()
    train_loader = DataLoader(Subset(train_source, train_idx), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(val_source,   val_idx),   batch_size=batch_size, shuffle=False)
    logger.info(f"train {len(train_idx)} crops  |  val {len(val_idx)} crops (temporal split)")

    # ---------- model: DINOv2 with the last N blocks UNFROZEN + a trainable head ----------
    backbone = load_backbone(backbone_name, device=device)
    for p in backbone.parameters():                       # freeze everything first
        p.requires_grad = False
    for blk in list(backbone.blocks)[-unfreeze_blocks:]:  # unfreeze the last N transformer blocks
        for p in blk.parameters():
            p.requires_grad = True
    if hasattr(backbone, "norm"):                         # + the final norm
        for p in backbone.norm.parameters():
            p.requires_grad = True

    feat_dim = getattr(backbone, "embed_dim", None)
    if feat_dim is None:                                  # fallback: infer from a dummy forward
        with torch.no_grad():
            feat_dim = backbone(torch.zeros(1, 3, 224, 224, device=device)).shape[1]
    head = nn.Linear(feat_dim, n_classes).to(device)

    trainable_backbone = any(p.requires_grad for p in backbone.parameters())   # derived — self-truthful
    n_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad) + sum(p.numel() for p in head.parameters())
    logger.info(f"feat_dim {feat_dim}  |  unfrozen blocks {unfreeze_blocks}  |  trainable params {n_trainable:,}")

    optimizer = torch.optim.Adam([
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": backbone_lr},  # small — protect pretrained weights
        {"params": head.parameters(), "lr": head_lr},                                          # larger — fresh head
    ])
    loss_fn = nn.CrossEntropyLoss()

    # ---------- train (backbone + head end-to-end), tracking the curves each epoch ----------
    banner_sub("FINE-TUNING (backbone + head)")
    train_losses, val_accs = [], []
    for epoch in range(num_epochs):
        backbone.train(); head.train()
        running = 0.0
        for tensors, labels, _frames in train_loader:
            tensors, labels = tensors.to(device), labels.to(device)
            feats  = backbone(tensors)          # TRAINABLE — gradients flow into the unfrozen blocks
            logits = head(feats)
            loss   = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)
        vp, vl = evaluate(backbone, head, val_loader, device)   # val EVERY epoch (small, cheap) -> the curve
        val_acc = (vp == vl).float().mean().item()
        train_losses.append(train_loss); val_accs.append(val_acc)
        logger.info(f"epoch {epoch+1}/{num_epochs}  train_loss {train_loss:.4f}  val_acc {val_acc:.3f}")

    # ---------- final eval + best-epoch (the overfitting signal) ----------
    banner_sub("VALIDATION — ACCURACY ON UNSEEN LATE FRAMES")
    val_preds, val_labels = evaluate(backbone, head, val_loader, device)   # final-epoch state
    correct = (val_preds == val_labels)
    acc = correct.float().mean().item()
    best_epoch = int(torch.tensor(val_accs).argmax()) + 1
    best_val   = max(val_accs)
    logger.info(f"OVERALL val accuracy (final epoch): {correct.sum().item()}/{len(val_labels)} = {acc:.3f}")
    logger.info(f"BEST val_acc {best_val:.3f} @ epoch {best_epoch}  |  final {val_accs[-1]:.3f}  |  gap = {best_val - val_accs[-1]:.3f} (overfitting if >0)")

    # ---------- overfitting curve: train loss (down) vs val accuracy (peaks then droops) ----------
    xs = range(1, num_epochs + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(xs, train_losses, "r-o", ms=3, label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="r"); ax1.tick_params(axis="y", labelcolor="r")
    ax2 = ax1.twinx()
    ax2.plot(xs, val_accs, "b-o", ms=3, label="val accuracy")
    ax2.set_ylabel("val accuracy", color="b"); ax2.tick_params(axis="y", labelcolor="b")
    ax2.axvline(best_epoch, color="gray", ls="--", lw=1)   # mark the best epoch (early-stop point)
    ax1.set_title(f"stretch{stretch} finetune — train loss vs val accuracy (best @ epoch {best_epoch})")
    fig.tight_layout()

    metrics = {"val_accuracy": acc, "best_val_accuracy": best_val, "best_epoch": best_epoch}
    slot_to_fish = {slot: fish for fish, slot in label_map.items()}
    for slot in range(n_classes):
        mask = val_labels == slot
        if mask.sum() > 0:
            fish_acc = correct[mask].float().mean().item()
            metrics[f"acc_fish_{slot_to_fish[slot]}"] = fish_acc
            logger.info(f"  fish {slot_to_fish[slot]} (slot {slot}): {correct[mask].sum().item()}/{mask.sum().item()} = {fish_acc:.3f}")

    banner_sub("CONFUSION — true fish -> what the head guessed")
    for slot in range(n_classes):
        mask = val_labels == slot
        if mask.sum() > 0:
            guessed = val_preds[mask]
            counts = {slot_to_fish[g]: int((guessed == g).sum()) for g in guessed.unique().tolist()}
            logger.info(f"  true fish {slot_to_fish[slot]} was called: {counts}")

    # ---------- log to MLflow ----------
    params = {
        "mode":               "finetune",
        "crops_run":          os.path.basename(crops_run),
        "stretch":            stretch,
        "backbone":           backbone_name,
        "unfreeze_blocks":    unfreeze_blocks,
        "trainable_backbone": trainable_backbone,   # derived from requires_grad
        "n_trainable_params": n_trainable,
        "num_epochs":         num_epochs,
        "head_lr":            head_lr,
        "backbone_lr":        backbone_lr,
        "batch_size":         batch_size,
        "feat_dim":           feat_dim,
        "n_classes":          n_classes,
        "n_train":            len(train_idx),
        "n_val":              len(val_idx),
        "device":             device,
    }
    log_to_mlflow(params, metrics, sweep_id, train_losses, val_accs, fig)

#============================================================
# MAIN — run each configured stretch SEPARATELY (never pooled)
#============================================================

def main(crops_run, stretches, backbone_name, num_epochs, head_lr, backbone_lr, batch_size, unfreeze_blocks):
    sweep_id = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    banner(f"FINETUNE RE-ID over {len(stretches)} stretch(es): {stretches}  (sweep {sweep_id})")
    for stretch in stretches:
        run_one_stretch(crops_run, stretch, backbone_name, num_epochs, head_lr, backbone_lr, batch_size, unfreeze_blocks, sweep_id)

#============================================================
# ENTRY POINT
#============================================================

if __name__ == "__main__":
    from scripts.logger import setup_logging
    setup_logging()
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 for per-video re-ID")
    parser.add_argument("--video_name", default=cfg['video'], help="key under finetune_reid.videos in config.yaml")
    args = parser.parse_args()
    main(*grab_video_name(args.video_name))
