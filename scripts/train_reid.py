
"""
Stage 6 — per-video re-ID head: FROZEN DINOv2 backbone + trained classifier head.

Per configured stretch: build fingerprints (frozen backbone) -> temporal train/val split ->
train the head -> eval (per-fish accuracy + confusion) -> log to MLflow. Stretches never pooled.

Usage:
    python -m scripts.train_reid --video_name IMG_1839
"""
#============================================================
# IMPORTS
#============================================================
import torch
from torch.utils.data import DataLoader, TensorDataset # TensorDataset lets us batch tensors directly (feats already cached, no preprocessing)
import os
import logging
import yaml
import argparse
import subprocess
import datetime
import matplotlib
matplotlib.use("Agg")            # headless — save figures, never open a window
import matplotlib.pyplot as plt
from scripts.console import banner, banner_sub  # readable console section headers
import torch.nn as nn
from scripts.reid_features import build_features   # shared: crops -> DINOv2 fingerprints (frozen backbone)

logger = logging.getLogger(__name__)  # module logger; setup_logging() configures format/level in the entry point

CONFIG_PATH = 'config.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)['train_reid']   # this script's config section

#============================================================
# CONFIG LOADER
#============================================================

def grab_video_name(video_name):
    "pull this video's re-ID params from config.yaml"
    video_cfg     = cfg['videos'][video_name]
    crops_run     = video_cfg['crops_run']
    stretches     = video_cfg['stretches']           # which curated stretch(es) to train on (identity-safety)
    backbone_name = video_cfg['backbone']
    num_epochs    = video_cfg['num_epochs']
    lr            = video_cfg['lr']
    batch_size    = video_cfg['batch_size']
    banner('LOADING CONFIGURATION')
    logger.info(f"loaded cfg: {video_cfg}")
    return crops_run, stretches, backbone_name, num_epochs, lr, batch_size

#============================================================
# MLFLOW LOGGING (one run = params + metrics + git state)
#============================================================

def log_to_mlflow(params, metrics, sweep_id, train_losses, val_accs, fig):
    """log this re-ID run's params + metrics + git state to MLflow — matches evaluate_tracker's pattern.
       Also logs the per-epoch curves (step metrics) + the overfitting figure.
       sweep_id groups all stretches launched by the SAME command into one identifiable batch."""
    import mlflow
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlflow.set_tracking_uri(os.path.join(repo_root, "mlruns"))   # local .../AquaMind/mlruns
    mlflow.set_experiment("aquamind_reid")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    dirty  = bool(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip())  # True = uncommitted edits
    with mlflow.start_run(run_name=f"{sweep_id}__stretch{params['stretch']}"):   # readable name in the UI
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for e, (tl, va) in enumerate(zip(train_losses, val_accs)):   # per-epoch curves (view in the MLflow UI)
            mlflow.log_metric("epoch_train_loss", tl, step=e)
            mlflow.log_metric("epoch_val_acc",   va, step=e)
        mlflow.log_figure(fig, f"overfit_curve_stretch{params['stretch']}.png")
        mlflow.set_tag("sweep_id", sweep_id)    # all stretches from ONE command share this -> filter/group by it
        mlflow.set_tag("git_commit", commit)
        mlflow.set_tag("git_dirty", dirty)      # so future-you knows this run had unsaved changes
    plt.close(fig)
    logger.info(f"logged to MLflow (aquamind_reid, sweep={sweep_id}, stretch={params['stretch']}, commit={commit[:8]}, dirty={dirty})")

#============================================================
# MAIN
#============================================================

def run_one_stretch(crops_run, stretch, backbone_name, num_epochs, lr, batch_size, sweep_id):
    """Train + evaluate a head on ONE stretch and log it to MLflow. Called once per configured stretch."""

    banner(f"FISH RE-ID — STRETCH {stretch}")

    # ---------- features: build for THIS stretch only (keeps stretches independent, no pooling) ----------
    all_feats, all_labels, all_frames, label_map = build_features(crops_run, [stretch], backbone_name)

    n_classes = len(label_map)          # number of fish (NOT hardcoded — comes from the data)
    feat_dim  = all_feats.shape[1]      # 384

    # ---------- TEMPORAL train/val split (early frames = train, late frames = val) ----------
    # sort by frame, take the earliest TRAIN_FRAC as train and the rest as val.
    # Splitting by TIME (not randomly) stops near-duplicate consecutive frames leaking
    # from train into val — the only honest way to tell recognition from memorization.
    banner_sub("TEMPORAL TRAIN/VAL SPLIT")
    TRAIN_FRAC = 0.7
    order   = torch.argsort(all_frames)                 # indices sorted early -> late
    n_train = int(len(order) * TRAIN_FRAC)
    train_idx, val_idx = order[:n_train], order[n_train:]
    train_feats, train_labels = all_feats[train_idx], all_labels[train_idx]
    val_feats,   val_labels   = all_feats[val_idx],   all_labels[val_idx]
    logger.info(f"train: {len(train_idx)} crops (frames {all_frames[train_idx].min()}..{all_frames[train_idx].max()})")
    logger.info(f"val:   {len(val_idx)} crops (frames {all_frames[val_idx].min()}..{all_frames[val_idx].max()})")

    # ---------- head + loss + optimizer (build once) ----------
    head = nn.Linear(feat_dim, n_classes)   # (in-size, out-size) fingerprint -> one score per fish
    head.train()                            # head in training mode
    loss_fin = nn.CrossEntropyLoss()        # the grader: takes raw logits, does softmax internally
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)   # nudges ONLY the head

    # ---------- training loop over the TRAIN split only ----------
    banner_sub("TRAINING THE HEAD")
    feat_ds = TensorDataset(train_feats, train_labels)                   # TRAIN crops only (val is held out)
    feat_loader = DataLoader(feat_ds, batch_size=batch_size, shuffle=True)   # batches of (fingerprint, label)

    train_losses, val_accs = [], []
    for epoch in range(num_epochs):
        head.train()
        running_loss = 0.0 # accumulator — reset each epoch (reporting only)
        for feats, labels in feat_loader: # batches of CACHED fingerprints — no crops, no backbone -> instant
            logits = head(feats) # fingerprints -> N scores (THIS has gradients)
            loss = loss_fin(logits, labels) # how wrong?
            optimizer.zero_grad() # reset old gradients
            loss.backward() # backprop: which way to nudge each head weight
            optimizer.step() # optimizer: actually nudge them
            running_loss += loss.item() # add this batch's loss into the bucket
        train_loss = running_loss / len(feat_loader)
        head.eval()                                     # val every epoch (cheap — head is tiny, feats cached) -> the curve
        with torch.no_grad():
            val_acc = (head(val_feats).argmax(dim=1) == val_labels).float().mean().item()
        train_losses.append(train_loss); val_accs.append(val_acc)
        logger.info(f"epoch {epoch+1}/{num_epochs}  train_loss {train_loss:.4f}  val_acc {val_acc:.3f}")

    best_epoch = int(torch.tensor(val_accs).argmax()) + 1
    logger.info(f"BEST val_acc {max(val_accs):.3f} @ epoch {best_epoch}  |  final {val_accs[-1]:.3f}  |  gap = {max(val_accs) - val_accs[-1]:.3f} (overfitting if >0)")

    # ---------- overfitting curve: train loss (down) vs val accuracy (peaks then droops if overfitting) ----------
    xs = range(1, num_epochs + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(xs, train_losses, "r-o", ms=3, label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="r"); ax1.tick_params(axis="y", labelcolor="r")
    ax2 = ax1.twinx()
    ax2.plot(xs, val_accs, "b-o", ms=3, label="val accuracy")
    ax2.set_ylabel("val accuracy", color="b"); ax2.tick_params(axis="y", labelcolor="b")
    ax2.axvline(best_epoch, color="gray", ls="--", lw=1)   # mark the best epoch (early-stop point)
    ax1.set_title(f"stretch{stretch} frozen head — train loss vs val accuracy (best @ epoch {best_epoch})")
    fig.tight_layout()

    # ----------  EVALUATE on the held-out VAL split (the honest number) ----------
    banner_sub("VALIDATION — ACCURACY ON UNSEEN LATE FRAMES")
    head.eval()                                         # inference mode
    with torch.no_grad():
        val_preds = head(val_feats).argmax(dim=1)       # highest-scoring fish per crop = the guess
    correct = (val_preds == val_labels)                 # boolean per crop: right or wrong
    acc = correct.float().mean().item()                 # overall rank-1 accuracy
    logger.info(f"OVERALL val accuracy: {correct.sum().item()}/{len(val_labels)} = {acc:.3f}")

    metrics = {"val_accuracy": acc, "best_val_accuracy": max(val_accs), "best_epoch": best_epoch}   # collect for MLflow
    slot_to_fish = {slot: fish for fish, slot in label_map.items()}   # invert map for readable output
    for slot in range(n_classes):                       # per-fish breakdown
        mask = val_labels == slot
        if mask.sum() > 0:
            fish_acc = correct[mask].float().mean().item()
            metrics[f"acc_fish_{slot_to_fish[slot]}"] = fish_acc
            logger.info(f"  fish {slot_to_fish[slot]} (slot {slot}): {correct[mask].sum().item()}/{mask.sum().item()} = {fish_acc:.3f}")

    # confusion: for each TRUE fish, what did the head PREDICT? (points you at the look-alike / swap partner)
    banner_sub("CONFUSION — true fish -> what the head guessed")
    for slot in range(n_classes):
        mask = val_labels == slot
        if mask.sum() > 0:
            guessed = val_preds[mask]                    # the head's guesses for this true fish's val crops
            counts = {slot_to_fish[g]: int((guessed == g).sum()) for g in guessed.unique().tolist()}
            logger.info(f"  true fish {slot_to_fish[slot]} was called: {counts}")

    # ---------- log this run (params + metrics + git state) to MLflow ----------
    params = {
        "crops_run":  os.path.basename(crops_run),
        "stretch":    stretch,
        "backbone":   backbone_name,
        "num_epochs": num_epochs,
        "lr":         lr,
        "batch_size": batch_size,
        "feat_dim":   feat_dim,
        "n_classes":  n_classes,
        "n_train":    n_train,
        "n_val":      len(val_idx),
    }
    log_to_mlflow(params, metrics, sweep_id, train_losses, val_accs, fig)

    # ---------- save the trained head so the TRACKER can use it as a discriminative appearance metric ----------
    head_path = f"{crops_run}/reid_head_stretch{int(stretch):02d}.pt"
    torch.save({
        "head_state": head.state_dict(),   # the nn.Linear(feat_dim, n_classes) weights
        "feat_dim":   feat_dim,
        "n_classes":  n_classes,
        "label_map":  label_map,
        "backbone":   backbone_name,        # tracker must use the SAME backbone
        "stretch":    stretch,
    }, head_path)
    logger.info(f"saved head -> {head_path}  (use as tracker appearance_head)")

#============================================================
# MAIN — run each configured stretch SEPARATELY (never pooled: cross-stretch fish IDs aren't verified)
#============================================================

def main(crops_run, stretches, backbone_name, num_epochs, lr, batch_size):
    sweep_id = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")   # one id shared by every stretch in THIS command
    banner(f"RE-ID over {len(stretches)} stretch(es): {stretches}  (sweep {sweep_id})")
    for stretch in stretches:                                 # one independent train+eval+MLflow run per stretch
        run_one_stretch(crops_run, stretch, backbone_name, num_epochs, lr, batch_size, sweep_id)

#============================================================
# ENTRY POINT
#============================================================

if __name__ == "__main__":
    from scripts.logger import setup_logging   # configures level (LOG_LEVEL env) + format
    setup_logging()                            # un-mutes logger.info so the output shows

    parser = argparse.ArgumentParser(description="Train per-video re-ID head")
    parser.add_argument("--video_name", default=cfg['video'], help="key under train_reid.videos in config.yaml")
    args = parser.parse_args()
    main(*grab_video_name(args.video_name))    # unpack the config tuple straight into main()
