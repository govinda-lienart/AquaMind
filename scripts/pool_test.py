"""
Cheap POOLING test — does pooling several manually-verified (no-swap) stretches lift accuracy,
especially the fish3/fish5 wall? Frozen DINOv2 + head (fast). Per-stretch temporal split so train
AND val both contain every stretch.

*** ASSUMPTION *** the stretches in config `train_reid.videos.<name>.stretches` have VERIFIED
cross-stretch ID consistency — fish_N is the SAME physical fish across all of them (your no-swap
windows + a check of the gaps between them). If a silent swap hides in a gap, this test is corrupted.

Compare the per-fish numbers here against a SINGLE-stretch train_reid run:
  - fish3/5 improves when pooled  -> more diverse data helps -> the accumulator is justified.
  - no change                     -> not a data-volume problem (fish likely near-identical) -> skip it.

Usage: python -m scripts.pool_test --video_name IMG_1839
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import yaml
import logging
import argparse
import subprocess
from scripts.reid_features import build_features
from scripts.console import banner, banner_sub

logger = logging.getLogger(__name__)
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)['train_reid']   # reuse train_reid's config; POOL its `stretches` list

TRAIN_FRAC = 0.7

def main(video_name):
    v = cfg['videos'][video_name]
    crops_run, stretches, backbone_name = v['crops_run'], v['stretches'], v['backbone']
    num_epochs, lr, batch_size = v['num_epochs'], v['lr'], v['batch_size']

    banner(f"POOLING TEST — pooling stretches {stretches} into ONE training set (frozen head)")
    logger.info("ASSUMES verified cross-stretch ID consistency (fish_N = same physical fish across these stretches)")

    # ---------- build features per stretch, temporal-split EACH, then pool the train-parts and val-parts ----------
    tr_f, tr_l, va_f, va_l = [], [], [], []
    label_map = None
    for s in stretches:
        feats, labels, frames, lm = build_features(crops_run, [s], backbone_name)
        if label_map is None:
            label_map = lm
        assert lm == label_map, f"stretch {s} label_map {lm} != {label_map} — a fish is missing in a stretch; can't pool cleanly"
        order = torch.argsort(frames)                      # early -> late, WITHIN this stretch
        n_tr  = int(len(order) * TRAIN_FRAC)
        tr_f.append(feats[order[:n_tr]]); tr_l.append(labels[order[:n_tr]])   # early crops -> train
        va_f.append(feats[order[n_tr:]]); va_l.append(labels[order[n_tr:]])   # late crops  -> val
    train_feats, train_labels = torch.cat(tr_f), torch.cat(tr_l)
    val_feats,   val_labels   = torch.cat(va_f), torch.cat(va_l)
    n_classes, feat_dim = len(label_map), train_feats.shape[1]
    logger.info(f"POOLED: train {len(train_labels)} crops | val {len(val_labels)} crops | {len(stretches)} stretches | {n_classes} fish")

    # ---------- train the head on the POOLED features (frozen backbone) ----------
    head = nn.Linear(feat_dim, n_classes); head.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=batch_size, shuffle=True)
    banner_sub("TRAINING THE HEAD (pooled)")
    for epoch in range(num_epochs):
        running = 0.0
        for feats, labels in loader:
            logits = head(feats); loss = loss_fn(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item()
        if (epoch + 1) % 20 == 0:
            logger.info(f"epoch {epoch+1}/{num_epochs}  avg loss {running/len(loader):.4f}")

    # ---------- eval (per-fish + confusion — the fish3/5 signal) ----------
    banner_sub("VALIDATION (pooled) — per-fish + confusion")
    head.eval()
    with torch.no_grad():
        val_preds = head(val_feats).argmax(dim=1)
    correct = (val_preds == val_labels)
    acc = correct.float().mean().item()
    logger.info(f"OVERALL pooled val accuracy: {correct.sum().item()}/{len(val_labels)} = {acc:.3f}")

    slot_to_fish = {slot: fish for fish, slot in label_map.items()}
    metrics = {"val_accuracy": acc}
    for slot in range(n_classes):
        mask = val_labels == slot
        if mask.sum() > 0:
            fish_acc = correct[mask].float().mean().item()
            metrics[f"acc_fish_{slot_to_fish[slot]}"] = fish_acc
            logger.info(f"  fish {slot_to_fish[slot]}: {correct[mask].sum().item()}/{mask.sum().item()} = {fish_acc:.3f}")
    for slot in range(n_classes):
        mask = val_labels == slot
        if mask.sum() > 0:
            g = val_preds[mask]
            counts = {slot_to_fish[x]: int((g == x).sum()) for x in g.unique().tolist()}
            logger.info(f"  true fish {slot_to_fish[slot]} was called: {counts}")

    # ---------- MLflow (same experiment; mode=pooled_frozen so it compares to single-stretch runs) ----------
    import mlflow
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlflow.set_tracking_uri(os.path.join(repo_root, "mlruns"))
    mlflow.set_experiment("aquamind_reid")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    dirty  = bool(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip())
    with mlflow.start_run(run_name=f"pooled__{'_'.join(stretches)}"):
        mlflow.log_params({
            "mode": "pooled_frozen", "crops_run": os.path.basename(crops_run),
            "stretches": str(stretches), "n_stretches": len(stretches), "backbone": backbone_name,
            "num_epochs": num_epochs, "lr": lr, "batch_size": batch_size,
            "feat_dim": feat_dim, "n_classes": n_classes,
            "n_train": len(train_labels), "n_val": len(val_labels),
        })
        mlflow.log_metrics(metrics)
        mlflow.set_tag("git_commit", commit)
        mlflow.set_tag("git_dirty", dirty)
    logger.info(f"logged to MLflow (aquamind_reid, mode=pooled_frozen, commit={commit[:8]}, dirty={dirty})")

if __name__ == "__main__":
    from scripts.logger import setup_logging
    setup_logging()
    parser = argparse.ArgumentParser(description="Pooling test: pool verified stretches, train a frozen head")
    parser.add_argument("--video_name", default=cfg['video'], help="key under train_reid.videos in config.yaml")
    args = parser.parse_args()
    main(args.video_name)
