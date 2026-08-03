"""
accumulate_finetune_kaggle.py — SELF-CONTAINED Kaggle version, split into CELLS.

Kaggle keeps variables in memory between cells, so paste each block below into its OWN cell
(marked with '# %% CELL N'). Then if a later cell fails you re-run only THAT cell — the slow
training (CELL 5) isn't repeated. Data upload is separate; re-running cells never re-uploads.

Needs a Kaggle Dataset with: <dataset>/crops/fish_<id>/frame_<n>_fish_<id>.jpg + fragments.csv + tracks.parquet
Settings: GPU T4 on, Internet on. Output -> /kaggle/working/ (CSV + plot + log).
"""

# %% CELL 1 — imports, locate data, config  (fast; re-run freely)
import os, glob
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

_hits = glob.glob("/kaggle/input/**/fragments.csv", recursive=True)
assert _hits, "fragments.csv not found under /kaggle/input — is the dataset attached?"
DATA_DIR = os.path.dirname(_hits[0])
OUT = "/kaggle/working/fragments_stitched.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| data:", DATA_DIR)

MIN_SEPARATION_PX = 150
UNFREEZE_BLOCKS, EPOCHS_ROUND, BB_LR, HEAD_LR, BATCH = 2, 8, 1e-5, 1e-3, 64
CONF_THRESH, MAX_ROUNDS = 0.90, 8
BACKBONE = "dinov2_vits14"

transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
_aug = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(8), T.ColorJitter(brightness=0.2)])  # NO hue (same-morph)
def train_tf(img): return transform(_aug(img))


# %% CELL 2 — helper definitions  (fast; just defines functions)
def flag_clean(tracks, min_dist):
    tracks = tracks.copy(); sep = np.zeros(len(tracks), bool)
    for _, g in tracks.groupby("frame_number"):
        pos = g[["x", "y"]].to_numpy()
        if len(pos) == 1: s = np.array([True])
        else:
            d = np.hypot(*(pos[:, None, :] - pos[None, :, :]).transpose(2, 0, 1)); np.fill_diagonal(d, np.inf)
            s = d.min(1) >= min_dist
        sep[tracks.index.get_indexer(g.index)] = s
    tracks["clean"] = sep & (tracks["occluded"] == 0)
    return tracks

def coexist_adj(fr):
    ids = list(fr); adj = {f: set() for f in ids}
    for a in range(len(ids)):
        s1, e1 = fr[ids[a]]
        for b in range(a + 1, len(ids)):
            s2, e2 = fr[ids[b]]
            if s1 <= e2 and s2 <= e1: adj[ids[a]].add(ids[b]); adj[ids[b]].add(ids[a])
    return adj

def find_seed(fr, sizes, n):
    best, bs, bn, fb = None, -1, 0, None
    for t in sorted({s for s, _ in fr.values()} | {e for _, e in fr.values()}):
        act = [f for f, (s, e) in fr.items() if s <= t <= e]; sc = sum(sizes[f] for f in act)
        if len(act) == n and sc > bs: best, bs = act, sc
        if len(act) > bn: bn, fb = len(act), act
    return best if best else fb

def crop_paths(frags):
    tr = flag_clean(pd.read_parquet(f"{DATA_DIR}/tracks.parquet"), MIN_SEPARATION_PX)
    cl = tr[tr["clean"]]; paths, cf = [], []
    for i, r in frags.iterrows():
        fid = int(r["fish_id"])
        sub = cl[(cl.fish_id == fid) & (cl.frame_number >= r.frame_start) & (cl.frame_number <= r.frame_end)]
        for fn in sub.frame_number:
            p = f"{DATA_DIR}/crops/fish_{fid}/frame_{int(fn)}_fish_{fid}.jpg"
            if os.path.exists(p): paths.append(p); cf.append(i)
    return paths, np.array(cf)

class CropDS(Dataset):
    def __init__(s, paths, labels, tf): s.paths, s.labels, s.tf = paths, labels, tf
    def __len__(s): return len(s.paths)
    def __getitem__(s, i): return s.tf(Image.open(s.paths[i]).convert("RGB")), s.labels[i]

def build_model(n_id):
    bb = torch.hub.load("facebookresearch/dinov2", BACKBONE).to(device); bb.eval()
    for p in bb.parameters(): p.requires_grad = False
    for blk in bb.blocks[-UNFREEZE_BLOCKS:]:
        for p in blk.parameters(): p.requires_grad = True
    if hasattr(bb, "norm"):
        for p in bb.norm.parameters(): p.requires_grad = True
    with torch.no_grad(): fd = bb(torch.zeros(1, 3, 224, 224).to(device)).shape[1]
    return bb, nn.Linear(fd, n_id).to(device)

def fine_tune(bb, head, paths, cf, labeled):
    idx, y = [], []
    for f, c in labeled.items():
        ci = np.where(cf == f)[0]; idx += ci.tolist(); y += [c] * len(ci)
    dl = DataLoader(CropDS([paths[i] for i in idx], y, train_tf), batch_size=BATCH, shuffle=True, num_workers=2)
    opt = torch.optim.Adam([{"params": [p for p in bb.parameters() if p.requires_grad], "lr": BB_LR},
                            {"params": head.parameters(), "lr": HEAD_LR}])
    lf = nn.CrossEntropyLoss(); bb.train(); head.train()
    for _ in range(EPOCHS_ROUND):
        for im, lb in dl:
            im, lb = im.to(device), lb.to(device)
            opt.zero_grad(); lf(head(bb(im)), lb).backward(); opt.step()

def predict(bb, head, paths, cf, n_frags):
    bb.eval(); head.eval(); pr = {}
    with torch.no_grad():
        for f in range(n_frags):
            ci = np.where(cf == f)[0]; ps = []
            for j in range(0, len(ci), BATCH):
                im = torch.stack([transform(Image.open(paths[k]).convert("RGB")) for k in ci[j:j + BATCH]]).to(device)
                ps.append(torch.softmax(head(bb(im)), 1).cpu())
            pr[f] = torch.cat(ps).mean(0).numpy()
    return pr

def embed_frags(bb, paths, cf, n_frags):
    bb.eval(); out = []
    with torch.no_grad():
        for f in range(n_frags):
            ci = np.where(cf == f)[0]; fs = []
            for j in range(0, len(ci), BATCH):
                im = torch.stack([transform(Image.open(paths[k]).convert("RGB")) for k in ci[j:j + BATCH]]).to(device)
                fs.append(bb(im).cpu())
            v = torch.cat(fs).mean(0); out.append((v / (v.norm() + 1e-9)).numpy())
    return np.stack(out)

LOG = []
def log(msg):
    print(msg); LOG.append(str(msg))
print("helpers defined")


# %% CELL 3 — load fragments + build crop list  (fast; fails FAST if paths are wrong, before any training)
frags = pd.read_csv(f"{DATA_DIR}/fragments.csv")
n_id = int(frags.fish_id.nunique())
fr = {i: (int(r.frame_start), int(r.frame_end)) for i, r in frags.iterrows()}
sizes = {i: int(r.n_frames) for i, r in frags.iterrows()}
adj = coexist_adj(fr)
paths, cf = crop_paths(frags)
print(f"{len(paths)} clean crops, {len(frags)} fragments, {n_id} fish")
assert len(paths) > 0, "no crops found — check the crops/ folder path"


# %% CELL 4 — build model + seed  (downloads DINOv2 once; ~1 min)
seed = find_seed(fr, sizes, n_id); labeled = {f: k for k, f in enumerate(seed)}
log(f"seed: {seed}")
bb, head = build_model(n_id)
print("model ready — last", UNFREEZE_BLOCKS, "blocks unfrozen")


# %% CELL 5 — ACCUMULATE (the SLOW part — minutes/round on GPU; watch the round logs)
history = [(0, len(labeled))]
for rnd in range(1, MAX_ROUNDS + 1):
    fine_tune(bb, head, paths, cf, labeled)
    pr = predict(bb, head, paths, cf, len(frags)); added = 0
    for f in sorted((f for f in range(len(frags)) if f not in labeled), key=lambda f: -pr[f].max()):
        c, conf = int(pr[f].argmax()), float(pr[f].max())
        if conf < CONF_THRESH: break
        if any(labeled.get(g) == c for g in adj[f]): continue
        labeled[f] = c; added += 1
    history.append((rnd, len(labeled)))
    log(f"round {rnd}: +{added} ({len(labeled)}/{len(frags)} labelled)")
    if added == 0: break


# %% CELL 6 — final assignment + save CSV  (fast)
fine_tune(bb, head, paths, cf, labeled); pr = predict(bb, head, paths, cf, len(frags))
for f in range(len(frags)):
    if f in labeled: continue
    forb = {labeled.get(g) for g in adj[f]}
    labeled[f] = next((int(c) for c in np.argsort(-pr[f]) if c not in forb), int(pr[f].argmax()))
coll = sum(1 for f in adj for g in adj[f] if f < g and labeled[f] == labeled[g])
n_conf = history[-1][1]
log(f"DONE — identities {len(set(labeled.values()))}/{n_id}, collisions {coll}, "
    f"confidently accumulated {n_conf}/{len(frags)} (frozen version stalled at 30)")
frags["cluster"] = [labeled[i] for i in range(len(frags))]
frags.to_csv(OUT, index=False); log(f"saved -> {OUT}")


# %% CELL 7 — plots + log  (fast; re-run alone if it errors — training is already done)
with open("/kaggle/working/accumulate_log.txt", "w") as fh:
    fh.write("\n".join(LOG))
emb = embed_frags(bb, paths, cf, len(frags))
ident = np.array([labeled[i] for i in range(len(frags))])
xy = TSNE(n_components=2, init="pca", perplexity=min(30, len(emb) - 1), random_state=0).fit_transform(emb)
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
r, n = zip(*history)
ax[0].plot(r, n, "o-"); ax[0].axhline(30, ls="--", c="grey", label="frozen stalled (30)")
ax[0].set_xlabel("accumulation round"); ax[0].set_ylabel("fragments confidently labelled")
ax[0].set_title(f"accumulation curve (final {n_conf}/{len(frags)})"); ax[0].legend(); ax[0].grid(alpha=.3)
for k in range(n_id):
    m = ident == k
    ax[1].scatter(xy[m, 0], xy[m, 1], s=60, alpha=.7, label=f"identity {k+1}")
ax[1].set_title("fine-tuned fragments by identity (clean islands = it worked)")
ax[1].legend(fontsize=8); ax[1].set_xticks([]); ax[1].set_yticks([])
fig.tight_layout(); fig.savefig("/kaggle/working/accumulate_result.png", dpi=130)
print("saved -> /kaggle/working/accumulate_result.png  (+ accumulate_log.txt)")
