# AquaMind — Development Diary

Full reasoning trail behind the compact current-state summaries in `CLAUDE.md`. Nothing here is
deleted — this is where the *why* lives when `CLAUDE.md` says "see diary.md."

Not auto-loaded by Claude Code — read on demand. Trimmed for concision on 2026-08-05 (cut repeated
"trust this over the block above" scaffolding and superseded checklists) — every fact, number, and
decision below is preserved; only the meta-commentary and restatement were cut.

## Contents
- [Stage 5 — Tracker evaluation](#stage-5--tracker-evaluation)
- [Stage 6 — Fish re-identification](#stage-6--fish-re-identification)
- [Stage 7 — Behaviour classifier](#stage-7--behaviour-classifier)
- [MLflow Model Registry](#mlflow-model-registry)

---

## Stage 5 — Tracker evaluation

**Status: DONE, verified 2026-07-13.** `scripts/evaluate_tracker.py` — **3.9% overall ID-switch rate (crossing 16.9%, occlusion_recovery 0.5%)**, logged to the `aquamind_tracker` MLflow experiment with config sidecar + git_commit.

**Why ground-truth instead of a heuristic:** the tracker's own log already flags every moment a switch *could* occur (crossings, occlusion recoveries) — a heuristic like distance-jump detection can't be trusted without a human check, and here verification only needs to cover the flagged moments, not the whole video.

**Method:** blind manual review of one ~2min video (`IMG_2349.MOV`, 4 fish, 0–130s, calibration ends at frame 600 @ 60fps), cross-referenced against the tracker's logged decision points rather than full blind labeling. A `show_frame_number` overlay was added so QuickTime could frame-step. Deviations logged to a table (`run_name | frame_start | frame_end | fish_ids | error_type | what_went_wrong`; `fish_ids` always sorted+comma-joined to match the log-parser's normalization). The tracker's log lines were regex-parsed into an events table and matched against ground truth via a **cross-join + containment filter** — not `merge_asof`, since ground truth is a *range* and the log is a *point*.

**`error_type` legend** (canonical — keep in sync with the parser's `event_type`, which only ever emits `crossing`/`occlusion_recovery`):

| error_type | Meaning | Cross-checked against |
|---|---|---|
| `crossing` | ID confusion during/after two fish crossing paths | Log-parser |
| `occlusion_recovery` | Wrong ID assigned when a fish reappears after being hidden | Log-parser |
| `false_occlusion` | Fish shown "lost" though clearly visible — a detection miss | `tracks.occluded` |
| `missing_track` | Fish has zero box at all — rare/anomalous | Human-only |
| `reflection_confusion` | Box briefly attaches to a reflection | Human-only |
| `other` | Anything not covered above | Human-only |

The last four categories are always-unmatched, human-only findings — reported separately, not folded into the ID-switch rate.

**Pipeline hardening** (carried forward after the tracker's later rename): auto-clear-before-insert on `tracks` (prevents duplicate-key crashes on re-run — `tracks` is working storage for the current run, not a permanent archive across configs); a config sidecar dumped next to each run's output (needed because live `config.yaml` gets overwritten before review of the current run finishes).

**STEP 0–5 checklist:** all sub-items done except STEP 5 (repeat the whole process for a 2nd tracker config) — skipped by choice; Govinda logs every run to MLflow and deletes duplicates manually instead of building a gating flag.

---

## Stage 6 — Fish re-identification

**Goal:** hold correct fish identity through crossings/occlusions where the tracker's geometry alone swaps IDs, using a learned appearance embedding.

### The concept, decided early
A metric-learning CNN ("the brain"): crop → fingerprint vector, same fish close / different fish far (triplet/contrastive loss). Two separate objects — the **booth** (the model, trained once, general, reused forever) vs **enrollment** (per-tank: run each fish through the booth, average its crops into one template, no gradient training). Transfer learning, not from-scratch (bottom layers = general vision, reusable; only the top is fish-identity-specific) — echoes the Stage 3 YOLO fine-tune. YOLO's own backbone was considered and rejected as a starting point: YOLO was trained to be identity-*blind* on purpose (all fish → one class), the opposite of what re-ID needs.

DeepSORT's architecture (Kalman + Hungarian + appearance tie-breaker) was arrived at independently by reasoning, but its library was rejected — its embedder is pedestrian-trained, useless on fish, and a fish embedder would need training anyway. Instead: add an `appearance_distance` term to the tracker's existing cost matrix (`position + direction + appearance`) using an in-house embedder.

**The unlock: tracker-tracklet auto-labelling.** The tracker already follows each fish reliably *between* crossings — every crop inside one confident tracklet is guaranteed the same individual, giving free identity labels (idtracker.ai's trick). Scope decision: only *within-session* identity is needed (every behavioural assay, including alarm-cue pre/post, only needs identity to hold within one video) — cross-week persistent identity was explicitly out of scope.

### Direction refinement (2026-07-24)
Real target = **same-morph, within-session** identity (his actual fish are near-identical); distinct-morph (GloFish vs golden) demoted to a pipeline warm-up. **Architecture = a fresh per-video classifier** (idtracker.ai's actual method — training itself IS the identity assignment, no separate enrollment step), not the general-brain-+-enrollment idea above — this sidesteps the whole "does it generalize to unseen fish" question, since it only ever separates ~N *known* fish.

Tracklet selection = idtracker.ai's "global tracklet": a window where all N fish are simultaneously visible AND well-separated — a **positive** guardrail, chosen over the negative "between logged crossings" because silent swaps aren't logged (a tracker that swaps silently can't flag what it didn't notice).

Feasibility check: "twins" is a *human* limit, not a CNN one — idtracker.ai hits >99% on visually-identical zebrafish within a session. Crop resolution was checked and is fine (median 120×71px). Apparent *size* is explicitly the wrong cue to rely on (perspective makes near-glass fish look bigger) — want pattern, which is depth-stable. Data quantity was never the constraint, *diversity* was: ~300–500 varied crops/fish is enough, but 500 *consecutive* frames is only a few effective views — subsample for spread, don't just collect more. Augmentation gotcha for same-morph: no hue/saturation/colour jitter — it erases the exact pigment signal that separates same-morph fish (the same augmentation that *helps* distinct-morph re-ID *hurts* here).

### Frozen classifier results
`train_reid.py` — DINOv2 backbone frozen, cached fingerprints → trainable linear head, temporal train/val split (no near-duplicate leakage), per-stretch (never pooled across stretches — cross-stretch fish_id labels are unverified). Accuracy **plateaued 58% (stretch02) → 72% (stretch04)** as crop diversity increased (data-starved, not purely feature-capped — a bigger backbone, ViT-S→ViT-B, made no difference: 61%→61%, proving the bottleneck was generic-feature *quality*, not model capacity). **stretch02 fish3/fish5 = a genuine same-morph look-alike wall:** ~10% accuracy, mutually confused, reproducible across 3 runs, unmoved by more data (fish4, visually distinct, hit 100%).

### Fine-tune attempt (6.4) — a valuable negative result
`finetune_reid.py` unfroze the last N DINOv2 blocks, trained end-to-end with small backbone-LR / larger head-LR, brightness-only augmentation. Result: **58%→56%**, fish3 still 7% (still confused with fish5). Train loss crashed to ~0.001 while val stayed ~56% — textbook overfitting: 626 crops from ~15s are mostly near-duplicate frames, not enough varied evidence for *any* model to learn a generalizable split. **Diagnosis: the bottleneck is DATA DIVERSITY, not model capacity** — frozen (feature quality) and fine-tuned (overfitting on tiny data) fail on the same wall for different reasons. Kept as a reusable tool/banked skill, not revisited by tweaking epochs/LR.

### Appearance fused into the live tracker — three issues found
Built DeepSORT-style fusion (`appearance: true`; per-track EMA appearance memory, separation-gated; cost = position + direction + weighted cosine). Testing (~69% head) surfaced:
1. **Detector-merge crossings unfixed.** At a merge, YOLO outputs ONE box for two fish → one track ghosts. Appearance can't disambiguate a single contaminated box, and greedy healthy-first matching locks the wrong assignment *before* any appearance comparison happens.
2. **Appearance is a cost tie-breaker, not a veto.** Even an *outstandingly* distinctive (sick) fish still swapped — once a track is geometrically sitting on the wrong fish (dist ~0) with the real match far away (~200px), the appearance penalty can't overcome the geometry gap. A better embedder alone would not have fixed this.
3. **Lost fast fish drift unmatched** — the motion model only predicted one frame ahead of the *last* position, never coasting a ghost forward through its missing frames, so a fast fish's search radius stayed anchored at the wrong place.

### Geometry-fix attempts, and the one that stuck
First attempt: coast the ghost forward (capped, guarded) + freeze velocity through a merge + tighten the re-acquire gate. **Caused systemic ghosting** — a straight-line coast diverges from a fish that turns/decelerates while lost, and the tightened gate (no widening) couldn't re-acquire it, so lost fish became permanent ghosts. Reverted to the original 1-frame prediction + widening gate (a rare teleport bug is a better trade than systemic ghosting).

**The fix that worked (2026-08-03, `merge_fix: true`):** detect a merge (fewer detections than confirmed tracks, 2+ tracks predicting onto one box) → **neither** contested track snaps onto the blob — both ghost and **coast on their clean pre-merge velocity** (frozen, not recomputed from the contaminated centroid) — combined with the **already-working widening gate** (not a new tight one). This is what finally worked: both fish show correctly as lost for the ~6 merge frames, then re-sort correctly at the split. Solves the actual failure mode (fast, head-on, straight-line crossings). Remaining gap: a fish that turns while fully merged/invisible — rare, documented as an accepted limitation.

### Tracker chapter closed, briefly reopened, settled
Reverted uncommitted geometry experiments to the stable committed state (constant-velocity + OC-SORT + appearance tie-breaker, gate off). Rationale: the remaining swap is a ceiling shared by **all** SORT-family trackers including DeepSORT (same architecture, same failure — explicitly considered and rejected, its embedder wouldn't help either); downstream stages already consume separation-filtered windows so the swap doesn't propagate; re-ID is the higher-value place left to spend effort.

Reopened once for an evidence-backed upgrade: measured the contrastive embedding's **reliability horizon** — kNN identity accuracy vs. time gap — **~95% at ≤1s, 90% at 2s, 66% at 18s**. Wired the contrastive head into the tracker as a cost tie-breaker + time-gate (falls back to geometry once memory exceeds ~2.5s) — the hold-time curve turned directly into a rule. Tested live: the tie-breaker alone didn't fix merge swaps (same root — nothing to compare during a blob); a veto/gate version **caused a ghost-spiral again** — a "safe" 1.8% static false-refuse rate doesn't predict live cascading failures (one wrong refuse → ghost → drift past geometry's reach → 10s ghost). **Settled: flag, don't fix.** Appearance's role is a swap-**auditor** — flag suspect frames near a crossing for a 5-second human check — not a live preventer. The auditor is designed but unbuilt.

### The offline stitcher — a parallel track that reversed the whole approach
idtracker.ai's real architecture: track live, then **offline** re-identify by appearance — exclude crossings, learn identity from clean tracklets between them, backfill the crossing from confident IDs before/after. Built in pieces:

- **Tracklet builder** (`stitch_ids.py`): cuts clean same-fish runs using the separation filter — 62% of frames clean, 157 tracklets, 4–5k clean frames/fish.
- **Fingerprint per tracklet**: raw frozen DINOv2, averaged per tracklet.
- **Small-scale proof** (`stitch_proof.py`, trusted by construction — split each fish's own tracklets into early/late halves): rejoin worked (cos 0.97–0.98, 8/10 correct nearest-neighbour) — but exposed fish3/fish5 as a genuine look-alike pair and fish1 as drifting within a stretch.
- **Full-scale clustering of raw fingerprints into 5: FAILED.** Silhouette 0.213 (weak), one catch-all cluster mixing all 5 fish. Local rejoin (small time gap) doesn't imply global identity — a fish's look drifts over the full 2-min video even though raw features are locally reliable.
- **Contrastive head** (`contrastive_reid.py`, self-supervised SupCon — same-tracklet=positive, coexisting-tracklet=negative, trained on all 157 tracklets): whole-video silhouette 0.213→0.512.
- **Decisive trusted-label test** (`eval_contrastive.py`, stretch04, real IDs known): raw DINOv2 was already ~98% accurate *locally* (nearest crop = same fish) but silhouette only 0.034 — each fish shattered into scattered specks, which is why whole-video clustering failed on confetti. Contrastive consolidated each fish into one island (silhouette → 0.485) and separated the fish3/5 pair — **the win was consolidation, not new local discrimination.** Caveat: stretch04 was itself in the contrastive training data, so this isn't a fully held-out proof.
- **Full pipeline end-to-end** (constrained assignment + relabel/render + timeline viz): the **pivotal finding** — the tracker's own IDs have **0% frame collisions by construction**; the appearance-first stitcher, once it reassigns identity across the whole video, **introduces 17% frame collisions** — worse than doing nothing. Root cause: contrastive features are excellent *within* a stretch but drift across the full 2-minute video, so whole-video clustering scrambles otherwise-clean tracks.

**Conclusion (arrived at independently from both the live-tracker veto experiment AND the offline stitcher): don't rebuild identity from appearance.** Start from the tracker's already-clean IDs; use appearance only to detect/flag the rare real silent swaps. This is the settled tracker-first + swap-auditor direction.

### Script ledger (2026-08-04)
**Keep — live pipeline:** `tracker.py`, `reid_features.py`, `console.py`, `stitch_ids.py` (tracklet builder — free labels), `contrastive_reid.py` (trains the head). **Keep — eval/figures:** `reid_quality.py` (97% kNN overall, per-fish 96–100%, the reliability-horizon plot), `eval_contrastive.py` (raw-vs-contrastive kNN + before/after t-SNE). **Archived (findings preserved above):** `stitch_proof.py`, `stitch_assign.py`, `stitch_accumulate.py` + Kaggle fine-tune variant (idtracker-style accumulation — stalled at 30/157 confident, worse than baseline; heavy compute, likely just memorizes on single-session data), `stitch_render.py`, `stitch_timeline.py`, `train_reid.py`, `finetune_reid.py`, `pool_test.py` (proved naive cross-stretch pooling corrupts), `embed_crops.py` (DINOv2 zero-training baseline, superseded).

### Bottom-line findings (the portfolio story)
1. Frozen DINOv2 clusters fish locally (0.98 kNN) but not globally (drifts over 2 minutes).
2. A self-supervised contrastive head consolidates identity (0.034→0.485 silhouette) and separates look-alikes short-range.
3. Reliability horizon ≈2s, measured directly.
4. Whole-video automated re-ID is blocked by single-session **data diversity**, confirmed four independent ways (frozen-cluster failure, contrastive drift, live gate-spiral, accumulation stall) — not by method or embedder quality.
5. Geometry (`merge_fix`) solved the tracker's actual crossing failures; appearance's best use is the post-crossing swap auditor.

**Work style (Govinda's request):** he codes it himself — guide/skeleton, not full code blocks; tiny micro-steps; explain as a beginner; ask a question to introduce each concept rather than answering it directly.

**Career note:** the most skill-dense stage for a Junior ML Engineer profile — transfer learning, embeddings/metric learning, foundation models, honest evaluation, paper-to-code.

### Second video confirms the stitcher conclusion (2026-08-10)
Re-ran `stitch_ids.py` on `IMG_2349` (tracker run `tracker_IMG_2349_basic_2026_08_10_1555`, 4 fish, 157 kept tracklets) purely to check whether the whole-video stitcher's failure was `IMG_1839`-specific. It isn't: the fish_id×cluster crosstab is far from block-diagonal (fish 1/2/3 all pile into the same two clusters; fish 4 smears across three), and the t-SNE (`tracklet_clusters.png`) shows each spatial cluster is a mix of all four tracker colors — the embedding is grouping by something other than identity (pose/lighting/session drift), same failure mode as `IMG_1839`. Two independent videos now agree: whole-video appearance stitching doesn't work here regardless of clip. Confirms rather than changes the settled tracker-first + swap-auditor direction — no further whole-video stitching experiments planned; next step remains building the actual swap-auditor (still unbuilt, see above).

---

## Stage 7 — Behaviour classifier

**Goal:** basic per-fish metrics (doubling as a real chemical-alarm-cue antipredator assay, his PhD domain) as skill-building + report layer first, then pairwise chasing features → labelled events → RF/XGBoost baselines → an LSTM (PyTorch enters only at Phase F). Chasing was the original lead ML showpiece; re-ID took that role 2026-07-22, chasing remains a strong secondary one. **"Geometric features ≠ no ML"** — chasing is Level 2 (distance/speed/path fed into `nn.LSTM`), real ML fed good inputs; Level 3 (CNN+LSTM on raw pixels) is an optional stretch, its real payoff saved for feeding strike instead.

**Calibration:** pixels→cm via the tank's own known dimensions (`pixels_per_cm = tank_width_px / tank_width_cm`) — sufficient without body-length normalization until a second, differently-sized tank exists. Side-view perspective (front-glass fish look bigger) is a named, accepted limitation.

**Alarm-cue metric shortlist** (pre/post design, `(pre−post)/pre`), from literature review — shelter dropped (fish don't use it, echoing a crayfish paper that abandoned shelter scoring at <9% use), foraging parked (needs the classifier):

| Metric | ML? | Notes |
|---|---|---|
| Activity (speed/distance/line crossings) | No | classic proxy |
| Freezing (hysteresis 0.40/0.30 cm/s) | No | two thresholds stop flicker |
| **Bottom-dwelling ⭐** | No | headline metric — side view makes depth directly measurable |
| Shoal cohesion (mean NND) | No | same code as the chasing substrate |
| Area avoidance | No | occupancy heatmap |
| Dashing | ~No | borderline rule-based |
| Latency to first movement | No | needs the stimulus frame logged |

**Why feeding strike gets the CNN, not chasing:** feeding strike is defined by *appearance* (mouth gape, S-bend, lunge) that `(x,y)` tracks literally cannot see — a CNN is *necessary* there, not illustrative. Chasing is relational/kinematic, so geometry is already the right tool; a CNN there would likely lose to the feature-LSTM on this small a dataset anyway. Cost accepted: feeding strike gets no geometric baseline to compare against, and needs its own (fiddly, rare-event) labelling round — deliberately saved for last.

**Progress (as of 2026-07-17):** Phase A0–A3 done in `scripts/analyse_behaviour.py` (calibration, reading tracks, per-fish speed, cumulative distance). A4 (bottom-dwelling depth) in progress — see `CLAUDE.md`'s `RESUME HERE` for the exact current line. Known limitation for the eventual report: single side-view camera means monocular depth ambiguity (background fish project lower than the front waterline) — unfixable without a 2nd camera, but consistent pre/post so it cancels in the alarm-cue comparison.

**Full PHASE A–G** (elaboration behind `CLAUDE.md`'s bare checklist): A = basic per-fish metrics (Pandas, no ML). B = pairwise metrics (distance, closing speed, heading, lagged path-following — the chase signature). C = human-labelled chase events (~15–30 + matched negatives). D = windowing (fixed windows, event-level train/test split, no leakage). E = Sklearn baselines (LR/RF/XGBoost, honest P/R/F1). F = the LSTM, evaluated on the same split as the baselines — the "did the LSTM actually beat the baseline, where does each fail" comparison is the story employers want. F+ = optional CNN+LSTM stretch on chasing, kept only as cheap insurance for a feature-vs-end-to-end comparison story. G = downstream consumers (overlay, counters, the automated scientific report) — built last, only once the model is trustworthy.

---

## MLflow Model Registry

**What changed:** `tracker.py`/`ml_backend.py` used to hardcode the YOLO weights path directly in `config.yaml`. `log_artifact_mlflow.py` now **registers** `best.pt` properly via `mlflow.pyfunc.log_model(...)` (a real `MLmodel` flavor, not a bare file pointer) under `aquamind-yolo-detector`, moving a `@champion` alias to the new version each time — promoting a model is now just "re-run the script."

**Textbook registration, not textbook loading:** loading uses `scripts/model_registry.py`'s `load_yolo()`, which downloads the weights and loads them with native `YOLO()` — not `mlflow.pyfunc.load_model()`, since the tracker depends on ultralytics' native `.boxes` API that generic pyfunc doesn't expose. The registry is for name resolution only.

**Bug caught by testing, not guessing:** assumed the artifact would nest under `<model>/artifacts/<dict_key>/<basename>`. Logged a throwaway model to a scratch DB and inspected the real output — MLflow 3.x actually flattens it to `artifacts/<basename>`, discarding the dict key. Also confirmed MLflow 3.x returns a `models:/m-<id>` **Logged Model** URI, a new entity type, not the old run-relative path.

**Backend split, unresolved:** the registry lives in SQLite (`sqlite:///mlflow.db`, required — `mlruns/` can't host a registry); other experiments still live in the older `mlruns/` file store. Point `mlflow ui --backend-store-uri` at whichever one you need.

**Verified end-to-end (2026-08-05):** registered v3 = the Jul 21 model (`5r_8c_9r_10r_11c_14c`, commit `bf02df6`), `@champion` moved to it, matches what the tracker was already running (v1 was an earlier raw-file registration, superseded).
