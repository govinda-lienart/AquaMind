"""
Resolve a YOLO model reference to a native ultralytics YOLO object.

Accepts either:
  - a plain file path         (e.g. runs/.../weights/best.pt)          → loaded directly
  - an MLflow registry URI    (e.g. models:/aquamind-yolo-detector@champion)

For a registry URI we DOWNLOAD the underlying weights and load them with native YOLO(),
rather than mlflow.pyfunc.load_model() — the tracker/ml_backend need the native ultralytics
Results API (.boxes.xyxy etc.), which the generic pyfunc wrapper does not expose.
"""

import logging

from ultralytics import YOLO

logger = logging.getLogger(__name__)

TRACKING_URI = 'sqlite:///mlflow.db'     # same store log_artifact_mlflow.py registers into


def resolve_ref(ref):
    """ref -> concrete 'models:/<name>/<version>' if ref is an alias URI (e.g. '...@champion'); else ref unchanged.

    Aliases are mutable (a later promotion can repoint '@champion' at a new version), so a run's
    sidecar must record the concrete version it actually used, not the alias string.
    """
    if ref.startswith('models:/') and '@' in ref:
        from mlflow.tracking import MlflowClient
        name, alias = ref[len('models:/'):].split('@', 1)
        version = MlflowClient(tracking_uri=TRACKING_URI).get_model_version_by_alias(name, alias).version
        return f'models:/{name}/{version}'
    return ref


def load_yolo(ref):
    """ref → native YOLO. A models:/ or runs:/ URI resolves through the MLflow registry; anything else is a path."""
    if ref.startswith('models:/') or ref.startswith('runs:/'):
        import mlflow
        mlflow.set_tracking_uri(TRACKING_URI)
        # log_model(artifacts={'weights': best_pt}) stores it flat as <model>/artifacts/<basename> —
        # the dict KEY ('weights') is not part of the path, only the file's own basename is. Verified
        # empirically (MLflow 3.x logged-model layout): models:/<name>/<version>/artifacts/best.pt
        weights = mlflow.artifacts.download_artifacts(artifact_uri=f"{ref}/artifacts/best.pt")
        logger.info(f"resolved {ref} → {weights}")
        return YOLO(weights)
    return YOLO(ref)
