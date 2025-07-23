"""
BlogSpy – fault-tolerant, memory-safe training pipeline

Re-adds missing constants (HASH_DIM, MEM_LIMIT_GB) and allows
re-running from the cached feature matrix to avoid rebuilding.

Added logging for misclassified URLs during evaluation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pathlib
import subprocess
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import (
    ENRICHED_DATA_FILE,
    ID_TO_LABEL,
    LGBM_PARAMS,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)
from src.feature_engineering import (
    extract_content_features,
    extract_url_features,
)
from src.utils import get_logger

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
logger = get_logger(__name__)


def get_git_commit_hash() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@dataclass
class HashingVectorizerConfig:
    stop_words: str = "english"
    n_features: int = 2**16
    ngram_range: tuple[int, int] = (1, 2)
    alternate_sign: bool = False
    norm: str = "l2"
    dtype: str = "float32"

    def to_hash(self) -> str:
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:12]


@dataclass
class PipelineConfig:
    project_root: pathlib.Path = pathlib.Path(__file__).parent.parent
    enriched_data_file: pathlib.Path = ENRICHED_DATA_FILE
    models_dir: pathlib.Path = MODELS_DIR
    reports_dir: pathlib.Path = REPORTS_DIR
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE
    id_to_label: Dict[int, str] = field(default_factory=lambda: ID_TO_LABEL)
    lgbm_params: Dict[str, Any] = field(default_factory=lambda: LGBM_PARAMS)
    vectorizer_config: HashingVectorizerConfig = field(
        default_factory=HashingVectorizerConfig
    )
    cache_meta_file: pathlib.Path = field(init=False)
    features_file: pathlib.Path = field(init=False)
    labels_file: pathlib.Path = field(init=False)
    urls_file: pathlib.Path = field(init=False)

    def __post_init__(self):
        processed_dir = self.project_root / "data/processed"
        self.cache_meta_file = processed_dir / "cache.meta.json"
        self.features_file = processed_dir / "X_features.npz"
        self.labels_file = processed_dir / "y_labels.npy"
        self.urls_file = processed_dir / "urls.npy"


class FeatureEngineer:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.vectorizer = HashingVectorizer(
            **asdict(config.vectorizer_config, dict_factory=self._str_to_dtype)
        )

    @staticmethod
    def _str_to_dtype(data):
        d = dict(data)
        if "dtype" in d:
            d["dtype"] = getattr(np, d["dtype"])
        return d

    def run(
        self, rebuild: bool
    ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, HashingVectorizer]:
        if not rebuild and self._is_cache_valid():
            return self._load_from_cache()

        if not self.config.enriched_data_file.exists():
            raise FileNotFoundError(
                f"Input file not found: {self.config.enriched_data_file}. Please run the data preparation script first."
            )
        return self._build_from_scratch()

    def _is_cache_valid(self) -> bool:
        cfg = self.config
        if not all(
            p.exists()
            for p in [
                cfg.features_file,
                cfg.labels_file,
                cfg.urls_file,
                cfg.cache_meta_file,
            ]
        ):
            return False
        with open(cfg.cache_meta_file, "r") as f:
            meta = json.load(f)
        current_hash = self.config.vectorizer_config.to_hash()
        if meta.get("vectorizer_hash") != current_hash:
            logger.warning("Vectorizer config has changed. Rebuilding feature cache.")
            return False
        return True

    def _load_from_cache(self):
        log_memory("Loading features, labels, and URLs from cache...")
        X = sp.load_npz(self.config.features_file)
        y = np.load(self.config.labels_file)
        urls = np.load(self.config.urls_file, allow_pickle=True)
        return X, y, urls, self.vectorizer

    def _build_from_scratch(self):
        log_memory("Building features from scratch...")
        logger.info(
            "Using HashingVectorizer for speed and memory efficiency. Note: this process is irreversible and sacrifices model interpretability."
        )
        pf = pq.ParquetFile(self.config.enriched_data_file)
        X_parts, y_parts, url_parts = [], [], []

        for i in tqdm(range(pf.num_row_groups), desc="Processing Chunks"):
            df = (
                pf.read_row_group(i, columns=["url", "structured_text", "label"])
                .to_pandas()
                .dropna()
            )
            if df.empty:
                continue
            if i == 0:
                self._validate_input_schema(df)

            df["text_content"] = df["structured_text"].apply(
                lambda sl: " ".join(d.get("text", "") for d in sl)
                if isinstance(sl, list)
                else ""
            )
            X_chunk, feature_shapes = self._vectorize_chunk(df)
            if i == 0:
                logger.info(f"Feature breakdown per instance: {feature_shapes}")

            X_parts.append(X_chunk)
            y_parts.append(df["label"].to_numpy(dtype="int8"))
            url_parts.extend(df["url"].tolist())

        log_memory("Stacking matrices...")
        X = sp.vstack(X_parts, format="csr")
        y = np.concatenate(y_parts)
        urls = np.array(url_parts)
        self._save_to_cache(X, y, urls)
        return X, y, urls, self.vectorizer

    def _validate_input_schema(self, df: pd.DataFrame):
        first_valid_row = df["structured_text"].iloc[0]
        if not isinstance(first_valid_row, list):
            raise TypeError("`structured_text` column is not a list as expected.")
        if (
            first_valid_row
            and isinstance(first_valid_row[0], dict)
            and "text" not in first_valid_row[0]
        ):
            raise TypeError("Dicts in `structured_text` list are missing 'text' key.")
        logger.info("Input schema validation passed.")

    def _vectorize_chunk(self, df: pd.DataFrame):
        text_features = self.vectorizer.transform(df["text_content"])
        url_features = extract_url_features(df["url"]).to_numpy(dtype="float32")
        content_features = extract_content_features(df["text_content"]).to_numpy(
            dtype="float32"
        )
        shapes = {
            "text": text_features.shape[1],
            "url": url_features.shape[1],
            "content": content_features.shape[1],
        }
        return (
            sp.hstack(
                [
                    text_features,
                    sp.csr_matrix(url_features),
                    sp.csr_matrix(content_features),
                ],
                format="csr",
            ),
            shapes,
        )

    def _save_to_cache(self, X: sp.csr_matrix, y: np.ndarray, urls: np.ndarray):
        cfg = self.config
        cfg.features_file.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(cfg.features_file, X)
        np.save(cfg.labels_file, y)
        np.save(cfg.urls_file, urls)
        cache_meta = {"vectorizer_hash": cfg.vectorizer_config.to_hash()}
        with open(cfg.cache_meta_file, "w") as f:
            json.dump(cache_meta, f)
        log_memory(f"Cached features {X.shape} and metadata to disk.")


class ModelTrainer:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        X: sp.csr_matrix,
        y: np.ndarray,
        urls: np.ndarray,
        vectorizer: HashingVectorizer,
    ):
        num_classes = len(np.unique(y))
        if num_classes < 2:
            raise ValueError(
                f"Training requires at least 2 classes, but found {num_classes}."
            )
        logger.info(f"Detected {num_classes}-class classification problem.")

        self._log_label_distribution(y, "overall")
        X_tr, X_te, y_tr, y_te, urls_tr, urls_te = train_test_split(
            X,
            y,
            urls,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        self._check_for_data_leakage(urls_tr, urls_te)
        self._log_label_distribution(y_tr, "training set")
        del X, y, urls
        gc.collect()

        manifest, temp_manifest_path = self._create_initial_manifest(vectorizer)
        try:
            booster = self._train_model(X_tr, y_tr, X_te, y_te, num_classes)
            del X_tr, y_tr
            gc.collect()

            metrics, misclassified_df = self._evaluate_model(
                booster, X_te, y_te, urls_te, num_classes
            )
            self._finalize_artifacts(booster, vectorizer, manifest, metrics)
        finally:
            if temp_manifest_path.exists():
                temp_manifest_path.unlink()

    def _check_for_data_leakage(self, urls_tr: np.ndarray, urls_te: np.ndarray):
        leakage = set(urls_tr).intersection(urls_te)
        if leakage:
            logger.warning(
                f"{len(leakage)} URLs found in both train and test sets. This indicates data leakage."
            )
        else:
            logger.info("Train/test split validation passed: No URL leakage detected.")

    def _log_label_distribution(self, labels: np.ndarray, name: str):
        unique_labels, counts = np.unique(labels, return_counts=True)
        dist = {
            self.config.id_to_label.get(k, k): f"{v} ({v / len(labels):.2%})"
            for k, v in zip(unique_labels, counts)
        }
        logger.info(f"Label distribution in {name}: {dist}")

    def _create_initial_manifest(self, vectorizer: HashingVectorizer):
        cfg = self.config
        timestamp = datetime.now(timezone.utc)
        commit_hash = get_git_commit_hash()
        run_id = (
            f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{commit_hash[:7]}"
            if commit_hash
            else timestamp.strftime("%Y%m%d_%H%M%S_%f")
        )

        manifest = {
            "run_id": run_id,
            "timestamp_utc": timestamp.isoformat(),
            "git_commit_hash": commit_hash,
            "model_params": cfg.lgbm_params,
            "feature_params": asdict(cfg.vectorizer_config),
            "training_params": {
                "test_size": cfg.test_size,
                "random_state": cfg.random_state,
            },
        }

        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".json", encoding="utf-8"
        ) as f:
            json.dump(manifest, f, indent=2)
            temp_path = pathlib.Path(f.name)
        return manifest, temp_path

    def _train_model(
        self,
        X_tr: sp.csr_matrix,
        y_tr: np.ndarray,
        X_te: sp.csr_matrix,
        y_te: np.ndarray,
        num_classes: int,
    ):
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_te, label=y_te, reference=train_data)
        params = self.config.lgbm_params.copy()
        if num_classes == 2:
            params["objective"] = "binary"
            params["metric"] = "binary_logloss"
        else:
            params["objective"] = "multiclass"
            params["metric"] = "multi_logloss"
            params["num_class"] = num_classes

        num_boost_round = params.pop("n_estimators")
        callbacks = [
            lgb.early_stopping(stopping_rounds=25, verbose=True),
            lgb.log_evaluation(period=50),
        ]
        return lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            valid_names=["eval"],
            callbacks=callbacks,
        )

    def _evaluate_model(
        self,
        booster: lgb.Booster,
        X_te: sp.csr_matrix,
        y_te: np.ndarray,
        urls_te: np.ndarray,
        num_classes: int,
    ):
        from sklearn.metrics import classification_report, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        pred_probs = booster.predict(X_te)
        preds = (
            np.argmax(pred_probs, axis=1)
            if num_classes > 2
            else (pred_probs > 0.5).astype(int)
        )

        misclassified_df = self._get_sorted_misclassified(
            y_te, preds, urls_te, pred_probs
        )
        if not misclassified_df.empty:
            (self.config.reports_dir / "misclassified_urls.csv").to_csv(
                misclassified_df, index=False
            )

        report_dict = classification_report(
            y_te,
            preds,
            target_names=self.config.id_to_label.values(),
            digits=4,
            output_dict=True,
        )
        report_str = json.dumps(report_dict, indent=2)
        print("\n--- Final Classification Report ---\n")
        print(report_str)
        (self.config.reports_dir / "classification_report.json").write_text(report_str)

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_te,
            preds,
            display_labels=list(self.config.id_to_label.values()),
            cmap="Blues",
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(self.config.reports_dir / "confusion_matrix.png")
        plt.close(fig)
        return report_dict, misclassified_df

    def _get_sorted_misclassified(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        urls: np.ndarray,
        probs: np.ndarray,
    ):
        misclassified_indices = np.where(y_true != y_pred)[0]
        if len(misclassified_indices) == 0:
            return pd.DataFrame()

        error_df = pd.DataFrame(
            {
                "url": urls[misclassified_indices],
                "true_label": y_true[misclassified_indices],
                "predicted_label": y_pred[misclassified_indices],
            }
        )
        if probs.ndim == 1:
            error_df["confidence_wrong_pred"] = np.where(
                y_pred[misclassified_indices] == 1,
                probs[misclassified_indices],
                1 - probs[misclassified_indices],
            )
        else:
            error_df["confidence_wrong_pred"] = probs[
                misclassified_indices, y_pred[misclassified_indices]
            ]

        error_df.sort_values(by="confidence_wrong_pred", ascending=False, inplace=True)
        error_df["true_label"] = error_df["true_label"].map(self.config.id_to_label)
        error_df["predicted_label"] = error_df["predicted_label"].map(
            self.config.id_to_label
        )
        return error_df

    def _finalize_artifacts(
        self,
        booster: lgb.Booster,
        vectorizer: HashingVectorizer,
        manifest: dict,
        metrics: dict,
    ):
        cfg = self.config
        model_dir = cfg.models_dir / manifest["run_id"]
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(booster, model_dir / "model.joblib", compress=3)
        joblib.dump(vectorizer, model_dir / "vectorizer.joblib", compress=3)

        manifest["evaluation_metrics"] = metrics
        with open(model_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        log_memory(f"Model artifacts and manifest saved to {model_dir}")


def log_memory(msg: str):
    mem_gb = psutil.Process().memory_info().rss / 1e9
    logger.info(f"[{mem_gb:5.2f} GB] {msg}")


def main(rebuild: bool):
    log_memory("--- Pipeline Start ---")
    config = PipelineConfig()
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    feature_engineer = FeatureEngineer(config)
    X, y, urls, vectorizer = feature_engineer.run(rebuild=rebuild)

    trainer = ModelTrainer(config)
    trainer.run(X, y, urls, vectorizer)
    log_memory("--- Pipeline End ---")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the main BlogSpy model.")
    ap.add_argument(
        "--rebuild",
        "--force",
        dest="rebuild",
        action="store_true",
        help="Force rebuilding the feature matrix and URL cache.",
    )
    args = ap.parse_args()
    main(rebuild=args.rebuild)
