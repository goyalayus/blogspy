"""
BlogSpy – fault-tolerant, memory-safe training pipeline

Re-adds missing constants (HASH_DIM, MEM_LIMIT_GB) and allows
re-running from the cached feature matrix to avoid rebuilding.

Added logging for misclassified URLs during evaluation.
"""

from __future__ import annotations

import argparse
import gc
import pathlib
import resource
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
import scipy.sparse as sp
from bs4 import XMLParsedAsHTMLWarning
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
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

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
logger = get_logger(__name__)


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
    hash_dim: int = 2**16
    mem_limit_gb: int = 6
    features_file: pathlib.Path = field(init=False)
    labels_file: pathlib.Path = field(init=False)
    urls_file: pathlib.Path = field(init=False)
    model_artifact_name: str = "blogspy_model.joblib"

    def __post_init__(self):
        processed_dir = self.project_root / "data/processed"
        self.features_file = processed_dir / "X_features.npz"
        self.labels_file = processed_dir / "y_labels.npy"
        self.urls_file = processed_dir / "urls.npy"


class FeatureEngineer:

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.vectorizer = HashingVectorizer(
            stop_words="english",
            n_features=config.hash_dim,
            ngram_range=(1, 2),
            alternate_sign=False,
            norm="l2",
            dtype=np.float32,
        )

    def run(
        self, rebuild: bool
    ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, HashingVectorizer]:
        cfg = self.config
        if (not rebuild and cfg.features_file.exists()
                and cfg.labels_file.exists() and cfg.urls_file.exists()):
            return self._load_from_cache()
        return self._build_from_scratch()

    def _load_from_cache(self):
        log_memory("Loading features, labels, and URLs from cache...")
        X = sp.load_npz(self.config.features_file)
        y = np.load(self.config.labels_file)
        urls = np.load(self.config.urls_file, allow_pickle=True)
        log_memory(f"Loaded X{X.shape}, y{y.shape}, and {len(urls)} URLs.")
        return X, y, urls, self.vectorizer

    def _build_from_scratch(self):
        log_memory("Building features from scratch...")
        pf = pq.ParquetFile(self.config.enriched_data_file)
        wanted_cols = ["url", "structured_text", "label"]
        X_parts, y_parts, url_parts = [], [], []

        for i in tqdm(range(pf.num_row_groups), desc="Processing Chunks"):
            df = pf.read_row_group(i, columns=wanted_cols).to_pandas().dropna()
            if df.empty:
                continue

            df["text_content"] = df["structured_text"].apply(
                lambda sl: " ".join(d.get("text", "") for d in sl))

            X_chunk = self._vectorize_chunk(df)
            X_parts.append(X_chunk)
            y_parts.append(df["label"].to_numpy(dtype="int8"))
            url_parts.extend(df["url"].tolist())

        log_memory("Stacking matrices...")
        X = sp.vstack(X_parts, format="csr")
        y = np.concatenate(y_parts)
        urls = np.array(url_parts)
        self._save_to_cache(X, y, urls)
        return X, y, urls, self.vectorizer

    def _vectorize_chunk(self, df: pd.DataFrame) -> sp.csr_matrix:
        text_features = self.vectorizer.transform(df["text_content"])
        url_features = extract_url_features(
            df["url"]).to_numpy(dtype="float32")
        content_features = extract_content_features(
            df["text_content"]).to_numpy(dtype="float32")
        return sp.hstack(
            [
                text_features,
                sp.csr_matrix(url_features),
                sp.csr_matrix(content_features),
            ],
            format="csr",
        )

    def _save_to_cache(self, X: sp.csr_matrix, y: np.ndarray,
                       urls: np.ndarray):
        cfg = self.config
        cfg.features_file.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(cfg.features_file, X)
        np.save(cfg.labels_file, y)
        np.save(cfg.urls_file, urls)
        log_memory(f"Cached features {X.shape} and labels to disk.")


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
        log_memory("Splitting data for training and testing...")
        X_tr, X_te, y_tr, y_te, _, urls_te = train_test_split(
            X,
            y,
            urls,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        del X, y, urls
        gc.collect()

        log_memory("Training LightGBM model...")
        booster = self._train_model(X_tr, y_tr, X_te, y_te)
        del X_tr, y_tr
        gc.collect()

        log_memory("Evaluating model...")
        self._evaluate_model(booster, X_te, y_te, urls_te)

        log_memory("Saving model artifacts...")
        self._save_artifacts(booster, vectorizer)

    def _train_model(self, X_tr: sp.csr_matrix, y_tr: np.ndarray,
                     X_te: sp.csr_matrix, y_te: np.ndarray) -> lgb.Booster:
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_te, label=y_te, reference=train_data)
        params = self.config.lgbm_params.copy()
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

    def _evaluate_model(self, booster: lgb.Booster, X_te: sp.csr_matrix,
                        y_te: np.ndarray, urls_te: np.ndarray):
        cfg = self.config
        pred_probs = booster.predict(X_te)
        preds = (pred_probs > 0.5).astype(int)

        self._log_misclassified(y_te, preds, urls_te)

        report = classification_report(y_te,
                                       preds,
                                       target_names=cfg.id_to_label.values(),
                                       digits=4)
        print("\n--- Final Classification Report ---\n")
        print(report)
        (cfg.reports_dir / "lgbm_classification_report.txt").write_text(report)

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_te,
                                                preds,
                                                display_labels=list(
                                                    cfg.id_to_label.values()),
                                                cmap="Blues",
                                                ax=ax)
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(cfg.reports_dir / "lgbm_confusion_matrix.png")
        plt.close(fig)
        log_memory("Saved classification report and confusion matrix.")

    def _log_misclassified(self, y_true: np.ndarray, y_pred: np.ndarray,
                           urls: np.ndarray):
        cfg = self.config
        misclassified_indices = np.where(y_true != y_pred)[0]
        if len(misclassified_indices) > 0:
            error_df = pd.DataFrame({
                "url":
                urls[misclassified_indices],
                "true_label":
                y_true[misclassified_indices],
                "predicted_label":
                y_pred[misclassified_indices],
            })
            error_df["true_label"] = error_df["true_label"].map(
                cfg.id_to_label)
            error_df["predicted_label"] = error_df["predicted_label"].map(
                cfg.id_to_label)
            error_log_path = cfg.reports_dir / "misclassified_urls.csv"
            error_df.to_csv(error_log_path, index=False)
            log_memory(
                f"Saved {len(error_df)} misclassified URLs to {error_log_path}"
            )

    def _save_artifacts(self, booster: lgb.Booster,
                        vectorizer: HashingVectorizer):
        artifact = {"vectorizer": vectorizer, "model": booster}
        artifact_path = self.config.models_dir / self.config.model_artifact_name
        joblib.dump(artifact, artifact_path)
        log_memory(f"Model artifacts saved to {artifact_path}")


def set_memory_cap(gb: int | None):
    if gb is None:
        return
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (gb * 1024**3, hard))
        log_memory(f"Soft memory limit set to {gb} GB")
    except (ValueError, OSError) as e:
        logger.warning(
            f"Could not set memory cap (not available on all OS): {e}")


def log_memory(msg: str):
    mem_gb = psutil.Process().memory_info().rss / 1e9
    logger.info(f"[{mem_gb:5.2f} GB] {msg}")


def main(rebuild: bool):
    log_memory("--- Pipeline Start ---")
    config = PipelineConfig()
    set_memory_cap(config.mem_limit_gb)

    config.models_dir.mkdir(parents=True, exist_ok=True)
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
        action="store_true",
        help="Force rebuilding the feature matrix and URL cache",
    )
    args = ap.parse_args()
    main(rebuild=args.rebuild)
