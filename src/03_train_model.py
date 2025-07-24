"""
BlogSpy – fault-tolerant, memory-safe training pipeline

[FIX-4] Re-adds missing constants (HASH_DIM, MEM_LIMIT_GB) and allows
re-running from the cached feature matrix to avoid rebuilding.
"""

from __future__ import annotations

from src.utils import get_logger
from src.feature_engineering import (
    extract_url_features,
    extract_structural_features,
    extract_content_features,
)
from src.config import (
    ENRICHED_DATA_FILE,
    TEST_SIZE,
    RANDOM_STATE,
    MODELS_DIR,
    REPORTS_DIR,
    ID_TO_LABEL,
    LGBM_PARAMS,
)

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import HashingVectorizer
import lightgbm as lgb
from bs4 import XMLParsedAsHTMLWarning
import scipy.sparse as sp
import pyarrow.parquet as pq
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

import warnings
import gc
import argparse
import sys
import pathlib
import resource
from typing import List

project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
logger = get_logger(__name__)

HASH_DIM = 2**16
MEM_LIMIT_GB = 6
FEATURES_FILE = project_root / "data/processed/X_hashtfidf.npz"
LABELS_FILE = project_root / "data/processed/y.npy"


def log_memory(msg: str):
    mem_gb = psutil.Process().memory_info().rss / 1e9
    logger.info(f"{f'[{mem_gb:.2f} GB]':<10} {msg}")


def _set_memory_cap(gb: int | None) -> None:
    if gb is None:
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (gb * 1024**3, hard))
        log_memory(f"Soft RLIMIT_AS set to {gb} GB")
    except Exception as e:
        logger.warning(f"Could not set memory cap: {e}")


def _vectorise_chunk(df: pd.DataFrame, hv: HashingVectorizer) -> sp.csr_matrix:
    txt = hv.transform(df["text_content"].fillna(""))
    url_dense = extract_url_features(df["url"]).to_numpy(dtype="float32")
    structural = extract_structural_features(df["html_content"]).to_numpy(
        dtype="float32"
    )
    content = extract_content_features(df["text_content"]).to_numpy(dtype="float32")
    return sp.hstack(
        [
            txt,
            sp.csr_matrix(url_dense),
            sp.csr_matrix(structural),
            sp.csr_matrix(content),
        ],
        format="csr",
    )


def build_features() -> tuple[sp.csr_matrix, np.ndarray, HashingVectorizer]:
    hv = HashingVectorizer(
        stop_words="english",
        n_features=HASH_DIM,
        ngram_range=(1, 2),
        alternate_sign=False,
        norm="l2",
        dtype=np.float32,
    )
    X_parts: List[sp.csr_matrix] = []
    y_parts: List[np.ndarray] = []

    log_memory("Starting feature building process...")
    pf = pq.ParquetFile(ENRICHED_DATA_FILE)
    wanted = ["url", "html_content", "text_content", "label"]
    for i in tqdm(range(pf.num_row_groups), desc="Processing Chunks"):
        df = (
            pf.read_row_group(i, columns=wanted)
            .to_pandas()
            .dropna(subset=["html_content", "text_content"])
        )
        if df.empty:
            continue
        X_parts.append(_vectorise_chunk(df, hv))
        y_parts.append(df["label"].to_numpy(dtype="int8"))

    log_memory("Finished reading chunks. Now stacking matrices...")
    X = sp.vstack(X_parts, format="csr")
    y = np.concatenate(y_parts)
    log_memory(f"Final corpus created: {X.shape}")

    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(FEATURES_FILE, X)
    np.save(LABELS_FILE, y)
    log_memory("Feature cache written to disk.")
    return X, y, hv


def load_or_build_features(
    rebuild: bool,
) -> tuple[sp.csr_matrix, np.ndarray, HashingVectorizer]:
    hv = HashingVectorizer(
        stop_words="english",
        n_features=HASH_DIM,
        ngram_range=(1, 2),
        alternate_sign=False,
        norm="l2",
        dtype=np.float32,
    )

    if (not rebuild) and FEATURES_FILE.exists() and LABELS_FILE.exists():
        log_memory("Found cached features. Loading from disk...")
        X = sp.load_npz(FEATURES_FILE)
        y = np.load(LABELS_FILE)
        log_memory(f"Loaded X{X.shape}, y{y.shape} from cache.")
        return X, y, hv

    log_memory(
        "No cache found or --rebuild flag used. Building features from scratch..."
    )
    return build_features()


def main(rebuild: bool) -> None:
    log_memory("--- Pipeline Start ---")
    _set_memory_cap(MEM_LIMIT_GB)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, vectorizer = load_or_build_features(rebuild)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log_memory(f"Data split into Train {X_tr.shape} and Test {X_te.shape}")
    del X, y
    gc.collect()
    log_memory("Full feature matrix (X, y) cleared from RAM.")

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_te, label=y_te, reference=train_data)
    log_memory("LightGBM Dataset objects created.")
    del X_tr, y_tr
    gc.collect()
    log_memory("Original training matrix (X_tr, y_tr) cleared.")

    params = LGBM_PARAMS.copy()
    num_boost_round = params.pop("n_estimators")
    callbacks = [
        lgb.early_stopping(stopping_rounds=25, verbose=True),
        lgb.log_evaluation(period=50),
    ]
    log_memory(
        f"Starting lgb.train with max_bin={params['max_bin']} and num_leaves={params['num_leaves']}..."
    )
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
        valid_names=["eval"],
        callbacks=callbacks,
    )
    log_memory("lgb.train has finished successfully.")

    log_memory("Evaluating model on the test set...")
    pred_probs = booster.predict(X_te)
    preds = (pred_probs > 0.5).astype(int)

    report = classification_report(
        y_te, preds, target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]], digits=4
    )
    print("\n--- Final Classification Report ---\n")
    print(report)
    (REPORTS_DIR / "lgbm_final_classification_report.txt").write_text(report)
    log_memory("Classification report saved.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_te,
        preds,
        display_labels=[ID_TO_LABEL[0], ID_TO_LABEL[1]],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix – Final Model")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "lgbm_final_classification_report.png")
    plt.close(fig)
    log_memory("Confusion-matrix plot saved.")

    artifact = {"vectorizer": vectorizer, "model": booster}
    joblib.dump(artifact, MODELS_DIR / "ayush.joblib")
    log_memory("Model artifacts (vectorizer + booster) saved. ✅")
    log_memory("--- Pipeline End ---")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the main BlogSpy model.")
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the feature matrix cache",
    )
    args = ap.parse_args()
    main(rebuild=args.rebuild)
