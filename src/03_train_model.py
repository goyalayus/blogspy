"""
BlogSpy – fault-tolerant, memory-safe, and parallelized training pipeline

[FINAL VERSION - WITH DETAILED DIAGNOSTIC LOGGING]
This script performs the full model training pipeline and is optimized for
multi-core machines.

1.  Loads or builds a feature matrix from the enriched Parquet data.
    - If --rebuild is used, it now uses a parallel multiprocessing approach
      and provides DETAILED real-time logs on memory usage per worker,
      accumulator size, and swap usage.
2.  Implements aggressive memory management to prevent out-of-memory errors.
3.  Trains a LightGBM model, evaluates its performance, and saves the final
    model artifact (containing both the trained model and its vectorizer).
4.  Adds error analysis by saving incorrectly classified test set URLs
    to a CSV file for review.
"""

from __future__ import annotations
from src.utils import get_logger
from src.feature_engineering import (extract_content_features,
                                     extract_structural_features,
                                     extract_url_features)
from src.config import (ENRICHED_DATA_FILE, HASH_DIM, ID_TO_LABEL, LGBM_PARAMS,
                        MEM_LIMIT_GB, MODELS_DIR, RANDOM_STATE, REPORTS_DIR,
                        TEST_SIZE)

# ---------- Standard Library Imports ---------------------------------------
import argparse
import gc
import multiprocessing
import os
import pathlib
import resource
import sys
import warnings
from typing import Dict, List, Tuple

# ---------- Third-Party Imports --------------------------------------------
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

# ---------- Local Imports --------------------------------------------------
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))


# ---------------------------------------------------------------------------

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
logger = get_logger(__name__)

# --- File Paths for Caching ---
PROCESSED_DATA_DIR = project_root / "data/processed"
FEATURES_FILE = PROCESSED_DATA_DIR / "X_features.npz"
LABELS_FILE = PROCESSED_DATA_DIR / "y_labels.npy"
URLS_FILE = PROCESSED_DATA_DIR / "urls.npy"

# Used to create a consistent vectorizer in multiple places
VECTORIZER_NGRAM_RANGE = (1, 2)


def log_memory(msg: str):
    """
    [LOGGING ENHANCEMENT] Logs a message with current process memory
    AND system-wide swap usage.
    """
    process_mem_gb = psutil.Process().memory_info().rss / 1e9
    swap = psutil.swap_memory()
    swap_str = f"SWAP: {swap.used / 1e9:.2f}/{swap.total / 1e9:.2f} GB"
    logger.info(f"{f'[{process_mem_gb:.2f} GB]':<10} {swap_str:<25} {msg}")


def _set_memory_cap(gb: int | None) -> None:
    if gb is None or sys.platform == "win32":
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (gb * 1024**3, hard))
        log_memory(f"Soft memory limit (RLIMIT_AS) set to {gb} GB")
    except (ValueError, ModuleNotFoundError) as e:
        logger.warning(f"Could not set memory cap: {e}")


# ───────────────── Feature Engineering (Parallelized with Logging) ─────────

def _vectorize_chunk(df: pd.DataFrame, hv: HashingVectorizer) -> sp.csr_matrix:
    """Vectorizes a single DataFrame chunk into a sparse feature matrix."""
    text_features = hv.transform(df["text_content"].fillna(""))
    url_features = extract_url_features(df["url"]).to_numpy(dtype="float32")
    structural_features = extract_structural_features(
        df["html_content"]).to_numpy(dtype="float32")
    content_features = extract_content_features(
        df["text_content"]).to_numpy(dtype="float32")
    return sp.hstack([
        text_features, sp.csr_matrix(url_features),
        sp.csr_matrix(structural_features), sp.csr_matrix(content_features)
    ], format="csr")


def _process_chunk_worker(chunk_index: int) -> tuple[sp.csr_matrix | None, np.ndarray | None, np.ndarray | None, float]:
    """
    [LOGGING ENHANCEMENT] Worker function executed by each parallel process.
    It now logs its own status and memory usage.
    """
    pid = os.getpid()
    process_name = f"PID[{pid}]"
    logger.info(f"{process_name:<10} START: Processing chunk {chunk_index}")

    hv = HashingVectorizer(
        stop_words="english", n_features=HASH_DIM,
        ngram_range=VECTORIZER_NGRAM_RANGE, alternate_sign=False,
        norm="l2", dtype=np.float32
    )
    pf = pq.ParquetFile(ENRICHED_DATA_FILE)

    required_cols = ["url", "html_content", "text_content", "label"]
    df_chunk = pf.read_row_group(
        chunk_index, columns=required_cols).to_pandas().dropna(subset=["url", "label"])

    if df_chunk.empty:
        logger.warning(
            f"{process_name:<10} SKIP: Chunk {chunk_index} was empty.")
        return None, None, None, 0.0

    X_chunk = _vectorize_chunk(df_chunk, hv)
    y_chunk = df_chunk["label"].to_numpy(dtype="int8")
    urls_chunk = df_chunk["url"].to_numpy(dtype=object)

    # [LOGGING ENHANCEMENT] Calculate and log memory usage for this specific worker/chunk
    worker_mem_mb = psutil.Process(pid).memory_info().rss / 1e6
    result_size_mb = (X_chunk.data.nbytes +
                      y_chunk.nbytes + urls_chunk.nbytes) / 1e6
    logger.info(
        f"{process_name:<10} DONE:  Chunk {chunk_index}. Worker Mem: {worker_mem_mb: >5.0f} MB. Result Size: {result_size_mb:.2f} MB"
    )

    return X_chunk, y_chunk, urls_chunk, result_size_mb


def build_features_from_scratch() -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, HashingVectorizer]:
    """
    [LOGGING ENHANCEMENT] Builds the complete feature matrix in parallel and now
    includes detailed logging of the accumulator size in the main process.
    """
    log_memory("Starting PARALLEL feature building from scratch...")

    pf = pq.ParquetFile(ENRICHED_DATA_FILE)
    num_chunks = pf.num_row_groups
    del pf

    num_workers = max(1, multiprocessing.cpu_count() - 1)
    log_memory(
        f"Distributing {num_chunks} chunks across {num_workers} worker processes...")

    results = []
    total_results_size_mb = 0.0
    log_interval = 20  # How often to log the main process status

    with multiprocessing.Pool(processes=num_workers) as pool, \
            tqdm(total=num_chunks, desc="Processing Chunks (Parallel)") as pbar:

        # [LOGGING ENHANCEMENT] Unrolling the list comprehension into a for loop
        # to allow for periodic logging inside the main process.
        for i, result_chunk in enumerate(pool.imap(_process_chunk_worker, range(num_chunks))):
            if result_chunk[0] is not None:
                results.append(result_chunk)
                # Add the size of the returned result
                total_results_size_mb += result_chunk[3]
            pbar.update(1)

            # [LOGGING ENHANCEMENT] Log main process status periodically
            if (i + 1) % log_interval == 0 or (i + 1) == num_chunks:
                msg = (f"MAIN PROCESS holding {len(results)} results. "
                       f"Accumulated Result Size: {total_results_size_mb / 1024:.2f} GB")
                log_memory(msg)

    # Filter out empty results one last time (though handled in loop)
    results = [res for res in results if res[0] is not None]
    X_parts, y_parts, url_parts, _ = zip(*results)

    log_memory("Finished processing all chunks. Now stacking results...")
    X = sp.vstack(X_parts, format="csr")
    y = np.concatenate(y_parts)
    urls = np.concatenate(url_parts)
    log_memory(
        f"Final feature matrix created: X{X.shape}, y{y.shape}, urls{urls.shape}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sp.save_npz(FEATURES_FILE, X)
    np.save(LABELS_FILE, y)
    np.save(URLS_FILE, urls)
    log_memory(f"Feature cache written to disk at '{PROCESSED_DATA_DIR}'.")

    final_hv = HashingVectorizer(
        stop_words="english", n_features=HASH_DIM,
        ngram_range=VECTORIZER_NGRAM_RANGE, alternate_sign=False,
        norm="l2", dtype=np.float32
    )
    return X, y, urls, final_hv


def load_or_build_features(rebuild: bool) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, HashingVectorizer]:
    """
    Loads features from cache if available. Otherwise, builds them from scratch
    using a parallelized process.
    """
    # NOTE: This logic for inferring n_features is slightly brittle. If you add
    # more non-text features, the '- 12' will need to be updated.
    num_non_text_features = 12

    if (not rebuild) and FEATURES_FILE.exists() and LABELS_FILE.exists() and URLS_FILE.exists():
        log_memory(
            "Found cached features, labels, and URLs. Loading from disk...")
        X = sp.load_npz(FEATURES_FILE)
        y = np.load(LABELS_FILE)
        urls = np.load(URLS_FILE, allow_pickle=True)
        log_memory(
            f"Loaded X{X.shape}, y{y.shape}, urls{urls.shape} from cache.")

        # Infer HASH_DIM from the shape of the loaded matrix
        inferred_hash_dim = X.shape[1] - num_non_text_features
        hv = HashingVectorizer(
            stop_words="english", n_features=inferred_hash_dim,
            ngram_range=VECTORIZER_NGRAM_RANGE, alternate_sign=False,
            norm="l2", dtype=np.float32
        )
        return X, y, urls, hv

    return build_features_from_scratch()


def _save_incorrect_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, pred_probs: np.ndarray,
    urls: np.ndarray, output_path: pathlib.Path, id_to_label_map: Dict[int, str]
) -> None:
    # This function remains unchanged as its logic is sound.
    incorrect_indices = np.where(y_true != y_pred)[0]
    if len(incorrect_indices) == 0:
        logger.info(
            "✅ No incorrect predictions found on the test set. Excellent!")
        return

    pred_is_binary_prob = len(pred_probs.shape) == 1

    incorrect_data = {
        "url": urls[incorrect_indices], "true_label": y_true[incorrect_indices],
        "predicted_label": y_pred[incorrect_indices],
    }

    df_incorrect = pd.DataFrame(incorrect_data)
    df_incorrect["true_label_name"] = df_incorrect["true_label"].map(
        id_to_label_map)
    df_incorrect["predicted_label_name"] = df_incorrect["predicted_label"].map(
        id_to_label_map)

    if pred_is_binary_prob:
        df_incorrect["confidence"] = pred_probs[incorrect_indices]
        df_incorrect["confidence_in_prediction"] = np.where(
            df_incorrect["predicted_label"] == 1,
            df_incorrect["confidence"], 1 - df_incorrect["confidence"],
        )

    cols_to_keep = ["url", "true_label_name",
                    "predicted_label_name", "confidence_in_prediction"]
    df_incorrect = df_incorrect[[
        c for c in cols_to_keep if c in df_incorrect.columns]]

    df_incorrect.to_csv(output_path, index=False)
    log_memory(
        f"Saved {len(df_incorrect)} incorrect predictions to {output_path}")


# ─────────────────────────── Main Training Pipeline ──────────────────────────

def main(rebuild: bool) -> None:
    # This function remains unchanged as its logic is sound.
    log_memory("--- Pipeline Start ---")
    _set_memory_cap(MEM_LIMIT_GB)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, urls, vectorizer = load_or_build_features(rebuild)

    num_classes = len(np.unique(y))
    if num_classes < 2:
        logger.error(
            f"Training requires at least 2 classes, but found {num_classes}. Aborting.")
        return

    X_tr, X_te, y_tr, y_te, _, urls_te = train_test_split(
        X, y, urls, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log_memory(f"Data split into Train {X_tr.shape} and Test {X_te.shape}")
    del X, y, urls
    gc.collect()
    log_memory("Full feature matrix (X, y, urls) cleared from RAM.")

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_te, label=y_te, reference=train_data)
    log_memory("LightGBM Dataset objects created.")
    del X_tr, y_tr
    gc.collect()
    log_memory("Original training matrix (X_tr, y_tr) cleared from RAM.")

    params = LGBM_PARAMS.copy()
    if num_classes > 2:
        params.update({"objective": "multiclass", "num_class": num_classes})

    num_boost_round = params.pop("n_estimators", 1000)
    callbacks = [lgb.early_stopping(
        stopping_rounds=50, verbose=True), lgb.log_evaluation(period=100)]

    log_memory(f"Starting lgb.train with max_bin={params.get('max_bin')}...")
    booster = lgb.train(
        params, train_data, num_boost_round=num_boost_round,
        valid_sets=[val_data], valid_names=['eval'], callbacks=callbacks
    )
    log_memory("Model training finished successfully.")

    log_memory("Evaluating model...")
    pred_probs = booster.predict(X_te)

    if num_classes > 2:
        preds = np.argmax(pred_probs, axis=1)
    else:
        preds = (pred_probs > 0.5).astype(int)

    target_names = [ID_TO_LABEL[i] for i in sorted(ID_TO_LABEL.keys())]
    report = classification_report(
        y_te, preds, target_names=target_names, digits=4)
    print("\n--- Final Classification Report ---\n")
    print(report)
    (REPORTS_DIR / "classification_report.txt").write_text(report)
    log_memory("Classification report saved.")

    _save_incorrect_predictions(
        y_true=y_te, y_pred=preds, pred_probs=pred_probs, urls=urls_te,
        output_path=REPORTS_DIR / "incorrect_predictions.csv", id_to_label_map=ID_TO_LABEL
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_te, preds, display_labels=target_names, cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png")
    plt.close(fig)
    log_memory("Confusion matrix plot saved.")

    artifact = {"vectorizer": vectorizer, "model": booster}
    model_path = MODELS_DIR / "ayush.joblib"
    joblib.dump(artifact, model_path, compress=3)
    log_memory(f"Final model artifact saved to '{model_path}'. ✅")
    log_memory("--- Pipeline End ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the main BlogSpy model.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the feature matrix from scratch, ignoring any cache."
    )
    args = parser.parse_args()
    main(rebuild=args.rebuild)
