"""
01_prepare_data.py   –  High-throughput structured text extraction (Production Edition)
──────────────────────────────────────────────────────────────────────────────────────
• Implements a robust producer-consumer pattern for smooth, observable progress.
• Provides periodic status updates (progress %, stats, memory, rate) to the console.
• Writes directly to compressed Parquet chunks, eliminating storage bottlenecks.
• Fully resumable and fault-tolerant.
"""

from __future__ import annotations

import csv
import gc
import os
import pathlib
import re
import time
import warnings
from concurrent.futures import as_completed, FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import local
from typing import Dict, Iterator, List

import pandas as pd
import psutil
import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup, SoupStrainer, XMLParsedAsHTMLWarning
from tqdm import tqdm

from src.config import (
    ENRICHED_DATA_FILE,
    MAX_WORKERS,
    PROCESSED_DATA_DIR,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
)
from src.utils import get_logger

logger = get_logger(__name__)

URL_LIST_FILE = PROCESSED_DATA_DIR / "urls_to_fetch.csv"
LOG_SUCCESS = False
LOG_INTERVAL_SECONDS = 30
MAX_INFLIGHT = MAX_WORKERS * 4

TARGET_TAGS = (
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "blockquote",
    "pre",
    "article",
)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
_thread_local = local()

try:
    import lxml

    PARSER = "lxml"
except ImportError:
    PARSER = "html.parser"
    logger.warning(
        "lxml not found, falling back to html.parser. For faster performance, run 'pip install lxml'."
    )


@dataclass
class Stats:
    success: int = 0
    failed: int = 0
    skipped: int = 0
    total: int = 0
    processed_since_log: int = 0

    def get_progress_str(self, elapsed_time: float) -> str:
        percent_done = (
            (self.success + self.failed) / self.total * 100 if self.total > 0 else 0
        )
        rate = self.processed_since_log / elapsed_time if elapsed_time > 0 else 0
        return (
            f"Progress: {percent_done:5.1f}% | "
            f"Success: {self.success}, Failed: {self.failed} | "
            f"Rate: {rate:5.1f} url/s"
        )


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def log(msg: str, t0: float | None = None):
    dt = f"{time.time() - t0:6.2f}s" if t0 else ""
    logger.info(f"{msg:<70} | mem={rss_mb():8.2f} MB {dt}")


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.headers.update(REQUEST_HEADERS)
        _thread_local.session = session
    return _thread_local.session


def extract_structured_text(html: str) -> List[Dict[str, str]]:
    strainer = SoupStrainer(TARGET_TAGS)
    soup = BeautifulSoup(html, PARSER, parse_only=strainer)
    content = []
    for tag in soup.find_all(TARGET_TAGS):
        text = re.sub(r"\s+", " ", tag.get_text(strip=True)).strip()
        if text:
            content.append({"tag": tag.name, "text": text})
    soup.decompose()
    return content


def fetch(url: str, label: int) -> dict:
    """Fetches URL and returns a dict with status, ready for processing."""
    out = {
        "url": url,
        "label": label,
        "structured_text": None,
        "html_content": None,
        "status": "failed",
    }
    try:
        session = get_session()
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        html = response.text
        extracted_content = extract_structured_text(html)

        if extracted_content:
            out.update(
                {
                    "structured_text": extracted_content,
                    "html_content": html,
                    "status": "success",
                }
            )
            if LOG_SUCCESS:
                logger.info(f"Success: {url}")
        else:
            out["html_content"] = html
            out["status"] = "success_empty"
    except requests.RequestException as e:
        logger.warning(f"Request failed for {url}: {type(e).__name__}")
    except Exception as e:
        logger.error(f"Unexpected error for {url}: {e}", exc_info=True)
    return out


def generate_urls_to_fetch() -> Iterator[Dict]:
    """Generator that yields URL and label dicts from the input CSV."""
    with open(URL_LIST_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {"url": row["url"], "label": int(row["label"])}


def consolidate_chunks(chunk_dir: pathlib.Path, final_file: pathlib.Path):
    """Reads all _temp_*.parquet files, combines them, and cleans up."""
    log(f"Consolidating temporary chunks from {chunk_dir}...")
    temp_files = sorted(list(chunk_dir.glob("_temp_*.parquet")))
    if not temp_files:
        log("No temporary chunks found to consolidate.")
        return

    if final_file.exists():
        final_file.unlink()

    writer = None
    with tqdm(total=len(temp_files), desc="Consolidating") as pbar:
        for i, chunk_file in enumerate(temp_files):
            try:
                table = pq.read_table(chunk_file)
                if i == 0:
                    writer = pq.ParquetWriter(
                        final_file, table.schema, compression="ZSTD"
                    )
                writer.write_table(table)
            except Exception as e:
                logger.error(f"Failed to read or write chunk {chunk_file}: {e}")
            finally:
                chunk_file.unlink()
                pbar.update(1)

    if writer:
        writer.close()
    try:
        chunk_dir.rmdir()
    except OSError:
        pass
    log("Consolidation complete.")


def main() -> None:
    t_global = time.time()
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    TEMP_CHUNK_DIR = PROCESSED_DATA_DIR / "temp_chunks"
    TEMP_CHUNK_DIR.mkdir(exist_ok=True)
    log("─ Start Structured Text Extraction (Production Edition) ─")

    if not URL_LIST_FILE.exists():
        logger.error(f"URL list missing: {URL_LIST_FILE}")
        return

    processed_urls = set()
    for chunk_file in TEMP_CHUNK_DIR.glob("_temp_*.parquet"):
        try:
            processed_urls.update(
                pq.read_table(chunk_file, columns=["url"])["url"].to_pylist()
            )
        except Exception:
            logger.warning(
                f"Could not read chunk {chunk_file} to get seen URLs. It may be corrupt."
            )

    all_urls = list(generate_urls_to_fetch())
    urls_to_process = [item for item in all_urls if item["url"] not in processed_urls]

    stats = Stats(total=len(urls_to_process), skipped=len(processed_urls))

    if not urls_to_process:
        log("All URLs have already been processed in chunks.")
        consolidate_chunks(TEMP_CHUNK_DIR, ENRICHED_DATA_FILE)
        log("─ Done ─", t_global)
        return

    log(
        f"Starting fetch for {stats.total} new URLs. ({stats.skipped} previously completed)"
    )

    CHUNK_SIZE = 1000
    results_buffer = []
    last_log_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, tqdm(
            total=stats.total, desc="Fetching URLs"
        ) as pbar:
            inflight = {}
            url_iterator = iter(urls_to_process)

            while True:
                while len(inflight) < MAX_INFLIGHT:
                    try:
                        item = next(url_iterator)
                        future = executor.submit(fetch, item["url"], item["label"])
                        inflight[future] = item["url"]
                    except StopIteration:
                        url_iterator = None
                        break

                if not inflight:
                    break

                done, _ = wait(
                    inflight.keys(), timeout=1.0, return_when=FIRST_COMPLETED
                )

                for future in done:
                    url = inflight.pop(future)
                    try:
                        result = future.result()
                        if result.get("status") in ("success", "success_empty"):
                            stats.success += 1
                            del result["status"]
                            results_buffer.append(result)
                        else:
                            stats.failed += 1
                    except Exception as e:
                        logger.error(f"Worker for {url} raised exception: {e}")
                        stats.failed += 1

                    pbar.update(1)
                    stats.processed_since_log += 1

                if len(results_buffer) >= CHUNK_SIZE:
                    chunk_num = len(list(TEMP_CHUNK_DIR.glob("_temp_*.parquet")))
                    pd.DataFrame(results_buffer).to_parquet(
                        TEMP_CHUNK_DIR / f"_temp_chunk_{chunk_num}.parquet",
                        compression="ZSTD",
                        index=False,
                    )
                    results_buffer.clear()
                    gc.collect()

                current_time = time.time()
                if current_time - last_log_time >= LOG_INTERVAL_SECONDS:
                    elapsed = current_time - last_log_time
                    log(stats.get_progress_str(elapsed))
                    last_log_time = current_time
                    stats.processed_since_log = 0

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Saving final buffer...")
    finally:
        if results_buffer:
            chunk_num = len(list(TEMP_CHUNK_DIR.glob("_temp_*.parquet")))
            pd.DataFrame(results_buffer).to_parquet(
                TEMP_CHUNK_DIR / f"_temp_chunk_{chunk_num+1}.parquet",
                compression="ZSTD",
                index=False,
            )

        log(
            f"Final Stats: Success={stats.success}, Failed={stats.failed}, Skipped={stats.skipped}"
        )

    consolidate_chunks(TEMP_CHUNK_DIR, ENRICHED_DATA_FILE)
    log("─ All Done ─", t_global)


if __name__ == "__main__":
    main()
