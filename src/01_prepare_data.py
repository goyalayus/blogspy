"""
01_prepare_data.py   –  High-throughput structured text extraction (Chunked Parquet Edition)
───────────────────────────────────────────────────────────────────────────────────────────
• Extracts structured text and saves the FULL raw HTML for advanced feature engineering.
• Bypasses large temporary text files by writing directly to compressed Parquet chunks.
• This method is highly efficient on disk space and robust against interruptions.
• At the end, it consolidates all temporary chunks into the final dataset.
"""

from __future__ import annotations

import csv
import gc
import os
import pathlib
import re
import time
import warnings
from concurrent.futures import as_completed, ThreadPoolExecutor
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
LOG_SUCCESS = True

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

    def __str__(self):
        return f"Completed: {self.success} | Failed: {self.failed} | Skipped: {self.skipped} | Total: {self.total}"


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def log(msg: str, t0: float | None = None):
    dt = f"{time.time() - t0:6.2f}s" if t0 else ""
    logger.info(f"{msg:<60} | mem={rss_mb():8.2f} MB {dt}")


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
    log("─ Start Structured Text Extraction (Chunked Mode) ─")

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

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {
                executor.submit(fetch, item["url"], item["label"]): item["url"]
                for item in urls_to_process
            }

            for future in tqdm(
                as_completed(future_to_url), total=stats.total, desc="Fetching"
            ):
                try:
                    result = future.result()
                    if result.get("status") == "success":
                        del result["status"]
                        results_buffer.append(result)
                        stats.success += 1
                    else:
                        stats.failed += 1
                except Exception as e:
                    logger.error(
                        f"Worker for {future_to_url[future]} raised exception: {e}"
                    )
                    stats.failed += 1

                if len(results_buffer) >= CHUNK_SIZE:
                    chunk_num = len(list(TEMP_CHUNK_DIR.glob("_temp_*.parquet")))
                    chunk_file = TEMP_CHUNK_DIR / f"_temp_chunk_{chunk_num}.parquet"
                    pd.DataFrame(results_buffer).to_parquet(
                        chunk_file, compression="ZSTD", index=False
                    )
                    results_buffer.clear()
                    gc.collect()

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Saving final buffer...")
    finally:
        if results_buffer:
            chunk_num = len(list(TEMP_CHUNK_DIR.glob("_temp_*.parquet")))
            chunk_file = TEMP_CHUNK_DIR / f"_temp_chunk_{chunk_num+1}.parquet"
            pd.DataFrame(results_buffer).to_parquet(
                chunk_file, compression="ZSTD", index=False
            )

        log("─ Fetching complete. Consolidating final data. ─")
        log(f"Final Stats: {stats}")

    consolidate_chunks(TEMP_CHUNK_DIR, ENRICHED_DATA_FILE)
    log("─ All Done ─", t_global)


if __name__ == "__main__":
    main()
