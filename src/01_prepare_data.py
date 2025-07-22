"""
01_prepare_data.py   –  High-throughput structured text extraction
──────────────────────────────────────────────────────────────────
• Extracts text from specific HTML tags (p, h1, li, etc.)
• Stores data in a structured format: [{'tag': 'h1', 'text': '...'}, ...]
• Uses a targeted parser for extreme efficiency (only parses relevant tags)
• Per-future hard deadline and graceful shutdown for robustness
• Creates a highly-optimized Parquet file with a nested schema
"""

from __future__ import annotations

import csv
import gc
import json
import os
import pathlib
import re
import time
import warnings
from concurrent.futures import (
    CancelledError,
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from threading import local
from typing import Dict, Iterator, List, Tuple

import psutil
import pyarrow as pa
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
    TEMP_JSONL_FILE,
)
from src.utils import get_logger

logger = get_logger(__name__)

URL_LIST_FILE = PROCESSED_DATA_DIR / "urls_to_fetch.csv"
MAX_INFLIGHT = MAX_WORKERS * 4
HARD_DEADLINE_SECONDS = 45
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
    cancelled: int = 0
    skipped: int = 0
    total: int = 0

    def __str__(self):
        return (f"Completed: {self.success} | "
                f"Failed: {self.failed} | "
                f"Cancelled: {self.cancelled} | "
                f"Skipped: {self.skipped} | "
                f"Total: {self.total}")


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
    out = {
        "url": url,
        "label": label,
        "structured_text": None,
        "status": "failed",
    }
    try:
        session = get_session()
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        extracted_content = extract_structured_text(response.text)
        if extracted_content:
            out["structured_text"] = extracted_content
            out["status"] = "success"
            if LOG_SUCCESS:
                logger.info(
                    f"Success: {url} ({len(extracted_content)} text blocks)")
        else:
            out["status"] = "success_empty"
            logger.info(f"Success (empty): No target text found in {url}")

    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed {url}: {type(e).__name__}")
    except Exception as e:
        logger.error(f"An unexpected error occurred for {url}: {e}",
                     exc_info=True)

    return out


def get_seen_urls() -> set[str]:
    if not TEMP_JSONL_FILE.exists():
        return set()

    done = set()
    with open(TEMP_JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("status",
                              "").startswith("success") and record.get("url"):
                    done.add(record["url"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def generate_urls_to_fetch() -> Iterator[Dict]:
    seen = get_seen_urls()
    logger.info(f"Found {len(seen)} previously processed URLs to skip.")

    with open(URL_LIST_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["url"] not in seen:
                yield {"url": row["url"], "label": int(row["label"])}


def jsonl_to_parquet(src: pathlib.Path, dst: pathlib.Path):
    log("JSONL → Parquet conversion start")
    schema = pa.schema([
        pa.field("url", pa.string()),
        pa.field("label", pa.int64()),
        pa.field(
            "structured_text",
            pa.list_(
                pa.struct([
                    pa.field("tag", pa.string()),
                    pa.field("text", pa.string())
                ])),
        ),
    ])

    def read_chunks():
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("status") == "success":
                        del record["status"]
                        yield record
                except json.JSONDecodeError:
                    continue

    writer = None
    rows_to_write = []
    chunk_size = 1000

    with tqdm(desc="Converting to Parquet", unit=" rows") as pbar:
        for record in read_chunks():
            rows_to_write.append(record)
            if len(rows_to_write) >= chunk_size:
                table = pa.Table.from_pylist(rows_to_write, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(dst,
                                              table.schema,
                                              compression="ZSTD")
                writer.write_table(table)
                pbar.update(len(rows_to_write))
                rows_to_write = []

        if rows_to_write:
            table = pa.Table.from_pylist(rows_to_write, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(dst,
                                          table.schema,
                                          compression="ZSTD")
            writer.write_table(table)
            pbar.update(len(rows_to_write))

    if writer:
        writer.close()
    log("Conversion finished")


def main() -> None:
    t_global = time.time()
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    log("─ Start Structured Text Extraction ─")

    if not URL_LIST_FILE.exists():
        logger.error(f"URL list missing: {URL_LIST_FILE}")
        return

    stats = Stats()
    urls_to_process = list(generate_urls_to_fetch())
    stats.total = len(urls_to_process)
    stats.skipped = len(get_seen_urls())

    if not urls_to_process:
        log("All URLs have already been processed.")
        if TEMP_JSONL_FILE.exists():
            log("Finalizing Parquet file from existing temp data.")
            jsonl_to_parquet(TEMP_JSONL_FILE, ENRICHED_DATA_FILE)
        log("─ Done ─", t_global)
        return

    log(f"Starting fetch for {stats.total} URLs.")

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, open(
                TEMP_JSONL_FILE, "a",
                encoding="utf-8") as f_out, tqdm(total=stats.total,
                                                 desc="Fetching URLs",
                                                 unit="url") as pbar:
            inflight: Dict[Future, Tuple[str, float]] = {}
            url_iterator = iter(urls_to_process)

            while True:
                while len(inflight) < MAX_INFLIGHT:
                    try:
                        item = next(url_iterator)
                        future = executor.submit(fetch, item["url"],
                                                 item["label"])
                        inflight[future] = (item["url"], time.time())
                    except StopIteration:
                        break

                if not inflight:
                    break

                done, _ = wait(inflight.keys(),
                               timeout=1.0,
                               return_when=FIRST_COMPLETED)
                for future in done:
                    url, _ = inflight.pop(future)
                    try:
                        result = future.result()
                        f_out.write(
                            json.dumps(result, ensure_ascii=False) + "\n")
                        status = result.get("status", "failed")
                        if status.startswith("success"):
                            stats.success += 1
                        else:
                            stats.failed += 1
                    except CancelledError:
                        logger.warning(f"Cancelled: {url}")
                        stats.cancelled += 1
                    except Exception as e:
                        logger.error(
                            f"Worker for {url} raised unexpected exception: {e}",
                            exc_info=True,
                        )
                        stats.failed += 1
                    pbar.update(1)

                now = time.time()
                for future, (url, submit_time) in list(inflight.items()):
                    if now - submit_time > HARD_DEADLINE_SECONDS:
                        future.cancel()
                        logger.warning(
                            f"Hard Deadline Exceeded: Cancelling {url}")

                f_out.flush()

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Shutting down gracefully...")
    finally:
        log("─ Fetching complete. Finalizing data. ─")
        log(f"Final Stats: {stats}")

    if TEMP_JSONL_FILE.exists():
        if ENRICHED_DATA_FILE.exists():
            ENRICHED_DATA_FILE.unlink()
        jsonl_to_parquet(TEMP_JSONL_FILE, ENRICHED_DATA_FILE)
        TEMP_JSONL_FILE.unlink()

    log("─ All Done ─", t_global)


if __name__ == "__main__":
    main()
