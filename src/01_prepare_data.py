"""
01_prepare_data.py   –  robust, low-RAM, timeout-hardened version
──────────────────────────────────────────────────────────────────
• At most MAX_WORKERS×4 futures alive (prevents RAM spikes)
• Per-future hard deadline (DNS + connect + read)   -> no stalls
• Graceful Ctrl-C  (--> executor.shutdown(cancel_futures=True))
• Optional gzip compression of the html_content column
"""

from __future__ import annotations
import csv
import gc
import json
import os
import pathlib
import re
import sys
import time
import warnings
import zlib
from concurrent.futures import (
    ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED, CancelledError
)
from typing import Dict, List, Iterator

import psutil
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup, SoupStrainer, XMLParsedAsHTMLWarning
from tqdm import tqdm

from src.config import (
    ENRICHED_DATA_FILE, MAX_WORKERS, PROCESSED_DATA_DIR,
    REQUEST_HEADERS, REQUEST_TIMEOUT, TEMP_JSONL_FILE
)
from src.utils import get_logger

logger = get_logger(__name__)


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def log(msg: str, t0: float | None = None):
    dt = f"{time.time() - t0:6.2f}s" if t0 else ""
    logger.info(f"{msg:<60} | mem={rss_mb():8.2f} MB {dt}")


URL_LIST_FILE = PROCESSED_DATA_DIR / "urls_to_fetch.csv"
BATCH_SIZE = 500
DELAY_BETWEEN_BATCHES = 0.05
MAX_INFLIGHT = MAX_WORKERS * 4
HARD_DEADLINE = 45
COMPRESS_HTML = False

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
PARSER_FILTER = SoupStrainer(text=True)


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml", parse_only=PARSER_FILTER)
    txt = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    soup.decompose()
    return txt


_local = {}


def sess() -> requests.Session:
    s = _local.get("session")
    if s is None:
        s = requests.Session()
        s.headers.update(REQUEST_HEADERS)
        _local["session"] = s
    return s

def fetch(url: str, label: int) -> dict:
    out = {"url": url, "label": label,
           "html_content": None, "text_content": None}
    try:
        r = sess().get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        html = r.text
        out["text_content"] = extract_text(html)
        if COMPRESS_HTML:
            out["html_content"] = zlib.compress(html.encode("utf-8"))
        else:
            out["html_content"] = html
    except Exception as e:
        logger.warning(f"Failed {url}: {e}")
    return out

def seen_urls() -> set[str]:
    done = set()
    if TEMP_JSONL_FILE.exists():
        with open(TEMP_JSONL_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["url"])
                except Exception:
                    pass
    return done


def url_stream() -> Iterator[List[Dict]]:
    already = seen_urls()
    with open(URL_LIST_FILE, encoding="utf-8") as f:
        rdr, buf = csv.DictReader(f), []
        for row in rdr:
            if row["url"] in already:
                continue
            buf.append({"url": row["url"], "label": int(row["label"])})
            if len(buf) == BATCH_SIZE:
                yield buf
                buf = []
        if buf:
            yield buf

def jsonl_to_parquet(src: pathlib.Path, dst: pathlib.Path, chunk=1000):
    log("JSONL → Parquet conversion start")
    schema = pa.schema([
        pa.field("url", pa.string()),
        pa.field("label", pa.int64()),
        pa.field("html_content",
                 pa.binary() if COMPRESS_HTML else pa.string()),
        pa.field("text_content", pa.string()),
    ])
    writer, rows = None, []
    with open(src, encoding="utf-8") as f:
        for ln in tqdm(f, desc="Convert"):
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
            if len(rows) >= chunk:
                tbl = pa.Table.from_pylist(rows, schema=schema)
                writer = writer or pq.ParquetWriter(dst, tbl.schema)
                writer.write_table(tbl)
                rows = []
        if rows:
            tbl = pa.Table.from_pylist(rows, schema=schema)
            writer = writer or pq.ParquetWriter(dst, tbl.schema)
            writer.write_table(tbl)
    if writer:
        writer.close()
    log("Conversion finished")

def main() -> None:
    t_global = time.time()
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    log("─ Start Data Fetch ─")
    if not URL_LIST_FILE.exists():
        logger.error("URL list missing – run 01a_url_finder.py first")
        return

    if (n := len(seen_urls())):
        logger.info(f"Resuming – {n} URLs already processed")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool, \
            open(TEMP_JSONL_FILE, "a", encoding="utf-8") as fh:

        inflight = set()
        try:
            for batch_no, urls in enumerate(url_stream(), 1):
                t_batch = time.time()
                for item in urls:
                    while len(inflight) >= MAX_INFLIGHT:
                        done, inflight = wait(
                            inflight, return_when=FIRST_COMPLETED)
                        for fut in done:
                            handle_future(fut, fh)
                    inflight.add(pool.submit(
                        fetch, item["url"], item["label"]))

                deadline = time.time() + HARD_DEADLINE
                while inflight:
                    timeout = max(0, deadline - time.time())
                    if timeout == 0:
                        break
                    done, inflight = wait(inflight, timeout=timeout,
                                          return_when=FIRST_COMPLETED)
                    for fut in done:
                        handle_future(fut, fh)

                for fut in list(inflight):
                    if not fut.done():
                        fut.cancel()
                        inflight.remove(fut)
                        logger.warning("Hard-cancelled one slow url")

                fh.flush()
                os.fsync(fh.fileno())
                gc.collect()
                log(f"Finished batch {batch_no} ({len(urls)} urls)", t_batch)
                time.sleep(DELAY_BETWEEN_BATCHES)

        except KeyboardInterrupt:
            logger.warning(
                "Interrupted by user ‑ cancelling outstanding tasks…")
            for fut in inflight:
                fut.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    if ENRICHED_DATA_FILE.exists():
        ENRICHED_DATA_FILE.unlink()
    jsonl_to_parquet(TEMP_JSONL_FILE, ENRICHED_DATA_FILE)

    import pandas as pd
    log(f"Parquet rows = {len(pd.read_parquet(ENRICHED_DATA_FILE))}")
    TEMP_JSONL_FILE.unlink(missing_ok=True)
    log("─ Done ─", t_global)

def handle_future(fut, fh):
    try:
        res = fut.result()
        fh.write(json.dumps(res, ensure_ascii=False) + "\n")
    except CancelledError:
        fh.write(json.dumps({"url": None, "label": -1,
                             "html_content": None, "text_content": None})+"\n")
    except Exception as e:
        logger.error(f"Worker raised: {e}")

if __name__ == "__main__":
    main()
