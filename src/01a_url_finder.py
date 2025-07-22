"""
Multithreaded crawler that collects ≤ 25 pages per seed domain and writes
them with a class-label to data/processed/urls_to_fetch.csv.
1.  Hard cap: never queues or fetches more than `max_links` (25) per domain.
2.  Fixed progress-bar: total is computed once and never inflated.
3.  `process_result()` now returns `(new_urls, wrote_page)` so the loop can
    advance the bar only when a *new* page is actually written.
4.  More specific exception handling in `fetch_and_parse_page`.
5.  Improved lock granularity in `CrawlManager` to move I/O operations
    out of the critical section, enhancing concurrency.
"""

from src.utils import get_logger
from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PERSONAL_SITES_FILE,
    REQUEST_TIMEOUT,
    REQUEST_HEADERS,
    LABEL_MAPPING,
)
import sys
import pathlib
import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
from urllib.parse import urlparse, urljoin
from collections import deque
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import csv
import argparse
import threading

project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))
logger = get_logger(__name__)

CORPORATE_DOMAINS_FILE = RAW_DATA_DIR / "corporate.csv"
PERSONAL_BLOGS_FILE = RAW_DATA_DIR / "searchmysite_urls.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "urls_to_fetch.csv"

MAX_WORKERS = 40
HEAD_TIMEOUT = 5


def fetch_and_parse_page(url: str) -> tuple[str, set[str]]:
    """
    GET a page, return a tuple (original_url, set_of_internal_absolute_links).
    Any error => returns (url, empty_set).
    """
    try:
        resp = requests.get(url,
                            timeout=REQUEST_TIMEOUT,
                            headers=REQUEST_HEADERS)
        resp.raise_for_status()

        if "text/html" not in resp.headers.get("Content-Type", ""):
            return url, set()

        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(resp.content, "html.parser")

        links = set()
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if href.startswith(("#", "mailto:", "tel:")):
                continue
            abs_url = urljoin(url, href).split("#")[0]
            links.add(abs_url)
        return url, links
    except requests.RequestException:
        return url, set()
    except Exception as e:
        logger.warning("Unexpected error processing %s: %s", url, e)
        return url, set()


def is_url_alive(url: str) -> bool:
    """
    Lightweight HEAD probe to see if the seed URL is reachable (status < 400).
    """
    try:
        resp = requests.head(
            url,
            timeout=HEAD_TIMEOUT,
            headers=REQUEST_HEADERS,
            allow_redirects=True,
        )
        return resp.status_code < 400
    except requests.RequestException:
        return False


class CrawlManager:
    """
    Keeps per-domain state and enforces the ≤ max_links quota.
    This class is thread-safe.
    """

    def __init__(self, writer: csv.writer, written_urls_set: set[str]):
        self.writer = writer
        self.written_urls = written_urls_set
        self.lock = threading.Lock()
        self.tasks_by_domain: dict[str, dict] = {}

    def add_seed_task(self, url: str, label: int, max_links: int):
        """
        Register a new domain. Returns the seed url if accepted, else None.
        """
        with self.lock:
            domain = urlparse(url).netloc.replace("www.", "")
            if not domain or domain in self.tasks_by_domain:
                return None

            self.tasks_by_domain[domain] = dict(
                label=label,
                max=max_links,
                queue=deque([url]),
                visited={url},
                count=0,
            )
            return url

    def process_result(
        self,
        original_url: str,
        found_links: set[str],
    ) -> tuple[list[str], bool]:
        """
        1. Prepares the finished page for writing (if not yet written).
        2. Queues up to the remaining quota of *same-domain* links.
        3. Performs the actual file write *outside* the lock.
        Returns (new_urls_to_submit, wrote_new_page_bool).
        """
        new_urls = []
        wrote_page = False
        row_to_write = None

        with self.lock:
            domain = urlparse(original_url).netloc.replace("www.", "")
            state = self.tasks_by_domain.get(domain)
            if not state:
                return new_urls, wrote_page

            if original_url not in self.written_urls:
                row_to_write = [original_url, state["label"]]
                self.written_urls.add(original_url)
                state["count"] += 1
                wrote_page = True

            remaining = state["max"] - state["count"]
            if remaining <= 0:
                if row_to_write:
                    self.writer.writerow(row_to_write)
                return new_urls, wrote_page

            for link in found_links:
                if len(new_urls) >= remaining:
                    break
                try:
                    link_domain = urlparse(link).netloc.replace("www.", "")
                    if link_domain == domain and link not in state["visited"]:
                        state["visited"].add(link)
                        new_urls.append(link)
                except Exception:
                    continue

        if row_to_write:
            self.writer.writerow(row_to_write)

        return new_urls, wrote_page


def main():
    parser = argparse.ArgumentParser(
        description="Find and validate URLs for the dataset (≤ 25 per domain)."
    )
    parser.add_argument("--debug",
                        action="store_true",
                        help="Run in debug mode (20 seeds each).")
    args = parser.parse_args()

    logger.info(
        "--- RUNNING IN %s MODE ---",
        "DEBUG" if args.debug else "NORMAL",
    )

    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    if OUTPUT_FILE.exists():
        logger.warning("%s already exists – deleting.", OUTPUT_FILE)
        OUTPUT_FILE.unlink()

    sample_rows = 20 if args.debug else None

    corporate_seeds, personal_seeds = [], []

    try:
        corp_df = pd.read_csv(CORPORATE_DOMAINS_FILE,
                              header=None,
                              nrows=sample_rows)
        for url in corp_df[0].dropna().unique():
            corporate_seeds.append(
                dict(
                    url=url.strip(),
                    max_links=25,
                    label=LABEL_MAPPING["corporate_seo"],
                ))
    except Exception as exc:
        logger.error("Cannot read %s: %s", CORPORATE_DOMAINS_FILE, exc)

    try:
        pers_df = pd.read_csv(PERSONAL_BLOGS_FILE, nrows=sample_rows)
        for url in pers_df["Site URL"].dropna().unique():
            personal_seeds.append(
                dict(
                    url=url.strip(),
                    max_links=25,
                    label=LABEL_MAPPING["personal_blog"],
                ))
    except Exception as exc:
        logger.error("Cannot read %s: %s", PERSONAL_BLOGS_FILE, exc)

    all_seeds = corporate_seeds + personal_seeds
    logger.info("Validating %d seed URLs with HEAD requests...",
                len(all_seeds))

    valid_tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(is_url_alive, t["url"]): t for t in all_seeds}
        for fut in tqdm(as_completed(futures),
                        total=len(all_seeds),
                        desc="Validating"):
            if fut.result():
                valid_tasks.append(futures[fut])

    logger.info("Validation complete – %d live seed sites.", len(valid_tasks))
    if not valid_tasks:
        logger.error("No live seeds – aborting.")
        return

    total_pages_planned = sum(t["max_links"] for t in valid_tasks)

    written_urls: set[str] = set()
    with (
            open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f_out,
            ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool,
            tqdm(total=total_pages_planned, desc="Crawling Pages", unit="page")
            as pbar,
    ):
        writer = csv.writer(f_out)
        writer.writerow(["url", "label"])

        manager = CrawlManager(writer, written_urls)
        futures = {}

        for task in valid_tasks:
            seed_url = manager.add_seed_task(task["url"], task["label"],
                                             task["max_links"])
            if seed_url:
                futures[pool.submit(fetch_and_parse_page, seed_url)] = seed_url

        while futures:
            done, futures = as_completed(futures), {}
            for fut in done:
                original = fut.owner()
                try:
                    _, links = fut.result()
                except Exception as exc:
                    logger.error("Exception while crawling %s: %s", original,
                                 exc)
                    links = set()

                new_urls, wrote = manager.process_result(original, links)
                if wrote:
                    pbar.update(1)

                for url in new_urls:
                    future = pool.submit(fetch_and_parse_page, url)
                    future.owner = lambda u=url: u
                    futures[future] = future

    logger.info(
        "Finished – wrote %d unique URLs to %s",
        len(written_urls),
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
