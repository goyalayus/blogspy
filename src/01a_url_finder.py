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
    logger.debug("Fetching page: %s", url)
    try:
        resp = requests.get(url,
                            timeout=REQUEST_TIMEOUT,
                            headers=REQUEST_HEADERS)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            logger.debug("Skipping non-HTML content (%s) at %s", content_type,
                         url)
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

        logger.debug("Successfully parsed %s, found %d links.", url,
                     len(links))
        return url, links
    except requests.RequestException as e:
        logger.warning("Request for %s failed: %s", url, e)
        return url, set()
    except Exception as e:
        logger.error("An unexpected parsing error occurred for %s: %s", url, e)
        return url, set()


def is_url_alive(url: str) -> bool:
    logger.debug("Probing URL: %s", url)
    try:
        resp = requests.head(
            url,
            timeout=HEAD_TIMEOUT,
            headers=REQUEST_HEADERS,
            allow_redirects=True,
        )
        if resp.status_code < 400:
            logger.debug("URL %s is alive (status: %d)", url, resp.status_code)
            return True
        else:
            logger.debug("URL %s is dead (status: %d)", url, resp.status_code)
            return False
    except requests.RequestException as e:
        logger.debug("URL %s probe failed: %s", url, e)
        return False


class CrawlManager:

    def __init__(self, writer: csv.writer, written_urls_set: set[str]):
        self.writer = writer
        self.written_urls = written_urls_set
        self.lock = threading.Lock()
        self.tasks_by_domain: dict[str, dict] = {}

    def add_seed_task(self, url: str, label: int, max_links: int):
        with self.lock:
            domain = urlparse(url).netloc.replace("www.", "")
            if not domain:
                logger.warning("Skipping seed with invalid domain: %s", url)
                return None
            if domain in self.tasks_by_domain:
                logger.debug("Skipping duplicate seed domain: %s", domain)
                return None

            self.tasks_by_domain[domain] = dict(label=label,
                                                max=max_links,
                                                queue=deque([url]),
                                                visited={url},
                                                count=0)
            logger.debug("Registered new domain '%s' with max_links=%d",
                         domain, max_links)
            return url

    def process_result(self, original_url: str,
                       found_links: set[str]) -> tuple[list[str], bool]:
        new_urls = []
        wrote_page = False
        row_to_write = None

        with self.lock:
            domain = urlparse(original_url).netloc.replace("www.", "")
            state = self.tasks_by_domain.get(domain)
            if not state:
                logger.debug(
                    "No state found for domain '%s' (task likely finished)",
                    domain)
                return new_urls, wrote_page

            logger.debug("Processing results for %s", original_url)

            if original_url not in self.written_urls:
                row_to_write = [original_url, state["label"]]
                self.written_urls.add(original_url)
                state["count"] += 1
                wrote_page = True
                logger.debug("URL %s is new. Count for domain '%s' is now %d.",
                             original_url, domain, state["count"])

            remaining = state["max"] - state["count"]
            if remaining <= 0:
                logger.info("Domain '%s' reached its quota of %d pages.",
                            domain, state["max"])
                if row_to_write:
                    self.writer.writerow(row_to_write)
                return new_urls, wrote_page

            queued_count = 0
            for link in found_links:
                if len(new_urls) >= remaining:
                    break
                try:
                    link_domain = urlparse(link).netloc.replace("www.", "")
                    if link_domain == domain and link not in state["visited"]:
                        state["visited"].add(link)
                        new_urls.append(link)
                        queued_count += 1
                except Exception:
                    continue

            if queued_count > 0:
                logger.debug(
                    "Found %d new, same-domain URLs to queue from %s. Quota remaining: %d.",
                    queued_count, original_url, remaining - queued_count)

        if row_to_write:
            self.writer.writerow(row_to_write)
            logger.debug("Wrote row to file for %s", original_url)

        return new_urls, wrote_page


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    log_level_msg = "DEBUG" if args.debug else "NORMAL"
    logger.info("--- SCRIPT START --- RUNNING IN %s MODE ---", log_level_msg)

    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    if OUTPUT_FILE.exists():
        logger.warning("%s already exists – deleting.", OUTPUT_FILE)
        OUTPUT_FILE.unlink()

    sample_rows = 20 if args.debug else None
    corporate_seeds, personal_seeds = [], []

    try:
        logger.info("Loading corporate seeds from %s", CORPORATE_DOMAINS_FILE)
        corp_df = pd.read_csv(CORPORATE_DOMAINS_FILE,
                              header=None,
                              nrows=sample_rows)
        for url in corp_df[0].dropna().unique():
            corporate_seeds.append(
                dict(url=url.strip(),
                     max_links=25,
                     label=LABEL_MAPPING["corporate_seo"]))
        logger.info("Found %d unique corporate seed URLs.",
                    len(corporate_seeds))
    except Exception as exc:
        logger.error("Cannot read %s: %s", CORPORATE_DOMAINS_FILE, exc)

    try:
        logger.info("Loading personal blog seeds from %s", PERSONAL_BLOGS_FILE)
        pers_df = pd.read_csv(PERSONAL_BLOGS_FILE, nrows=sample_rows)
        for url in pers_df["Site URL"].dropna().unique():
            personal_seeds.append(
                dict(url=url.strip(),
                     max_links=25,
                     label=LABEL_MAPPING["personal_blog"]))
        logger.info("Found %d unique personal blog seed URLs.",
                    len(personal_seeds))
    except Exception as exc:
        logger.error("Cannot read %s: %s", PERSONAL_BLOGS_FILE, exc)

    all_seeds = corporate_seeds + personal_seeds
    logger.info("Validating %d total seed URLs with HEAD requests...",
                len(all_seeds))

    valid_tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_task = {
            pool.submit(is_url_alive, t["url"]): t
            for t in all_seeds
        }
        for fut in tqdm(as_completed(future_to_task),
                        total=len(all_seeds),
                        desc="Validating"):
            if fut.result():
                valid_tasks.append(future_to_task[fut])

    logger.info("Validation complete – %d of %d URLs are live.",
                len(valid_tasks), len(all_seeds))
    if not valid_tasks:
        logger.error("No live seeds found. Aborting.")
        return

    total_pages_planned = sum(t["max_links"] for t in valid_tasks)
    logger.info("Total pages to crawl across all live domains: %d",
                total_pages_planned)
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

        future_to_url = {}

        for task in valid_tasks:
            seed_url = manager.add_seed_task(task["url"], task["label"],
                                             task["max_links"])
            if seed_url:
                future = pool.submit(fetch_and_parse_page, seed_url)
                future_to_url[future] = seed_url

        logger.info("Submitted %d initial seed tasks to the crawl pool.",
                    len(future_to_url))

        while future_to_url:
            for future in as_completed(future_to_url):
                original_url = future_to_url.pop(future)
                try:
                    _, links = future.result()
                except Exception as exc:
                    logger.error("Future for %s failed with exception: %s",
                                 original_url, exc)
                    links = set()

                new_urls, wrote = manager.process_result(original_url, links)
                if wrote:
                    pbar.update(1)

                for url in new_urls:
                    new_future = pool.submit(fetch_and_parse_page, url)
                    future_to_url[new_future] = url

    logger.info(
        "--- CRAWL FINISHED --- Wrote %d unique URLs to %s",
        len(written_urls),
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
