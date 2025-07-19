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
import time
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
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
        response.raise_for_status()

        if "text/html" not in response.headers.get("Content-Type", ""):
            return url, set()

        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(response.content, "html.parser")

        found_links = set()
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if not href or href.startswith(("#", "mailto:", "tel:")):
                continue
            absolute_url = urljoin(url, href).split("#")[0]
            found_links.add(absolute_url)
        return url, found_links
    except Exception:
        return url, set()


class CrawlManager:
    def __init__(self, writer, written_urls_set):
        self.writer = writer
        self.written_urls = written_urls_set
        self.lock = threading.Lock()
        self.tasks_by_domain = {}

    def add_seed_task(self, url: str, label: int, max_links: int):
        with self.lock:
            try:
                domain = urlparse(url).netloc.replace("www.", "")
                if not domain or domain in self.tasks_by_domain:
                    return None
            except Exception:
                return None

            self.tasks_by_domain[domain] = {
                "label": label,
                "max_links": max_links,
                "queue": deque([url]),
                "visited": {url},
                "count": 0,
            }
        return url

    def process_result(self, original_url: str, found_links: set[str]) -> list[str]:
        new_urls_to_submit = []
        with self.lock:
            try:
                domain = urlparse(original_url).netloc.replace("www.", "")
                state = self.tasks_by_domain.get(domain)

                if not state:
                    return []

                if original_url not in self.written_urls:
                    self.writer.writerow([original_url, state["label"]])
                    self.written_urls.add(original_url)
                    state["count"] += 1

                for link in found_links:
                    if state["count"] >= state["max_links"]:
                        break
                    try:
                        link_domain = urlparse(link).netloc.replace("www.", "")
                        if link_domain == domain and link not in state["visited"]:
                            state["visited"].add(link)
                            new_urls_to_submit.append(link)
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"Error processing result for {original_url}: {e}")

        return new_urls_to_submit


def is_url_alive(url: str) -> bool:
    try:
        response = requests.head(
            url, timeout=HEAD_TIMEOUT, headers=REQUEST_HEADERS, allow_redirects=True
        )
        return response.status_code < 400
    except requests.RequestException:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Find and validate URLs for the dataset."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode (20 of each class)."
    )
    args = parser.parse_args()

    if args.debug:
        logger.info("--- RUNNING IN DEBUG MODE (SMALL DATASET) ---")
    else:
        logger.info("--- Starting URL Finder with Pre-flight Validation ---")

    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    if OUTPUT_FILE.exists():
        logger.warning(f"{OUTPUT_FILE} already exists. Deleting to create a new one.")
        os.remove(OUTPUT_FILE)

    num_samples = 20 if args.debug else None
    corporate_seeds = []
    try:
        df_corp_urls = pd.read_csv(
            CORPORATE_DOMAINS_FILE, header=None, nrows=num_samples
        )
        for url in df_corp_urls[0].dropna().unique():
            corporate_seeds.append(
                {
                    "url": url.strip(),
                    "max_links": 50,
                    "label": LABEL_MAPPING["corporate_seo"],
                }
            )
    except Exception as e:
        logger.error(f"Could not load corporate file '{CORPORATE_DOMAINS_FILE}': {e}")

    personal_seeds = []
    try:
        df_pers = pd.read_csv(PERSONAL_BLOGS_FILE, nrows=num_samples)
        for url in df_pers["Site URL"].dropna().unique():
            personal_seeds.append(
                {
                    "url": url.strip(),
                    "max_links": 50,
                    "label": LABEL_MAPPING["personal_blog"],
                }
            )
    except Exception as e:
        logger.error(f"Could not load personal sites file '{PERSONAL_BLOGS_FILE}': {e}")

    all_seeds = corporate_seeds + personal_seeds
    logger.info(f"Validating {len(all_seeds)} seed URLs with HEAD requests...")
    valid_tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(is_url_alive, task["url"]): task for task in all_seeds
        }
        for future in tqdm(
            as_completed(future_to_task), total=len(all_seeds), desc="Validating URLs"
        ):
            if future.result():
                valid_tasks.append(future_to_task[future])

    logger.info(f"Validation complete. Found {len(valid_tasks)} live seed sites.")
    if not valid_tasks:
        logger.error("No live seed URLs found. Aborting.")
        return

    written_urls = set()
    with open(
        OUTPUT_FILE, "w", newline="", encoding="utf-8"
    ) as f_out, ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        writer = csv.writer(f_out)
        writer.writerow(["url", "label"])

        manager = CrawlManager(writer, written_urls)
        futures = {}

        for task in valid_tasks:
            seed_url = manager.add_seed_task(
                task["url"], task["label"], task["max_links"]
            )
            if seed_url:
                future = executor.submit(fetch_and_parse_page, seed_url)
                futures[future] = seed_url

        with tqdm(total=len(written_urls), desc="Crawling Pages", unit="page") as pbar:
            while futures:
                for future in as_completed(futures):
                    original_url = futures.pop(future)
                    try:
                        _, found_links = future.result()
                        new_tasks = manager.process_result(original_url, found_links)
                        for new_url in new_tasks:
                            new_future = executor.submit(fetch_and_parse_page, new_url)
                            futures[new_future] = new_url
                    except Exception as exc:
                        logger.error(
                            f"Task for {original_url} generated an exception: {exc}"
                        )
                        manager.process_result(original_url, set())

                    pbar.total = sum(
                        d["count"]
                        for d in manager.tasks_by_domain.values()
                        if d["count"] >= d["max_links"]
                    ) + len(futures)
                    pbar.n = len(written_urls)
                    pbar.refresh()

    url_count = len(written_urls)
    if url_count == 0:
        logger.error("No URLs were found or written. Aborting.")
    else:
        logger.info(f"Wrote {url_count} total unique URLs to {OUTPUT_FILE}")
    logger.info("--- URL Finder Complete ---")


if __name__ == "__main__":
    main()
