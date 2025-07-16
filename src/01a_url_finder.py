# src/01a_url_finder.py

from src.utils import get_logger
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, PERSONAL_SITES_FILE, REQUEST_TIMEOUT, REQUEST_HEADERS, LABEL_MAPPING
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

# --- Path and Logger Setup ---
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

# Correctly define the path to your corporate CSV

logger = get_logger(__name__)

# --- Correctly Define File Paths ---
CORPORATE_DOMAINS_FILE = RAW_DATA_DIR / 'corporate.csv'
# Using the correct file from digest
PERSONAL_BLOGS_FILE = RAW_DATA_DIR / 'searchmysite_urls.csv'

# --- Configuration ---
OUTPUT_FILE = PROCESSED_DATA_DIR / "urls_to_fetch.csv"
MAX_WORKERS = 40
CRAWL_DELAY = 0.1
HEAD_TIMEOUT = 5


def is_url_alive(url: str) -> bool:
    """Performs a lightweight HEAD request to check if a URL is responsive."""
    try:
        response = requests.head(
            url, timeout=HEAD_TIMEOUT, headers=REQUEST_HEADERS, allow_redirects=True)
        return response.status_code < 400
    except requests.RequestException:
        return False


def crawl_site(seed_url: str, max_links: int) -> set:
    """Crawls a single domain to find unique internal links."""
    found_urls = {seed_url}
    queue = deque([seed_url])
    visited = {seed_url}

    try:
        base_domain = urlparse(seed_url).netloc.replace('www.', '')
    except Exception:
        return set()

    while queue and len(found_urls) < max_links:
        current_url = queue.popleft()
        try:
            time.sleep(CRAWL_DELAY)
            response = requests.get(
                current_url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
            response.raise_for_status()

            if 'text/html' not in response.headers.get('Content-Type', ''):
                continue

            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(response.content, 'html.parser')

            for link in soup.find_all('a', href=True):
                href = link['href']
                if not href or href.startswith(('#', 'mailto:', 'tel:')):
                    continue

                absolute_url = urljoin(current_url, href).split('#')[0]

                if urlparse(absolute_url).scheme in ['http', 'https'] and \
                   urlparse(absolute_url).netloc.replace('www.', '') == base_domain and \
                   absolute_url not in visited:

                    visited.add(absolute_url)
                    found_urls.add(absolute_url)
                    queue.append(absolute_url)
                    if len(found_urls) >= max_links:
                        break
        except Exception:
            continue

    return found_urls


def run_crawl_phase(tasks, desc: str, writer, written_urls):
    """A generic function to run a concurrent crawling phase."""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(
            crawl_site, task['url'], task['max_links']): task for task in tasks}

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc=desc):
            task = future_to_task[future]
            try:
                urls_found = future.result()
                for url in urls_found:
                    if url not in written_urls:
                        writer.writerow([url, task['label']])
                        written_urls.add(url)
            except Exception as exc:
                logger.error(
                    f"Task for {task['url']} generated an exception: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Find and validate URLs for the dataset.")
    parser.add_argument('--debug', action='store_true',
                        help="Run in debug mode on a small subset of data (20 of each class).")
    args = parser.parse_args()

    if args.debug:
        logger.info("--- RUNNING IN DEBUG MODE (SMALL DATASET) ---")
    else:
        logger.info("--- Starting URL Finder with Pre-flight Validation ---")

    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    if OUTPUT_FILE.exists():
        logger.warning(
            f"{OUTPUT_FILE} already exists. Deleting to create a new one.")
        os.remove(OUTPUT_FILE)

    num_samples = 20 if args.debug else None

    corporate_seeds = []
    try:
        # The corporate file now contains full URLs, so we read them directly.
        df_corp_urls = pd.read_csv(
            CORPORATE_DOMAINS_FILE, header=None, nrows=num_samples)

        # Select the first (and only) column which now contains full URLs
        url_series = df_corp_urls[0]

        for url in url_series.dropna().unique():
            # --- MODIFICATION: Increased max_links to 50 ---
            corporate_seeds.append({'url': url.strip(),
                                   'max_links': 50, 'label': LABEL_MAPPING['corporate_seo']})
    except Exception as e:
        logger.error(
            f"Could not load corporate file '{CORPORATE_DOMAINS_FILE}': {e}")

    personal_seeds = []
    try:
        df_pers = pd.read_csv(PERSONAL_BLOGS_FILE, nrows=num_samples)
        for url in df_pers['Site URL'].dropna().unique():
            # --- MODIFICATION: Increased max_links to 50 ---
            personal_seeds.append(
                {'url': url.strip(), 'max_links': 50, 'label': LABEL_MAPPING['personal_blog']})
    except Exception as e:
        logger.error(
            f"Could not load personal sites file '{PERSONAL_BLOGS_FILE}': {e}")

    all_seeds = corporate_seeds + personal_seeds
    logger.info(f"Validating {len(all_seeds)} seed URLs with HEAD requests...")
    valid_tasks = []
    # (The rest of the script is correct and remains the same)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(
            is_url_alive, task['url']): task for task in all_seeds}
        for future in tqdm(as_completed(future_to_task), total=len(all_seeds), desc="Validating URLs"):
            if future.result():
                valid_tasks.append(future_to_task[future])

    valid_corporate_tasks = [
        t for t in valid_tasks if t['label'] == LABEL_MAPPING['corporate_seo']]
    valid_personal_tasks = [
        t for t in valid_tasks if t['label'] == LABEL_MAPPING['personal_blog']]

    logger.info(
        f"Validation complete. Found {len(valid_corporate_tasks)} live corporate sites and {len(valid_personal_tasks)} live personal sites.")

    written_urls = set()
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['url', 'label'])

        if valid_corporate_tasks:
            logger.info(
                f"Starting to crawl {len(valid_corporate_tasks)} live corporate sites...")
            run_crawl_phase(valid_corporate_tasks,
                            "Crawling Corporate", writer, written_urls)

        if valid_personal_tasks:
            logger.info(
                f"Starting to crawl {len(valid_personal_tasks)} live personal sites...")
            run_crawl_phase(valid_personal_tasks,
                            "Crawling Personal", writer, written_urls)

    url_count = len(written_urls)
    if url_count == 0:
        logger.error("No URLs were found or written. Aborting.")
    else:
        logger.info(f"Wrote {url_count} total unique URLs to {OUTPUT_FILE}")

    logger.info("--- URL Finder Complete ---")


if __name__ == "__main__":
    main()
