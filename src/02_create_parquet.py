# src/02_create_parquet.py

from src.utils import get_logger
from src.config import ENRICHED_DATA_FILE, TEMP_JSONL_FILE, PROCESSED_DATA_DIR
import sys
import pathlib
import json
import os
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import psutil

# --- Path and Logger Setup ---
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))


logger = get_logger(__name__)


def log_memory_usage(pbar: tqdm | None = None):
    """Logs current memory usage and updates a tqdm progress bar if provided."""
    process = psutil.Process(os.getpid())
    mem_usage_mb = process.memory_info().rss / (1024 * 1024)
    if pbar:
        pbar.set_postfix_str(f"Mem: {mem_usage_mb:.0f} MB")
    else:
        logger.info(f"MEMORY USAGE: {mem_usage_mb:.2f} MB")


def convert_jsonl_to_parquet(jsonl_path: pathlib.Path, parquet_path: pathlib.Path, chunk_size: int = 200):
    """
    Reads a large JSONL file and writes to a Parquet file using a VERY SMALL chunk size
    to guarantee low memory usage.
    """
    if not jsonl_path.exists():
        logger.error(f"Input file not found: {jsonl_path}")
        return

    logger.info(
        f"Starting memory-safe conversion with chunk_size={chunk_size}...")
    log_memory_usage()

    schema = pa.schema([
        pa.field('url', pa.string()),
        pa.field('label', pa.int64()),
        pa.field('html_content', pa.string()),
        pa.field('text_content', pa.string())
    ])

    writer = None
    pbar = None
    try:
        # Get total lines for tqdm progress bar
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        with open(jsonl_path, 'r', encoding='utf-8') as f_in:
            chunk = []
            pbar = tqdm(f_in, total=total_lines,
                        desc="Converting to Parquet", unit=" lines")
            for line in pbar:
                try:
                    data = json.loads(line)
                    data.pop('error', None)
                    chunk.append(data)
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Skipping malformed line.")
                    continue

                if len(chunk) >= chunk_size:
                    table = pa.Table.from_pylist(chunk, schema=schema)
                    if writer is None:
                        writer = pq.ParquetWriter(parquet_path, table.schema)
                    writer.write_table(table)
                    chunk = []

                # Log memory every 500 lines
                if pbar.n % 500 == 0:
                    log_memory_usage(pbar)

            if chunk:
                table = pa.Table.from_pylist(chunk, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(parquet_path, table.schema)
                writer.write_table(table)
    finally:
        if pbar:
            pbar.close()
        if writer:
            writer.close()

    logger.info("Conversion complete.")


def main():
    """
    Main function to run the conversion.
    """
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)

    if ENRICHED_DATA_FILE.exists():
        logger.warning(f"Deleting existing Parquet file: {ENRICHED_DATA_FILE}")
        os.remove(ENRICHED_DATA_FILE)

    convert_jsonl_to_parquet(TEMP_JSONL_FILE, ENRICHED_DATA_FILE)

    if ENRICHED_DATA_FILE.exists():
        logger.info(f"Verification successful: {ENRICHED_DATA_FILE} created.")
    else:
        logger.error("Parquet file creation failed.")


if __name__ == "__main__":
    main()
