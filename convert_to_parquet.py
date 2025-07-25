# FILE: convert_to_parquet.py (REVISED AND SAFE VERSION)

from src.config import TEMP_JSONL_FILE, ENRICHED_DATA_FILE
import json
import pathlib
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Ensure the 'src' directory is in the path to import config
project_root = pathlib.Path(__file__).parent
sys.path.append(str(project_root))


# --- Configuration ---
# THIS IS THE CRITICAL CHANGE. REDUCED FROM 5000 to 200.
CHUNK_SIZE = 200
SRC_FILE = TEMP_JSONL_FILE
DST_FILE = ENRICHED_DATA_FILE


def convert():
    """
    Safely converts a large JSONL file to Parquet format using minimal RAM.
    """
    print(
        f"Starting conversion with SAFE CHUNK_SIZE={CHUNK_SIZE}:\n  From: {SRC_FILE}\n  To:   {DST_FILE}")

    schema = pa.schema([
        pa.field("url", pa.string()),
        pa.field("label", pa.int64()),
        pa.field("html_content", pa.string()),
        pa.field("text_content", pa.string()),
    ])

    if DST_FILE.exists():
        print(f"Warning: Deleting existing file at {DST_FILE}")
        DST_FILE.unlink()

    writer = None
    rows = []

    try:
        with open(SRC_FILE, 'r', encoding='utf-8') as f_in:
            for line in tqdm(f_in, desc="Converting to Parquet"):
                try:
                    record = json.loads(line)
                    if record.get("url"):
                        rows.append(record)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line.")
                    continue

                if len(rows) >= CHUNK_SIZE:
                    tbl = pa.Table.from_pylist(rows, schema=schema)
                    if writer is None:
                        writer = pq.ParquetWriter(DST_FILE, tbl.schema)
                    writer.write_table(tbl)
                    rows = []  # Clear the memory

            if rows:
                tbl = pa.Table.from_pylist(rows, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(DST_FILE, tbl.schema)
                writer.write_table(tbl)

    finally:
        if writer:
            writer.close()
            print("\n✅ Conversion complete!")
            print(f"Final dataset is ready at: {DST_FILE}")
        else:
            print(
                "\n❌ Error: No data was written. The source file might be empty or invalid.")


if __name__ == "__main__":
    convert()
