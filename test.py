import pandas as pd
import pathlib
import os

# --- Configuration ---
# Define the path to the file that needs to be updated.
CORPORATE_FILE_PATH = pathlib.Path(
    __file__).parent / "data" / "raw" / "corporate.csv"


def standardize_file(file_path: pathlib.Path):
    """
    Reads a single-column CSV of domains, prepends 'https://' to each,
    and overwrites the file with the standardized URLs.
    """
    if not file_path.exists():
        print(f"Error: File not found at '{file_path}'. Cannot standardize.")
        return

    print(f"Standardizing URLs in: {file_path}")

    # Read the file, which is assumed to have no header.
    df = pd.read_csv(file_path, header=None, names=['domain_or_url'])

    # Find rows that do not already start with http and prepend 'https://'
    # This makes the script safe to run multiple times.
    df['url'] = df['domain_or_url'].apply(
        lambda x: f"https://{x}" if not str(x).startswith('http') else str(x)
    )

    # Overwrite the original file with only the standardized URL column.
    # This ensures a clean, single-column CSV with no header.
    df[['url']].to_csv(file_path, header=False, index=False)

    print(
        f"✅ Standardization complete. '{os.path.basename(file_path)}' now contains full URLs.")


# --- Run the main function ---
if __name__ == "__main__":
    standardize_file(CORPORATE_FILE_PATH)
