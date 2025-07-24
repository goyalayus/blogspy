import pandas as pd
import pathlib
import os

CORPORATE_FILE_PATH = pathlib.Path("data") / "raw" / "corporate.csv"


def standardize_file(file_path: pathlib.Path):
    """
    Reads a single-column CSV of domains, prepends 'https://' to each,
    and overwrites the file with the standardized URLs.
    """
    if not file_path.exists():
        print(f"Error: File not found at '{file_path}'. Cannot standardize.")
        return

    print(f"Standardizing URLs in: {file_path}")

    df = pd.read_csv(file_path, header=None, names=["domain_or_url"])

    df["url"] = df["domain_or_url"].apply(
        lambda x: f"https://{x}" if not str(x).startswith("http") else str(x)
    )

    df[["url"]].to_csv(file_path, header=False, index=False)

    print(
        f"✅ Standardization complete. '{os.path.basename(file_path)}' now contains full URLs."
    )


if __name__ == "__main__":
    standardize_file(CORPORATE_FILE_PATH)
