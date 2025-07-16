import pandas as pd
from src.config import ENRICHED_DATA_FILE, CORPORATE_SITES_FILE, PERSONAL_SITES_FILE
from urllib.parse import urlparse

# --- Helper function to get the domain ---
def get_domain(url: str) -> str:
    try:
        # A simplified version just for counting
        return urlparse(url).netloc.replace('www.', '')
    except:
        return ""

# --- Calculate the total expected number of unique domains ---
print("--- Calculating Expected Number of Unique URLs ---")
try:
    df_corp = pd.read_csv(CORPORATE_SITES_FILE)
    df_corp['domain'] = df_corp['Domain'].apply(get_domain)

    df_pers = pd.read_csv(PERSONAL_SITES_FILE)
    df_pers['domain'] = df_pers['Site URL'].apply(get_domain)

    all_domains = pd.concat([df_corp['domain'], df_pers['domain']])
    expected_count = all_domains.nunique()
    print(f"Total unique domains in raw files: {expected_count}")
except Exception as e:
    print(f"Could not read raw files to get expected count: {e}")
    expected_count = "Unknown"


# --- Check the actual processed file ---
print("\n--- Verifying enriched_data.parquet ---")
try:
    df_enriched = pd.read_parquet(ENRICHED_DATA_FILE)
    actual_count = len(df_enriched)
    print(f"Number of rows in enriched_data.parquet: {actual_count}")

    if expected_count != "Unknown":
        if actual_count == expected_count:
            print("\n✅ SUCCESS: The number of rows matches the expected number of unique domains.")
            print("Your data file is complete.")
        else:
            print(f"\n❌ FAILED: The number of rows ({actual_count}) does NOT match the expected count ({expected_count}).")
            print("Your data file is incomplete.")

except FileNotFoundError:
    print("\n❌ FAILED: enriched_data.parquet not found.")
    print("The data preparation script did not complete.")
except Exception as e:
    print(f"\n❌ FAILED: Error reading the Parquet file: {e}")
    print("The file may be corrupt or was not written correctly.")
