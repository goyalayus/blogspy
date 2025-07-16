import csv
import tldextract
import os

# --- Configuration ---
INPUT_FILE = 'ranked_domains.csv'
# The output file will now have a .csv extension
OUTPUT_FILE = 'corporate.csv'


def clean_and_deduplicate_domains(input_path, output_path):
    """
    Reads a mixed-format file of domains and URLs, extracts the root domain,
    de-duplicates them with a preference for the .com version, and saves the
    result to a single-column CSV file.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    # This dictionary will store the preferred domain for each base name.
    # Key: 'google', Value: 'google.com'
    # This handles the preference for .com TLDs.
    domain_groups = {}

    print(f"Processing file: {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            raw_domain = ""

            # --- Heuristics to parse the mixed-format file ---

            # 1. Check if it's a CSV line from your file
            if line.startswith('"') and '","' in line:
                try:
                    reader = csv.reader([line])
                    row = next(reader)
                    if len(row) > 1:
                        raw_domain = row[1]
                except (csv.Error, IndexError):
                    continue
            # 2. Check if it's a full URL
            elif line.startswith('http'):
                raw_domain = tldextract.extract(line).fqdn
            # 3. Assume it's a plain domain
            else:
                raw_domain = line

            if not raw_domain:
                continue

            # --- Normalize and filter the extracted domain ---

            ext = tldextract.extract(raw_domain)
            if not ext.domain:
                continue

            base_name = ext.domain
            registrable_domain = ext.registered_domain

            # --- Apply the .com preference logic ---

            if base_name not in domain_groups:
                domain_groups[base_name] = registrable_domain
            else:
                if registrable_domain.endswith('.com'):
                    domain_groups[base_name] = registrable_domain

    # Extract the final list of domains from our dictionary values
    final_domains = sorted(list(domain_groups.values()))

    # --- MODIFIED SECTION: Write the clean list to a CSV file ---
    with open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        # Create a CSV writer object
        writer = csv.writer(f_out)

        # Write the header row
        writer.writerow(['Domain'])

        # Write the data rows
        for domain in final_domains:
            # writerow expects a list, so we wrap the domain in one
            writer.writerow([domain])

    print("-" * 30)
    print("Processing complete!")
    print(f"Found and processed {len(final_domains)} unique domains.")
    print(f"Clean CSV file saved to: {output_path}")


# --- Run the main function ---
if __name__ == "__main__":
    clean_and_deduplicate_domains(INPUT_FILE, OUTPUT_FILE)
