# BlogSpy: Training Pipeline

This guide provides the exact steps to set up the environment and run the full training pipeline from scratch.

### 1. Project Setup

First, create a virtual environment and install the required Python packages.

```bash
# From the project's root directory (blogspy)

# Create the virtual environment
python3 -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

### 2. Data Preparation and Model Training

Run the following scripts in order. Some steps, especially data fetching, may take several hours to complete.

```bash
# Make sure your virtual environment is active before running!

# Step 1: Prepare the initial corporate domain list
# Reads data/raw/ranked_domains.csv -> Creates data/raw/corporate.csv
python data/raw/test.py

# Step 2: Crawl seed sites to find a large list of URLs
# This is a long-running step.
# Reads corporate.csv & searchmysite_urls.csv -> Creates data/processed/urls_to_fetch.csv
python -m src.01a_url_finder

# Step 3: Fetch the content for every URL and create the master dataset
# This is the longest-running step and will take many hours.
# Reads urls_to_fetch.csv -> Creates data/processed/enriched_data.parquet
python -m src.01_prepare_data

python verify_data.py

# Step 4: Train the final model on the prepared data
# Reads enriched_data.parquet -> Creates outputs/models/ayush.joblib
python -m src.03_train_model
```

### 3. Optional Steps

After training, you can use these utility scripts.

```bash
# Verify that the data preparation was successful and the dataset is complete
python verify_data.py

# Use the trained model to classify a new URL
python -m src.predict "https://www.some-blog-to-classify.com/"

# Use one of the experimental models for prediction
python -m src.predict --model content_only "https://www.some-blog-to-classify.com/"
```
