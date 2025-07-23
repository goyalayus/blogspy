"""
Single-URL prediction script for BlogSpy.
─────────────────────────────────────────
- Loads the trained model artifact (model + vectorizer).
- Fetches live HTML from a given URL.
- Performs the *exact* same feature engineering and stacking as the training script.
- Prints the classification and confidence score.
"""

import argparse
import pathlib
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
from bs4 import BeautifulSoup, SoupStrainer, XMLParsedAsHTMLWarning

project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import ID_TO_LABEL, MODELS_DIR, REQUEST_HEADERS, REQUEST_TIMEOUT
from src.feature_engineering import (
    extract_content_features,
    extract_structural_features,
    extract_url_features,
)
from src.utils import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

TARGET_TAGS = (
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "blockquote",
    "pre",
    "article",
)


def fetch_and_parse(url: str) -> dict | None:
    """Fetches and parses a URL, returning a dictionary of contents."""
    logger.info(f"Fetching content from {url}...")
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
        response.raise_for_status()
        html_content = response.text

        strainer = SoupStrainer(TARGET_TAGS)
        soup = BeautifulSoup(html_content, "lxml", parse_only=strainer)
        text_content = " ".join(
            p.get_text(strip=True) for p in soup.find_all(TARGET_TAGS)
        )

        if not text_content:
            logger.warning(
                "No text found in target tags. Prediction may be inaccurate."
            )

        return {"url": url, "html_content": html_content, "text_content": text_content}
    except requests.RequestException as e:
        logger.error(f"Failed to fetch or process {url}: {e}")
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while parsing {url}: {e}", exc_info=True
        )
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Classify a website as a corporate site or personal blog."
    )
    parser.add_argument(
        "url", help="The full URL to classify (e.g., 'https://example.com')"
    )
    args = parser.parse_args()

    model_path = MODELS_DIR / "ayush.joblib"
    if not model_path.exists():
        logger.error(
            f"Model file not found at {model_path}. Please run the training script first."
        )
        sys.exit(1)

    logger.info("Loading model artifact...")
    try:
        artifact = joblib.load(model_path)
        model = artifact["model"]
        vectorizer = artifact["vectorizer"]
    except (KeyError, TypeError):
        logger.error(
            "Artifact format is incorrect. Expected a dictionary with 'model' and 'vectorizer'. Please retrain."
        )
        sys.exit(1)
    logger.info("Model loaded successfully.")

    site_data = fetch_and_parse(args.url)
    if site_data is None:
        print(f"\n❌ Could not analyze {args.url}. Aborting.")
        return

    try:
        df = pd.DataFrame([site_data])

        text_features = vectorizer.transform(df["text_content"])

        url_features = extract_url_features(df["url"]).to_numpy(dtype="float32")
        structural_features = extract_structural_features(df["html_content"]).to_numpy(
            dtype="float32"
        )
        content_features = extract_content_features(df["text_content"]).to_numpy(
            dtype="float32"
        )

        features = sp.hstack(
            [
                text_features,
                sp.csr_matrix(url_features),
                sp.csr_matrix(structural_features),
                sp.csr_matrix(content_features),
            ],
            format="csr",
        )

        probabilities = model.predict(features)
        prediction_id = (probabilities > 0.5).astype(int)[0]
        confidence = probabilities[0] if prediction_id == 1 else 1 - probabilities[0]
        label = ID_TO_LABEL[prediction_id]

        print("-" * 50)
        print(f"✅ Results for: {args.url}")
        print(f"   Prediction: {label.replace('_', ' ').title()}")
        print(f"   Confidence: {confidence:.2%}")
        print("-" * 50)

    except Exception as e:
        logger.error(
            f"An error occurred during prediction for {args.url}: {e}", exc_info=True
        )
        print(f"\n❌ Failed to generate a prediction for {args.url}.")


if __name__ == "__main__":
    main()
