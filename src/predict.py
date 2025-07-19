from src.utils import get_logger
from src.config import MODELS_DIR, ID_TO_LABEL, REQUEST_TIMEOUT, REQUEST_HEADERS
from src.feature_engineering import extract_url_features, extract_structural_features, extract_content_features
import sys
import pathlib
import argparse
import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
import scipy.sparse as sp
import numpy as np

project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))
logger = get_logger(__name__)

def print_debug_info(vector_name: str, features, num_to_print=20):
    """Prints detailed debug information for a feature vector."""
    is_sparse = hasattr(features, "toarray")
    if is_sparse:
        features_dense = features.toarray().flatten()
    else:
        features_dense = np.array(features).flatten()

    print(f"\n--- DEBUG: {vector_name} (Python) ---")
    print(f"Shape: {features_dense.shape}")
    print(f"Sum: {np.sum(features_dense):.6f}")
    print(f"Non-zero count: {np.count_nonzero(features_dense)}")

    if is_sparse and "HashingVectorizer" in vector_name:
        print(
            f"Sum of Squares (L2 Norm Check): {np.sum(features_dense**2):.6f}")

    non_zero_indices = np.nonzero(features_dense)[0]
    print(f"First ~{num_to_print} non-zero indices and values:")
    for i in range(min(num_to_print, len(non_zero_indices))):
        idx = non_zero_indices[i]
        val = features_dense[idx]
        print(f"  Index: {idx:<5}, Value: {val:.6f}")
    print(f"-----------------------------------")

def fetch_and_parse(url: str) -> dict | None:
    """ Fetches and parses a URL, returning a dictionary of contents. """
    logger.info(f"Fetching content from {url}...")
    try:
        response = requests.get(
            url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
        response.raise_for_status()
        html_content = response.text
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html_content, 'lxml')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        text_content = ' '.join(soup.stripped_strings)
        return {"url": url, "html_content": html_content, "text_content": text_content}
    except requests.RequestException as e:
        logger.error(f"Failed to fetch or process {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while parsing {url}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Classify websites.")
    parser.add_argument('urls', nargs='+',
                        help="One or more full URLs to classify")
    parser.add_argument('--model', type=str, default='main', choices=[
                        'main', 'content_only', 'url_only'], help="The model to use: 'main', 'content_only', or 'url_only'.")
    args = parser.parse_args()

    model_name_map = {
        'main': 'ayush.joblib',
        'content_only': 'lgbm_content_only_pipeline.joblib',
        'url_only': 'url_only_pipeline.joblib'
    }
    model_filename = model_name_map[args.model]
    model_path = MODELS_DIR / model_filename

    if not model_path.exists():
        logger.error(
            f"Model file not found at {model_path}. Please train it first.")
        sys.exit(1)

    logger.info(f"Loading model: {model_filename}...")
    artifact = joblib.load(model_path)
    logger.info("Model loaded successfully.")

    for url in args.urls:
        print("-" * 50)
        logger.info(f"Analyzing URL: {url}")

        if args.model == 'url_only':
            site_data = {"url": url}
        else:
            site_data = fetch_and_parse(url)
            if site_data is None:
                print(f"  ❌ Could not analyze {url}. Skipping.")
                continue

        try:
            if args.model == 'main':
                vectorizer = artifact['vectorizer']
                model = artifact['model']

                df = pd.DataFrame([site_data])

                txt_features = vectorizer.transform(
                    df["text_content"].fillna(""))
                print_debug_info(
                    "Text Features (HashingVectorizer)", txt_features)

                url_features = extract_url_features(
                    df["url"]).to_numpy(dtype="float32")
                structural_features = extract_structural_features(
                    df["html_content"]).to_numpy(dtype="float32")
                content_features = extract_content_features(
                    df["text_content"]).to_numpy(dtype="float32")

                print_debug_info("URL Features", url_features)
                print_debug_info("Structural Features", structural_features)
                print_debug_info("Content Features", content_features)

                features = sp.hstack([
                    txt_features,
                    sp.csr_matrix(url_features),
                    sp.csr_matrix(structural_features),
                    sp.csr_matrix(content_features)
                ], format="csr")

                probabilities = model.predict(features)
                prediction_id = (probabilities > 0.5).astype(int)[0]
                confidence = probabilities[0] if prediction_id == 1 else 1 - \
                    probabilities[0]

            else:
                pipeline = artifact
                predict_input = [site_data['url']] if args.model == 'url_only' else [
                    site_data['text_content']]
                prediction_id = pipeline.predict(predict_input)[0]
                probabilities = pipeline.predict_proba(predict_input)[0]
                confidence = probabilities[prediction_id]

            label = ID_TO_LABEL[prediction_id]
            print(f"\n✅ Results for: {url}")
            print(f"  Prediction: {label.upper()}")
            print(f"  Confidence: {confidence:.2%}")

        except Exception as e:
            logger.error(
                f"An error occurred during prediction for {url}: {e}", exc_info=True)
            print(f"  ❌ Failed to predict for {url}.")

    print("-" * 50)


if __name__ == "__main__":
    main()
