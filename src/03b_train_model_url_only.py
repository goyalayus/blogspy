# src/03b_train_model_url_only.py

from src.utils import get_logger
from src.config import TEST_SIZE, RANDOM_STATE, MODELS_DIR, REPORTS_DIR, ID_TO_LABEL, PROCESSED_DATA_DIR
import sys
import pathlib
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --- Path and Logger Setup ---
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))


logger = get_logger(__name__)

# --- Configuration for URL-only Model ---
URL_NGRAM_RANGE = (3, 5)
URL_MAX_FEATURES = 10000

# --- Correct Data Source for This Experiment ---
URL_DATA_FILE = PROCESSED_DATA_DIR / "urls_to_fetch.csv"


def show_most_informative_features(vectorizer, clf, n=20):
    """
    Prints the most and least informative features learned by a linear model.
    """
    logger.info(f"--- Top {n} Informative Features ---")
    try:
        feature_names = vectorizer.get_feature_names_out()
        coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))

        # Top features for "personal_blog" (class 1)
        top_positive = coefs_with_fns[:-(n + 1):-1]
        print("\n-- Top features for PERSONAL_BLOG --")
        for coef, fn in top_positive:
            print(f"{fn:<20} Coef: {coef:.3f}")

        # Top features for "corporate_seo" (class 0)
        top_negative = coefs_with_fns[:n]
        print("\n-- Top features for CORPORATE_SEO --")
        for coef, fn in top_negative:
            print(f"{fn:<20} Coef: {coef:.3f}")
    except Exception as e:
        logger.warning(f"Could not show informative features: {e}")


def main():
    """
    EXPERIMENTAL: Trains a Logistic Regression model using ONLY the URL string.
    """
    logger.info(
        "--- EXPERIMENT: URL-Only Model Training (Logistic Regression) ---")

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1. Load Data directly from the URL list
    logger.info(f"Loading data from {URL_DATA_FILE}")
    try:
        # We only need 'url' and 'label' columns
        df = pd.read_csv(URL_DATA_FILE, usecols=['url', 'label'])
        df.dropna(inplace=True)
        logger.info(f"Loaded {len(df)} usable samples.")
    except FileNotFoundError:
        logger.error(
            f"{URL_DATA_FILE} not found. Please run the URL finder script (01a_url_finder.py) first.")
        return
    except Exception as e:
        logger.error(f"Failed to load data from {URL_DATA_FILE}: {e}")
        return

    # 2. Train-Test Split
    X = df['url']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        f"Data split into {len(X_train)} training and {len(X_test)} test samples.")
    del df

    # 3. Build the URL-Only Pipeline
    logger.info("Building URL-only character n-gram pipeline...")
    pipeline = Pipeline(steps=[
        ('vectorizer', TfidfVectorizer(
            analyzer='char',
            ngram_range=URL_NGRAM_RANGE,
            max_features=URL_MAX_FEATURES
        )),
        ('classifier', LogisticRegression(solver='liblinear',
         random_state=RANDOM_STATE, max_iter=1000))
    ])

    # 4. Train the Model
    logger.info("Fitting the URL-only pipeline...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate the Model
    logger.info("Evaluating on the test set...")
    predictions = pipeline.predict(X_test)

    report = classification_report(
        y_test, predictions, target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]]
    )
    report_path = REPORTS_DIR / "url_only_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")
    print(f"\nClassification Report for URL-Only Model:\n{report}")

    # 6. Show the most important features
    show_most_informative_features(
        pipeline.named_steps['vectorizer'],
        pipeline.named_steps['classifier']
    )

    # 7. Save the Trained Pipeline
    model_path = MODELS_DIR / "url_only_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    logger.info(f"Trained URL-only pipeline saved to {model_path}")

    logger.info("--- URL-Only Experiment Complete ---")


if __name__ == "__main__":
    main()
