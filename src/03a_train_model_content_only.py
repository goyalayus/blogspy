# src/03a_train_model_content_only.py

# --- Start of path fix ---
from src.utils import get_logger
from src.config import (
    ENRICHED_DATA_FILE, TEST_SIZE, RANDOM_STATE, MAX_FEATURES, NGRAM_RANGE,
    MODELS_DIR, REPORTS_DIR, ID_TO_LABEL, LGBM_PARAMS
)
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import sys
import pathlib
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))
# --- End of path fix ---


# Import configurations and utilities

logger = get_logger(__name__)


def main():
    """
    EXPERIMENTAL: Trains and evaluates a model using ONLY the text content (TF-IDF).
    """
    logger.info("--- Starting EXPERIMENT: Content-Only Model Training ---")

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1. Load Data
    logger.info(f"Loading data from {ENRICHED_DATA_FILE}")
    try:
        # We only need 'text_content' and 'label' for this experiment
        df = pd.read_parquet(ENRICHED_DATA_FILE, columns=[
                             'text_content', 'label'])
        df.dropna(subset=['text_content'], inplace=True)
        logger.info(f"Loaded {len(df)} usable samples.")
    except FileNotFoundError:
        logger.error(
            "Enriched data file not found. Please run your data preparation script first.")
        return

    # 2. Train-Test Split
    X = df['text_content']  # X is now just the Series of text
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        f"Data split into {len(X_train)} training and {len(X_test)} test samples.")
    del df

    # 3. Build a much simpler "Content-Only" Pipeline
    logger.info("Building content-only TF-IDF pipeline...")
    pipeline = Pipeline(steps=[
        ('vectorizer', TfidfVectorizer(
            stop_words='english',
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE
        )),
        ('classifier', LGBMClassifier(**LGBM_PARAMS))
    ])

    # 4. Train the Model
    logger.info("Fitting the content-only pipeline...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate the Model
    logger.info("Evaluating on the test set...")
    predictions = pipeline.predict(X_test)

    # Generate and save classification report with a new name
    report = classification_report(
        y_test, predictions, target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]]
    )
    report_path = REPORTS_DIR / "lgbm_content_only_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")
    print(f"\nClassification Report for Content-Only Model:\n{report}")

    # Generate and save confusion matrix plot with a new name
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test, ax=ax, cmap='Blues',
        display_labels=[ID_TO_LABEL[0], ID_TO_LABEL[1]]
    )
    ax.set_title("Confusion Matrix: Content-Only Model")
    plt.tight_layout()
    cm_path = REPORTS_DIR / "lgbm_content_only_confusion_matrix.png"
    plt.savefig(cm_path)
    logger.info(f"Confusion matrix plot saved to {cm_path}")
    plt.close(fig)

    # 6. Save the Trained Pipeline with a new name
    model_path = MODELS_DIR / "lgbm_content_only_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    logger.info(f"Trained content-only pipeline saved to {model_path}")

    # 7. Generate and Save Feature Importance Plot
    try:
        logger.info(
            "Generating feature importance plot for content-only model...")
        feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out(
        )
        importances = pipeline.named_steps['classifier'].feature_importances_

        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(25)

        fig, ax = plt.subplots(figsize=(10, 8))
        feature_importance_df.plot(
            kind='barh', x='feature', y='importance', ax=ax, legend=False)
        ax.invert_yaxis()
        ax.set_title("Top 25 Feature Importances (Content-Only Model)")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        fi_path = REPORTS_DIR / "lgbm_content_only_feature_importance.png"
        plt.savefig(fi_path)
        logger.info(f"Feature importance plot saved to {fi_path}")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Could not generate feature importance plot: {e}")

    logger.info("--- Content-Only Experiment Complete ---")


if __name__ == "__main__":
    main()
