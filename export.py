# This is the content of your simple export.py
import joblib
import pathlib

# Note: We are running this script from a different directory,
# so we need to define the project root path explicitly.
# <-- Adjust if your path is different
PROJECT_ROOT = pathlib.Path.home() / "Desktop/code/blogspy"

MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
MODEL_ARTIFACT_PATH = MODELS_DIR / "ayush.joblib"
EXPORTED_MODEL_PATH = MODELS_DIR / "lgbm_model_for_go.txt"


def export():
    print(f"Loading model artifact from {MODEL_ARTIFACT_PATH}...")
    artifact = joblib.load(MODEL_ARTIFACT_PATH)
    booster = artifact['model']
    print(f"Exporting model to file (using lightgbm v3.3.5 from temp env)...")
    booster.save_model(str(EXPORTED_MODEL_PATH))
    print(f"✅ Model successfully exported to: {EXPORTED_MODEL_PATH}")


if __name__ == "__main__":
    export()
