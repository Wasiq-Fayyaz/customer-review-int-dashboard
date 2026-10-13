import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")

STAGES = [
    "stage1_preprocessing.py",
    "stage2_linguistic.py",
    "stage3_vectorization.py",
    "stage4_analysis.py",
    "stage5_classification.py",
    "stage6_dashboard_data.py",
]


def setup():
    import nltk
    for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                'averaged_perceptron_tagger_eng']:
        nltk.download(pkg)


def run(cmd, cwd):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def pipeline(skip_preprocessing=False):
    # Stage scripts use relative file paths, so they run from inside data/
    for stage in STAGES:
        if skip_preprocessing and stage in STAGES[:3]:
            continue
        run([sys.executable, stage], DATA)
    run([sys.executable, "report.py"], ROOT)


def serve():
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, app_dir=ROOT)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "help"
    if command == "setup":
        setup()
    elif command == "pipeline":
        # `--skip-prep` reuses the existing cleaned/linguistic CSVs and skips stages 1-3
        pipeline(skip_preprocessing="--skip-prep" in sys.argv)
    elif command == "serve":
        serve()
    else:
        print("Usage:\n"
              "  python main.py setup                    download NLTK data\n"
              "  python main.py pipeline [--skip-prep]   run all stages + PDF report\n"
              "  python main.py serve                    start API + dashboard at http://127.0.0.1:8000")
