"""Builds notebooks/colab_full_run.ipynb: a self-contained Colab notebook (source modules embedded)."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Full-data entity matching → `matching_results.csv` + `candidates.csv` (saved to Drive)\n\n"
       "1. **Runtime → Change runtime type → High-RAM** if available (full data needs ~10-14 GB).\n"
       "2. **Runtime → Run all.** Approve the Google Drive access prompt.\n"
       "3. Outputs go to `MyDrive/ML-CHALLENGE-AMAZON/outputs/`.\n\n"
       "If Colab disconnects, run all again: the trained matcher is saved to Drive and reused, so only the "
       "test stage re-runs. Delete `outputs/matcher_lgbm.txt` to retrain."),
    code("!pip -q install polars pyarrow bm25s sparse_dot_topn jellyfish unidecode rapidfuzz lightgbm"),
    code("import os\n"
         "DATA_DIR = os.environ.get('DATA_DIR', '/content/drive/MyDrive/ML-CHALLENGE-AMAZON')\n"
         "OUT_DIR  = os.environ.get('OUT_DIR',  DATA_DIR + '/outputs')\n"
         "SEP = ','              # use '\\t' if the submission must be tab-separated\n"
         "WRITE_CANDIDATES = True  # candidates.csv is large (~1-2 GB on full data)\n"
         "try:\n"
         "    from google.colab import drive\n"
         "    drive.mount('/content/drive')\n"
         "except ImportError:\n"
         "    print('Not in Colab; using DATA_DIR =', DATA_DIR)\n"
         "print(sorted(p for p in os.listdir(DATA_DIR) if p.endswith('.tsv')))"),
    code("# Copy the TSVs to local disk: reading 2 GB through the Drive mount is much slower\n"
         "import shutil\n"
         "LOCAL = '/content/data' if os.path.exists('/content') else DATA_DIR\n"
         "if LOCAL != DATA_DIR:\n"
         "    os.makedirs(LOCAL, exist_ok=True)\n"
         "    for f in os.listdir(DATA_DIR):\n"
         "        if f.startswith(('train_', 'test_')) and f.endswith('.tsv') and not os.path.exists(f'{LOCAL}/{f}'):\n"
         "            shutil.copy(f'{DATA_DIR}/{f}', f'{LOCAL}/{f}'); print('copied', f)"),
    md("## Pipeline source (from the repo, embedded so the notebook is self-contained)"),
    code("os.makedirs('src', exist_ok=True)\nopen('src/__init__.py', 'w').close()"),
]
for f in ("fast_normalization.py", "fast_blocking.py", "match_pipeline.py"):
    cells.append(code(f"%%writefile src/{f}\n" + (ROOT / "src" / f).read_text()))
cells += [
    md("## Run: train matcher on `train_*` → predict `test_*` → write CSVs to Drive"),
    code("import sys, logging, importlib\n"
         "sys.path.insert(0, os.getcwd())\n"
         "logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', force=True)\n"
         "for m in ('src.fast_normalization', 'src.fast_blocking', 'src.match_pipeline'):\n"
         "    if m in sys.modules: importlib.reload(sys.modules[m])\n"
         "from src.match_pipeline import run_all, RunConfig\n"
         "info = run_all(LOCAL, OUT_DIR, RunConfig(), sep=SEP, write_candidates=WRITE_CANDIDATES)\n"
         "info"),
    code("import polars as pl\n"
         "res = pl.read_csv(f'{OUT_DIR}/matching_results.csv', separator=SEP)\n"
         "print(res.shape); res.head(10)"),
]
nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "None",
               "colab": {"provenance": []}}
nbf.write(nb, ROOT / "notebooks" / "colab_full_run.ipynb")
print("wrote notebooks/colab_full_run.ipynb")
