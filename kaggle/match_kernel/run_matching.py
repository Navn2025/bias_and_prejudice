# Kaggle kernel: full entity-matching run (src/match_pipeline.run_all) on the amazon-ml dataset.
# Code is fetched from GitHub at COMMIT; outputs go to /kaggle/working/outputs.
#   push:    kaggle kernels push -p kaggle/match_kernel
#   status:  kaggle kernels status navneettripathi546/amazon-ml-matching
#   output:  kaggle kernels output navneettripathi546/amazon-ml-matching -p <dir>
import glob, io, logging, os, subprocess, sys, tarfile, time, urllib.request

REPO = "Navn2025/bias_and_prejudice"
COMMIT = "00bcac3"
MAX_DF_RATIO = None          # e.g. 0.3 if India BM25 is too slow for the 12-hour limit
DATA_DIR = "/kaggle/input/amazon-ml"
OUT_DIR = "/kaggle/working/outputs"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars", "pyarrow", "bm25s",
                "sparse_dot_topn", "jellyfish", "unidecode", "rapidfuzz", "lightgbm"], check=False)

code_dir = "/kaggle/working/code"
with urllib.request.urlopen(f"https://codeload.github.com/{REPO}/tar.gz/{COMMIT}") as r:
    tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz").extractall("/kaggle/tmp_code")
os.rename(glob.glob("/kaggle/tmp_code/*")[0], code_dir)
sys.path.insert(0, code_dir)

if not os.path.exists(os.path.join(DATA_DIR, "train_source1.tsv")):
    DATA_DIR = os.path.dirname(glob.glob("/kaggle/input/**/train_source1.tsv", recursive=True)[0])

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                    format="%(asctime)s %(name)s %(message)s")
from src.match_pipeline import RunConfig, run_all  # noqa: E402

cfg = RunConfig()
cfg.blocking.max_df_ratio = MAX_DF_RATIO
print(f"commit={COMMIT} data={DATA_DIR} cpus={os.cpu_count()} cfg={cfg}", flush=True)
t = time.time()
info = run_all(DATA_DIR, OUT_DIR, cfg, sep=",", write_candidates=True)
print("RESULT", info, f"wall={(time.time() - t) / 3600:.2f}h", flush=True)
