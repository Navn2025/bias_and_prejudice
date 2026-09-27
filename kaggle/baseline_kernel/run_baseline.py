# Kaggle kernel: cheap, near-guaranteed-to-finish baseline (no BM25/LightGBM).
# Exact-match S1 -> S2/S3 on (country, clean_name), falling back to (country, phonetic_hash)
# for S1 entities that got no exact-name match. Uses src.fast_normalization only (the one
# stage confirmed fast on real data: ~220s for all 3 train sources).
#   push:    kaggle kernels push -p kaggle/baseline_kernel
#   status:  kaggle kernels status navneettripathi546/amazon-ml-baseline
#   output:  kaggle kernels output navneettripathi546/amazon-ml-baseline -p <dir>
import glob, io, logging, os, subprocess, sys, tarfile, time, urllib.request

REPO = "Navn2025/bias_and_prejudice"
COMMIT = "00bcac3"
DATA_DIR = "/kaggle/input/amazon-ml"
OUT_DIR = "/kaggle/working/outputs"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars", "pyarrow",
                "jellyfish", "unidecode"], check=False)

WORK = "/kaggle/working"
with urllib.request.urlopen(f"https://codeload.github.com/{REPO}/tar.gz/{COMMIT}") as r:
    tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    code_dir = os.path.join(WORK, tar.getnames()[0].split("/")[0])
    tar.extractall(WORK)
sys.path.insert(0, code_dir)

if not os.path.exists(os.path.join(DATA_DIR, "train_source1.tsv")):
    DATA_DIR = os.path.dirname(glob.glob("/kaggle/input/**/train_source1.tsv", recursive=True)[0])

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                    format="%(asctime)s %(name)s %(message)s")
import pandas as pd
import polars as pl
from src.fast_normalization import normalize_sources  # noqa: E402

os.makedirs(OUT_DIR, exist_ok=True)
t0 = time.time()


def norm(prefix: str):
    srcs = [pd.read_csv(os.path.join(DATA_DIR, f"{prefix}_source{i}.tsv"), sep="\t", dtype=str)
            for i in (1, 2, 3)]
    n1, n2, n3 = normalize_sources(srcs, joint_thresholds=True)
    del srcs
    cols = ["entity_id", "country", "clean_name", "phonetic_hash"]
    return (pl.from_pandas(n1[cols]).rename({"entity_id": "a"}),
            pl.from_pandas(n2[["entity_id"] + cols[1:]]).rename({"entity_id": "b"}),
            pl.from_pandas(n3[["entity_id"] + cols[1:]]).rename({"entity_id": "b"}))


def match(s1, s2, s3):
    tgt = pl.concat([s2, s3])
    exact = (s1.join(tgt, on=["country", "clean_name"], how="inner")
                .select("a", "b").unique())
    matched_a = set(exact["a"].to_list())
    rest = s1.filter(~pl.col("a").is_in(matched_a))
    phon = (rest.join(tgt, on=["country", "phonetic_hash"], how="inner")
                .select("a", "b").unique())
    pairs = pl.concat([exact, phon]).unique()
    print(f"  exact_name={exact.height} phonetic_fallback={phon.height} total_pairs={pairs.height}", flush=True)
    grouped = (pairs.group_by("a").agg(pl.col("b").unique().str.join(",")))
    base = s1.select(pl.col("a").alias("source1_entity_id"))
    res = base.join(grouped.rename({"a": "source1_entity_id", "b": "matched_entity_ids"}),
                    on="source1_entity_id", how="left").with_columns(
        pl.col("matched_entity_ids").fill_null(""))
    return res


print("=== NORMALIZE + MATCH: TRAIN (for a recall estimate only) ===", flush=True)
s1, s2, s3 = norm("train")
print(f"normalized train in {time.time() - t0:.1f}s", flush=True)
tr_res = match(s1, s2, s3)

gt = pd.read_csv(os.path.join(DATA_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str)
tp = (pl.from_pandas(gt.dropna(subset=["matched_entity_ids"]))
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .select(pl.col("source1_entity_id").alias("a"),
                pl.col("matched_entity_ids").str.strip_chars().alias("b"))
        .filter(pl.col("b") != ""))
pred_pairs = (tr_res.filter(pl.col("matched_entity_ids") != "")
                    .with_columns(pl.col("matched_entity_ids").str.split(","))
                    .explode("matched_entity_ids")
                    .select(pl.col("source1_entity_id").alias("a"),
                            pl.col("matched_entity_ids").alias("b")))
hit = tp.join(pred_pairs, on=["a", "b"], how="inner")
recall = hit.height / tp.height if tp.height else float("nan")
precision = hit.height / pred_pairs.height if pred_pairs.height else float("nan")
print(f"TRAIN baseline: recall={recall:.4f} precision={precision:.4f} "
      f"true_pairs={tp.height} pred_pairs={pred_pairs.height}", flush=True)
del s1, s2, s3, tr_res, gt, tp, pred_pairs, hit

print("=== NORMALIZE + MATCH: TEST ===", flush=True)
t1 = time.time()
s1, s2, s3 = norm("test")
print(f"normalized test in {time.time() - t1:.1f}s", flush=True)
te_res = match(s1, s2, s3)
te_res.write_csv(os.path.join(OUT_DIR, "matching_results.csv"))
n_with_match = int((te_res["matched_entity_ids"] != "").sum())
print(f"RESULT test_s1={te_res.height} s1_with_match={n_with_match} "
      f"wall={(time.time() - t0) / 60:.1f}min", flush=True)
