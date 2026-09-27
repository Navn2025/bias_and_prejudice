"""
End-to-end: raw TSVs -> candidate pool -> learned matcher -> matching_results.csv + candidates.csv.

    train:  normalize -> candidates -> pair features -> LightGBM (labels from train ground truth,
            validation split by S1 entity; picks the probability threshold that maximizes pair F1)
    test:   normalize -> candidates -> features -> predict -> threshold -> one row per S1 entity

Memory: pairs are kept as int handles; string features are computed in chunks.
"""
from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.fast_blocking import BlockingConfig, EntityTable, generate_global_candidate_pool, \
    merge_pools, run_multi_source_geo_blocking
from src.fast_normalization import normalize_sources

log = logging.getLogger(__name__)

FEATURES = [
    "bm25_text_score", "bm25_phonetic_score", "geo_ngram_score", "target_s3",
    "text_rank", "text_rel", "phon_rank", "phon_rel", "geo_rank", "n_cands",
    "name_ratio", "name_token_set", "name_partial", "name_jw", "addr_token_set", "addr_ratio",
    "same_phonetic", "same_geo_key", "len_diff",
]


@dataclass
class RunConfig:
    blocking: BlockingConfig = field(default_factory=lambda: BlockingConfig(include_s2_s3=False, decode_ids=False))
    joint_thresholds: bool = True
    feature_chunk: int = 2_000_000
    neg_sample: float = 0.25          # fraction of negative pairs kept for training
    val_frac: float = 0.2
    seed: int = 42


def read_sources(data_dir: str | Path, prefix: str) -> list[pd.DataFrame]:
    out = []
    for i in (1, 2, 3):
        p = Path(data_dir) / f"{prefix}_source{i}.tsv"
        t = time.perf_counter()
        df = pd.read_csv(p, sep="\t", dtype=str)
        log.info("read %s: %d rows in %.1fs", p.name, len(df), time.perf_counter() - t)
        out.append(df)
    return out


@dataclass
class Candidates:
    ent: EntityTable
    pool: pl.DataFrame      # a, b (int64 handles) + 3 scores
    name: pl.Series         # clean_name per handle
    addr: pl.Series         # clean_address per handle
    s1_ids: np.ndarray


def build_candidates(sources: list[pd.DataFrame], cfg: RunConfig) -> Candidates:
    s1_ids = sources[0]["entity_id"].to_numpy(dtype=object)
    norm = normalize_sources(sources, joint_thresholds=cfg.joint_thresholds)
    sources.clear()
    gc.collect()
    ent = EntityTable.build(norm)
    name = pl.concat([pl.from_pandas(n["clean_name"].astype(str)) for n in norm])
    addr = pl.concat([pl.from_pandas(n["clean_address"].astype(str)) for n in norm])
    del norm
    gc.collect()
    bm = generate_global_candidate_pool(ent, cfg.blocking)
    geo = run_multi_source_geo_blocking(ent, cfg.blocking)
    pool = merge_pools(bm, geo)
    del bm, geo
    n = len(ent)
    pool = pool.select((pl.col("key") // n).alias("a"), (pl.col("key") % n).alias("b"),
                       "bm25_text_score", "bm25_phonetic_score", "geo_ngram_score")
    # keep only S1-anchored pairs (the ground truth / submission are S1 -> S2/S3)
    src = pl.Series(ent.source)
    pool = pool.filter(src.gather(pool["a"]) == 0)
    log.info("candidate pairs (S1-anchored): %d", pool.height)
    return Candidates(ent, pool, name, addr, s1_ids)


def _rank_features(pool: pl.DataFrame, source: np.ndarray) -> pl.DataFrame:
    tgt = pl.Series(source).gather(pool["b"])
    p = pool.with_columns((tgt == 2).cast(pl.Float32).alias("target_s3"))
    grp = ["a", "target_s3"]
    return p.with_columns(
        pl.col("bm25_text_score").rank("min", descending=True).over(grp).cast(pl.Float32).alias("text_rank"),
        (pl.col("bm25_text_score") / pl.col("bm25_text_score").max().over(grp).clip(1e-6)).alias("text_rel"),
        pl.col("bm25_phonetic_score").rank("min", descending=True).over(grp).cast(pl.Float32).alias("phon_rank"),
        (pl.col("bm25_phonetic_score") / pl.col("bm25_phonetic_score").max().over(grp).clip(1e-6)).alias("phon_rel"),
        pl.col("geo_ngram_score").rank("min", descending=True).over(grp).cast(pl.Float32).alias("geo_rank"),
        pl.len().over(grp).cast(pl.Float32).alias("n_cands"),
    )


def pair_features(c: Candidates, pool: pl.DataFrame, chunk: int) -> np.ndarray:
    """float32 matrix [len(pool), len(FEATURES)], computed in chunks to bound RAM."""
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler

    pool = _rank_features(pool, c.ent.source)
    phon, geo = c.ent.phonetic, pl.Series(c.ent.geo_key.astype(str))
    X = np.empty((pool.height, len(FEATURES)), np.float32)
    base_cols = FEATURES[:10]
    X[:, :10] = pool.select(base_cols).to_numpy().astype(np.float32)
    for s in range(0, pool.height, chunk):
        part = pool.slice(s, chunk)
        a, b = part["a"], part["b"]
        na, nb = c.name.gather(a).to_list(), c.name.gather(b).to_list()
        aa, ab = c.addr.gather(a).to_list(), c.addr.gather(b).to_list()
        e = s + part.height
        kw = dict(workers=-1, dtype=np.float32)
        X[s:e, 10] = process.cpdist(na, nb, scorer=fuzz.ratio, **kw)
        X[s:e, 11] = process.cpdist(na, nb, scorer=fuzz.token_set_ratio, **kw)
        X[s:e, 12] = process.cpdist(na, nb, scorer=fuzz.partial_ratio, **kw)
        X[s:e, 13] = process.cpdist(na, nb, scorer=JaroWinkler.normalized_similarity, **kw)
        X[s:e, 14] = process.cpdist(aa, ab, scorer=fuzz.token_set_ratio, **kw)
        X[s:e, 15] = process.cpdist(aa, ab, scorer=fuzz.ratio, **kw)
        X[s:e, 16] = (phon.gather(a) == phon.gather(b)).to_numpy().astype(np.float32)
        X[s:e, 17] = (geo.gather(a) == geo.gather(b)).to_numpy().astype(np.float32)
        X[s:e, 18] = np.abs(np.fromiter(map(len, na), np.float32, len(na)) - np.fromiter(map(len, nb), np.float32, len(nb)))
        del na, nb, aa, ab
        log.info("features %d/%d", e, pool.height)
    return X


def gt_pairs(gt: pd.DataFrame) -> pl.DataFrame:
    g = pl.from_pandas(gt[["source1_entity_id", "matched_entity_ids"]].dropna().astype(str))
    return (g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .select(pl.col("source1_entity_id").str.strip_chars().alias("A"),
                     pl.col("matched_entity_ids").str.strip_chars().alias("B"))
             .filter(pl.col("B") != "").unique())


def _labels(c: Candidates, pool: pl.DataFrame, gt: pd.DataFrame) -> tuple[np.ndarray, pl.DataFrame]:
    ids = pl.Series(c.ent.ids.astype(str))
    p = pool.select(ids.gather(pool["a"]).alias("A"), ids.gather(pool["b"]).alias("B"))
    tp = gt_pairs(gt).with_columns(pl.lit(1, pl.Int8).alias("y"))
    y = p.join(tp, on=["A", "B"], how="left", maintain_order="left")["y"].fill_null(0).to_numpy()
    return y, tp


def _best_threshold(p: np.ndarray, y: np.ndarray, n_true: int) -> tuple[float, float]:
    """Threshold maximizing pair-level F1; n_true includes true pairs the blocker missed."""
    order = np.argsort(-p)
    tp = np.cumsum(y[order])
    k = np.arange(1, len(p) + 1)
    f1 = 2 * tp / (k + n_true)
    i = int(np.argmax(f1))
    return float(p[order][i]), float(f1[i])


def train_matcher(c: Candidates, gt: pd.DataFrame, cfg: RunConfig):
    import lightgbm as lgb

    y, tp = _labels(c, c.pool, gt)
    rng = np.random.default_rng(cfg.seed)
    a = c.pool["a"].to_numpy()
    uniq_a = np.unique(a)
    val_a = rng.choice(uniq_a, int(len(uniq_a) * cfg.val_frac), replace=False)
    is_val = np.isin(a, val_a)
    keep = is_val | (y == 1) | (rng.random(len(y)) < cfg.neg_sample)
    idx = np.flatnonzero(keep)
    log.info("labels: %d positives in %d candidates (recall ceiling %.4f)", y.sum(), len(y), y.sum() / tp.height)
    X = pair_features(c, c.pool[idx], cfg.feature_chunk)
    yk, vk = y[idx], is_val[idx]
    w = np.where(yk == 1, 1.0, 1.0 / cfg.neg_sample).astype(np.float32)   # undo negative subsampling
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, seed=cfg.seed, verbose=-1)
    dtr = lgb.Dataset(X[~vk], yk[~vk], weight=w[~vk], feature_name=FEATURES)
    dva = lgb.Dataset(X[vk], yk[vk], feature_name=FEATURES, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    # threshold on the validation S1 entities; denominator = all their true pairs
    pv = model.predict(X[vk], num_iteration=model.best_iteration)
    ids = pl.Series(c.ent.ids.astype(str))
    val_ids = set(ids.gather(pl.Series(val_a)).to_list())
    n_true_val = tp.filter(pl.col("A").is_in(list(val_ids))).height
    thr, f1 = _best_threshold(pv, yk[vk], n_true_val)
    log.info("validation: best threshold %.4f -> pair F1 %.4f (best_iter %d)", thr, f1, model.best_iteration)
    return model, thr, {"val_f1": f1, "threshold": thr, "best_iter": model.best_iteration,
                        "candidate_recall": float(y.sum() / tp.height)}


def predict(c: Candidates, model, thr: float, cfg: RunConfig) -> pl.DataFrame:
    X = pair_features(c, c.pool, cfg.feature_chunk)
    p = np.empty(len(X), np.float32)
    for s in range(0, len(X), cfg.feature_chunk):
        p[s:s + cfg.feature_chunk] = model.predict(X[s:s + cfg.feature_chunk], num_iteration=model.best_iteration)
    del X
    ids = pl.Series(c.ent.ids.astype(str))
    return c.pool.select(ids.gather(c.pool["a"]).alias("entity_A"), ids.gather(c.pool["b"]).alias("entity_B"),
                         "bm25_text_score", "bm25_phonetic_score", "geo_ngram_score").with_columns(
        pl.Series("match_prob", p))


def matching_results(scored: pl.DataFrame, s1_ids: np.ndarray, thr: float) -> pl.DataFrame:
    """Exactly one row per S1 entity (same rule as generate_baseline_submission)."""
    m = (scored.filter(pl.col("match_prob") >= thr).sort("match_prob", descending=True)
               .group_by("entity_A", maintain_order=True).agg(pl.col("entity_B").unique(maintain_order=True).str.join(",")))
    base = pl.DataFrame({"source1_entity_id": pd.Series(s1_ids).astype(str).tolist()})
    return base.join(m.rename({"entity_A": "source1_entity_id", "entity_B": "matched_entity_ids"}),
                     on="source1_entity_id", how="left", maintain_order="left").with_columns(
        pl.col("matched_entity_ids").fill_null(""))


def run_all(data_dir: str | Path, out_dir: str | Path, cfg: RunConfig | None = None,
            sep: str = ",", write_candidates: bool = True) -> dict:
    """train_* -> fit matcher; test_* -> matching_results.csv (+ candidates.csv) in out_dir."""
    cfg = cfg or RunConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    import json
    import lightgbm as lgb

    model_path, meta_path = out / "matcher_lgbm.txt", out / "matcher_meta.json"
    if model_path.exists() and meta_path.exists():      # resume after a disconnect: skip training
        log.info("=== TRAIN skipped: loading %s ===", model_path)
        model = lgb.Booster(model_file=str(model_path))
        info = json.loads(meta_path.read_text())
        thr = info["threshold"]
        model.best_iteration = info["best_iter"]
    else:
        log.info("=== TRAIN ===")
        gt = pd.read_csv(Path(data_dir) / "train_ground_truth.tsv", sep="\t", dtype=str)
        tr = build_candidates(read_sources(data_dir, "train"), cfg)
        model, thr, info = train_matcher(tr, gt, cfg)
        model.save_model(str(model_path), num_iteration=model.best_iteration)
        meta_path.write_text(json.dumps(info))
        del tr, gt
        gc.collect()

    log.info("=== TEST ===")
    te = build_candidates(read_sources(data_dir, "test"), cfg)
    scored = predict(te, model, thr, cfg)
    res = matching_results(scored, te.s1_ids, thr)
    res.write_csv(out / "matching_results.csv", separator=sep)
    if write_candidates:
        scored.write_csv(out / "candidates.csv", separator=sep, float_precision=5)
    info.update(test_s1=len(te.s1_ids), test_candidates=scored.height,
                s1_with_match=int((res["matched_entity_ids"] != "").sum()),
                runtime_min=round((time.perf_counter() - t0) / 60, 1))
    log.info("done: %s", info)
    return info
