"""
evaluate.py — offline eval harness for BookSearch-FA (README §3.3, §10).

Questions live in eval/questions.jsonl; gold is resolved at RUN TIME from
quote + printed page (±1), never from chunk ids, so re-chunking doesn't break
labels (README §10). Every variant runs the SAME questions; per-question
metric vectors are kept so paired permutation tests can compare variants.

Usage:
  python evaluate.py --variants base,bm25,dense,w_dense_2,cc [--split dev]
  python evaluate.py --variants base --abstain-sweep   # tune min_dense_sim on the "none" slice
Results are appended (append-only) to eval/results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import search
from search import Hit, Searcher, load_config, normalize_search, tokenize_bm25, valid_hit

_WS_RE = re.compile(r"\s+")
DEFAULT_KS = (1, 5, 10, 30, 100)


# ----------------------------------------------------------------------------
# questions
# ----------------------------------------------------------------------------
@dataclass
class Witness:
    quote: str
    page: int | None = None
    grade: int = 1


@dataclass
class Question:
    id: str
    slice: str
    split: str
    question: str
    witnesses: list[Witness] = field(default_factory=list)
    note: str = ""


def load_questions(path: str | Path = "eval/questions.jsonl") -> list[Question]:
    qs: list[Question] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = json.loads(line)
            qs.append(Question(
                id=d["id"], slice=d["slice"], split=d.get("split", "test"),
                question=d["question"],
                witnesses=[Witness(w["quote"], w.get("page"), int(w.get("grade", 1)))
                           for w in d.get("witnesses", [])],
                note=d.get("note", "")))
    return qs


# ----------------------------------------------------------------------------
# gold: qid -> {row: grade}  (quote match inside declared page ± gap)
# ----------------------------------------------------------------------------
def _norm(text: str) -> str:
    """normalize_search + collapse ALL whitespace (chunk texts contain newlines
    a copied quote spans)."""
    return _WS_RE.sub(" ", normalize_search(text)).strip()


def _page_gap(chunk: dict, page: int) -> int:
    try:
        lo, hi = int(str(chunk["l0"])), int(str(chunk["l1"]))
    except (TypeError, ValueError):
        return 1 << 30                       # non-numeric label -> never matches
    if lo - 1 <= page <= hi + 1:
        return 0
    return min(abs(page - lo), abs(page - hi))


def resolve_gold(qs: list[Question], searcher: Searcher, gap: int = 1) -> dict[str, dict[int, int]]:
    """A witness matches when its normalized quote occurs in the chunk text AND
    the chunk's printed page is within `gap` of the declared page (README §10:
    gold بر اساس quote + صفحه، نه chunk_id)."""
    chunks = searcher._chunks
    texts = [_norm(c["text"]) for c in chunks]          # cache per run
    out: dict[str, dict[int, int]] = {}
    for q in qs:
        gold: dict[int, int] = {}
        for w in q.witnesses:
            qn = _norm(w.quote)
            if not qn:
                continue
            for r, t in enumerate(texts):
                if qn not in t:
                    continue
                if w.page is not None and _page_gap(chunks[r], w.page) > gap:
                    continue
                gold[r] = max(gold.get(r, 0), w.grade)
        out[q.id] = gold
    return out


# ----------------------------------------------------------------------------
# variant runner
# ----------------------------------------------------------------------------
@dataclass
class RunRow:
    rows: list[int]          # fused order, best first
    found: bool
    context: str             # final passages text — exactly what the LLM would see
    timings_ms: dict


def run_variant(searcher: Searcher, qs: list[Question],
                overrides: dict | None = None) -> dict[str, RunRow]:
    ov = dict(overrides or {})
    if ov.get("fusion") == "cc":
        return _run_cc(searcher, qs, alpha=float(ov.get("alpha", 0.5)),
                       depth=int(ov.get("depth", searcher.cfg.search.depth)),
                       fused_top=int(ov.get("fused_top", searcher.cfg.search.fused_top)),
                       beta=float(ov.get("beta", 0.0)))
    saved_nb = searcher.cfg.search.neighbors
    if "neighbors" in ov:                     # query-time variant (README §10)
        searcher.cfg.search.neighbors = int(ov["neighbors"])
    try:
        out: dict[str, RunRow] = {}
        for q in qs:
            kw = {k: v for k, v in ov.items() if k != "neighbors"}
            res = searcher.search(q.question, overrides=kw)   # retrieve(overrides=…)
            out[q.id] = RunRow(rows=[h.row for h in res.hits], found=res.found,
                               context="\n".join(p.text for p in res.passages),
                               timings_ms=dict(res.timings_ms))
        return out
    finally:
        searcher.cfg.search.neighbors = saved_nb


def _minmax(ids: np.ndarray, sc: np.ndarray) -> dict[int, float]:
    if len(ids) == 0:
        return {}
    lo, hi = float(sc.min()), float(sc.max())
    if hi - lo < 1e-9:
        return {int(r): 1.0 for r in ids}
    return {int(r): float((s - lo) / (hi - lo)) for r, s in zip(ids, sc)}


def _run_cc(searcher: Searcher, qs: list[Question], alpha: float,
            depth: int, fused_top: int, beta: float = 0.0) -> dict[str, RunRow]:
    """§8 CC: α·minmax(dense) + (1−α)·minmax(bm25) روی همان top-N اجتماع.
    بدون dense، به minmax(bm25) تنزل می‌کند (BM25-only index)."""
    n = len(searcher._chunks)
    min_sim = searcher.cfg.search.min_dense_sim
    out: dict[str, RunRow] = {}
    for q in qs:
        t0 = time.perf_counter()
        toks = tokenize_bm25(q.question)
        vocab = getattr(searcher._bm25, "vocab_dict", None)
        toks = [t for t in toks if vocab is None or t in vocab]
        bm_ids = np.empty(0, np.int64)
        bm_sc = np.empty(0, np.float32)
        if toks:
            res, sc = searcher._bm25.retrieve([toks], k=min(depth, n), show_progress=False)
            bm_ids = res[0].astype(np.int64)
            bm_sc = np.asarray(sc[0], dtype=np.float32)
        qvec = None
        d_ids = np.empty(0, np.int64)
        d_sc = np.empty(0, np.float32)
        if searcher._dense is not None:
            if not hasattr(searcher, "embedder"):
                searcher.embedder = searcher._make_embedder()
            qvec = searcher.embedder.encode_query(q.question)
            sims = searcher._dense @ qvec
            k = min(depth, len(sims))
            part = np.argpartition(-sims, k - 1)[:k]
            d_ids = part[np.argsort(-sims[part])].astype(np.int64)
            d_sc = sims[d_ids].astype(np.float32)
        # C1: لیست سوم sparse وقتی ایندکس sparse دارد و embedder از backend m3 است
        sp_ids = np.empty(0, np.int64)
        sp_sc = np.empty(0, np.float32)
        m3_cls = getattr(search, "M3Embedder", None)
        use_sparse = (searcher._dense is not None
                      and getattr(searcher, "_sparse", None) is not None
                      and m3_cls is not None and hasattr(searcher, "embedder")
                      and isinstance(searcher.embedder, m3_cls))
        if use_sparse:
            sp_ids, sp_sc = searcher._sparse_top(searcher.embedder.encode_query_sparse(q.question), depth)
        nb, nd = _minmax(bm_ids, bm_sc), _minmax(d_ids, d_sc)
        ns = _minmax(sp_ids, sp_sc)
        beta = float(beta) if ns else 0.0
        cand = sorted(set(nb) | set(nd) | set(ns))
        scored = sorted(((alpha * nd.get(r, 0.0) + max(0.0, 1 - alpha - beta) * nb.get(r, 0.0)
                          + beta * ns.get(r, 0.0), r)
                         for r in cand), key=lambda t: (-t[0], t[1]))[:fused_top]
        rb = {int(r): i + 1 for i, r in enumerate(bm_ids)}
        rd = {int(r): i + 1 for i, r in enumerate(d_ids)}
        rs = {int(r): i + 1 for i, r in enumerate(sp_ids)}
        hits: list[Hit] = []
        for s, r in scored:
            ranks: dict[str, int] = {}
            if r in rb:
                ranks["bm25"] = rb[r]
            if r in rd:
                ranks["dense"] = rd[r]
            if r in rs:
                ranks["sparse"] = rs[r]
            d_sim = float(searcher._dense[r] @ qvec) if qvec is not None else None
            ch = searcher._chunks[r]
            hits.append(Hit(row=r, chunk_id=ch["id"], score=round(float(s), 6), ranks=ranks,
                            valid_hit=valid_hit(ranks.get("bm25"), d_sim, min_sim)))
        passages = searcher._assemble(hits)
        ms = (time.perf_counter() - t0) * 1000
        out[q.id] = RunRow(rows=[h.row for h in hits], found=any(h.valid_hit for h in hits),
                           context="\n".join(p.text for p in passages),
                           timings_ms={"total_ms": round(ms, 1)})
    return out


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def _dcg(rels: list[int]) -> float:
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def metrics(run: dict[str, RunRow], gold: dict[str, dict[int, int]],
            ks: tuple[int, ...] = DEFAULT_KS,
            quotes: dict[str, list[str]] | None = None) -> dict:
    """Aggregate metrics over ANSWERED questions (gold non-empty) + per-question
    vectors for paired tests. ContextHit needs the witness quotes."""
    per: dict[str, list[float]] = {"context_hit": [], "hit@5": [], "mrr@10": []}
    acc: dict[str, list[float]] = {"mrr@10": [], "ndcg@10": []}
    for k in ks:
        acc[f"hit@{k}"] = []
        acc[f"recall@{k}"] = []
    lat: list[float] = []
    for qid, r in run.items():
        g = gold.get(qid) or {}
        if not g:
            continue
        top = r.rows
        for k in ks:
            acc[f"hit@{k}"].append(1.0 if any(x in g for x in top[:k]) else 0.0)
            acc[f"recall@{k}"].append(sum(1 for x in top[:k] if x in g) / len(g))
        idcg = _dcg(sorted(g.values(), reverse=True)[:10])
        dcg = _dcg([g.get(x, 0) for x in top[:10]])
        acc["ndcg@10"].append(dcg / idcg if idcg > 0 else 0.0)
        mrr = 0.0
        for i, x in enumerate(top[:10]):
            if x in g:
                mrr = 1.0 / (i + 1)
                break
        acc["mrr@10"].append(mrr)
        per["hit@5"].append(acc["hit@5"][-1])
        per["mrr@10"].append(mrr)
        if quotes is not None and qid in quotes:
            ctx = _norm(r.context)
            ok = all(_norm(qq) in ctx for qq in quotes[qid] if qq.strip())
            per["context_hit"].append(1.0 if ok else 0.0)
        lat.append(float(r.timings_ms.get("total_ms") or 0))
    agg: dict = {"n": len(acc["hit@5"])}
    for key, vals in acc.items():
        agg[key] = round(float(np.mean(vals)), 4) if vals else None
    if quotes is not None:
        agg["context_hit"] = round(float(np.mean(per["context_hit"])), 4) if per["context_hit"] else None
    agg["p50_ms"] = round(float(np.percentile(lat, 50)), 1) if lat else None
    agg["p95_ms"] = round(float(np.percentile(lat, 95)), 1) if lat else None
    agg["per_q"] = per
    return agg


def abstain_stats(qs: list[Question], run: dict[str, RunRow]) -> dict:
    """Abstain-acc روی slice «none» (باید found=False شود) + نرخ abstain اشتباه
    روی سؤال‌های جواب‌دار (هر دو به‌علاوه‌ی هم مهم‌اند)."""
    none_ids = [q.id for q in qs if q.slice == "none" and q.id in run]
    ans_ids = [q.id for q in qs if q.slice != "none" and q.id in run]
    n_acc = sum(1 for i in none_ids if not run[i].found) / len(none_ids) if none_ids else None
    f_abs = sum(1 for i in ans_ids if not run[i].found) / len(ans_ids) if ans_ids else None
    return {"abstain_acc": None if n_acc is None else round(n_acc, 3),
            "false_abstain": None if f_abs is None else round(f_abs, 3)}


def paired_test(a: list[float], b: list[float], n: int = 10000, seed: int = 0) -> float:
    """Two-sided paired permutation test (sign-flip) on per-question metric
    vectors → p-value (README §10: با ۲۰–۵۰ سؤال، تفاوت ۲٪ معمولاً نویز است)."""
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(va) != len(vb) or len(va) == 0:
        return 1.0
    d = va - vb
    if not np.any(d):
        return 1.0
    obs = abs(float(d.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    return float((np.count_nonzero(perm >= obs - 1e-12) + 1) / (n + 1))


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
VARIANTS: dict[str, dict] = {
    "base": {},
    "bm25": {"lists": ["bm25"]},
    "dense": {"lists": ["dense"]},
    "w_bm25_0.7": {"weights": {"bm25": 0.7, "dense": 1.0}},
    "w_dense_2": {"weights": {"bm25": 1.0, "dense": 2.0}},
    "k20": {"rrf_k": 20},
    "k100": {"rrf_k": 100},
    "depth30": {"depth": 30},
    "depth300": {"depth": 300},
    "nb0": {"neighbors": 0},
    "nb2": {"neighbors": 2},
    "cc": {"fusion": "cc", "alpha": 0.5},
    "cc0.3": {"fusion": "cc", "alpha": 0.3},
    "cc0.7": {"fusion": "cc", "alpha": 0.7},
    "rerank": {"rerank": True},          # C2: cross-encoder bge-reranker-v2-m3 روی fused
    "rerank3": {"rerank": True, "lists": ["bm25", "dense"], "fused_top": 3},
}


def _lookup_variant(name: str) -> dict | None:
    if name in VARIANTS:
        return dict(VARIANTS[name])
    if name.startswith("cc"):
        try:                                   # arbitrary alphas, e.g. cc0.42
            return {"fusion": "cc", "alpha": float(name[2:])}
        except ValueError:
            return None
    return None


def _abstain_sweep(searcher: Searcher, qs: list[Question], gold: dict[str, dict[int, int]]) -> None:
    if searcher._dense is None:
        print("[note] BM25-only index — min_dense_sim sweep needs dense; nothing to do.")
        return
    print(f"{'thr':>5} {'abstain_acc':>11} {'false_abstain':>14}")
    saved = searcher.cfg.search.min_dense_sim
    try:
        for thr in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
            searcher.cfg.search.min_dense_sim = thr
            run = run_variant(searcher, qs, {})
            st = abstain_stats(qs, run)
            print(f"{thr:>5.2f} {st['abstain_acc']:>11.3f} {st['false_abstain']:>14.3f}")
    finally:
        searcher.cfg.search.min_dense_sim = saved


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="BookSearch-FA evaluation (README §10)")
    ap.add_argument("--variants", default="base", help="comma list, see VARIANTS")
    ap.add_argument("--split", default="all", choices=["dev", "test", "all"])
    ap.add_argument("--index", default="index")
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--results", default="eval/results.jsonl")
    ap.add_argument("--abstain-sweep", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    qs_all = load_questions(args.questions)
    if args.split != "all":
        qs_all = [q for q in qs_all if q.split == args.split]
    searcher = Searcher(index_dir=args.index)
    gold = resolve_gold(qs_all, searcher)

    bad = [q.id for q in qs_all if q.slice != "none" and not gold[q.id]]
    leak = [q.id for q in qs_all if q.slice == "none" and gold[q.id]]
    if bad:
        print(f"[warn] gold حل نشد برای: {', '.join(bad)} (quote/صفحه را چک کنید)")
    if leak:
        print(f"[warn] سؤال «none» در کتاب match شد: {', '.join(leak)}")

    if args.abstain_sweep:
        _abstain_sweep(searcher, qs_all, gold)
        return

    answered = [q for q in qs_all if q.slice != "none" and gold[q.id]]
    quotes = {q.id: [w.quote for w in q.witnesses] for q in answered}
    answered_ids = [q.id for q in answered]

    table: dict[str, dict] = {}
    for name in args.variants.split(","):
        name = name.strip()
        ov = _lookup_variant(name)
        if ov is None:
            print(f"[skip] variant نامعتبر: '{name}'")
            continue
        run = run_variant(searcher, qs_all, ov)
        m = metrics({k: run[k] for k in answered_ids}, gold, quotes=quotes)
        m["abstain"] = abstain_stats(qs_all, run)
        table[name] = m

    hdr = (f"{'variant':<11} {'n':>3} {'CH':>6} {'Hit@1':>6} {'Hit@5':>6} {'Hit@10':>7} "
           f"{'R@100':>6} {'MRR':>6} {'nDCG':>6} {'abst':>6} {'f-abst':>7} {'p50':>7} {'p95':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, m in table.items():
        ab = m.get("abstain") or {}
        print(f"{name:<11} {m['n']:>3} "
              f"{(m.get('context_hit') if m.get('context_hit') is not None else float('nan')):>6.3f} "
              f"{m['hit@1']:>6.3f} {m['hit@5']:>6.3f} {m['hit@10']:>7.3f} {m['recall@100']:>6.3f} "
              f"{m['mrr@10']:>6.3f} {m['ndcg@10']:>6.3f} "
              f"{(ab.get('abstain_acc') if ab.get('abstain_acc') is not None else float('nan')):>6.2f} "
              f"{(ab.get('false_abstain') if ab.get('false_abstain') is not None else float('nan')):>7.2f} "
              f"{(m['p50_ms'] if m['p50_ms'] is not None else float('nan')):>7.0f} "
              f"{(m['p95_ms'] if m['p95_ms'] is not None else float('nan')):>8.0f}")

    # per-slice breakdown of the FIRST variant (baseline)
    first = next(iter(table), None)
    if first and args.split != "test":
        base_run = run_variant(searcher, qs_all, _lookup_variant(first))
        print(f"\nper-slice ({first}):")
        for sl in ("name", "paraphrase", "number", "multi", "footnote", "mixed"):
            ids = [q.id for q in qs_all if q.slice == sl and gold[q.id]]
            if not ids:
                continue
            sub = metrics({k: base_run[k] for k in ids}, gold, quotes=quotes)
            print(f"  {sl:<11} n={sub['n']:>2} CH={sub.get('context_hit')} Hit@5={sub['hit@5']} MRR={sub['mrr@10']}")

    # paired permutation tests vs base (README §10)
    if len(table) > 1 and "base" in table and table["base"]["per_q"]["context_hit"]:
        print("\npaired permutation test vs base (p-value, lower=better signal):")
        for name, m in table.items():
            if name == "base":
                continue
            p_ch = paired_test(table["base"]["per_q"]["context_hit"], m["per_q"]["context_hit"])
            p_h5 = paired_test(table["base"]["per_q"]["hit@5"], m["per_q"]["hit@5"])
            print(f"  base vs {name:<11} CH p={p_ch:.4f}  Hit@5 p={p_h5:.4f}")

    if args.json:
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_q"} for k, v in table.items()},
                         ensure_ascii=False, indent=2, default=str))

    # append-only log (README §3 file layout)
    rpath = Path(args.results)
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "split": args.split, "norm": search.NORM_VERSION, "tok": search.BM25_TOKEN_VERSION,
           "n_questions": len(qs_all),
           "variants": {k: {kk: vv for kk, vv in v.items() if kk != "per_q"} for k, v in table.items()}}
    with open(rpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
