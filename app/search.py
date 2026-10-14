"""
search.py — query-side of BookSearch-FA (stage 1: minimal version).

Heavy PDF/OCR work lives in ingest.py; this file only needs numpy (+ optional
torch / sentence-transformers), so a long-running app can import it cheaply.

Stage-1 Searcher loads:
  index/merged/CURRENT/gen-*/manifest.json   (row layout: books -> chunks)
  index/merged/CURRENT/gen-*/bm25/           (bm25s save() output)
  index/books/<key>/chunks.jsonl             (chunk texts, read in manifest order)
  dense vectors: fp16 .npy per book, converted once to ONE float32 matrix (CPU)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import warnings
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

# هشدار deprecation داخلی hf_hub — بی‌اثر برای ما و فقط نویز در CLI (قبل از import ST)
warnings.filterwarnings("ignore", message=r".*HF_HUB_ENABLE_HF_TRANSFER.*")

# ----------------------------------------------------------------------------
# Optional imports: torch/sentence-transformers only if the user asked for dense
# ----------------------------------------------------------------------------
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

try:
    from sentence_transformers import SentenceTransformer
    HAVE_ST = True
except Exception:
    HAVE_ST = False


NORM_VERSION = "fa-norm-3"  # bump if normalization rules change (index invalidation)
BM25_TOKEN_VERSION = "fa-tok-2"  # bump if the BM25 tokenizer changes (manifest "bm25.tok"; stale -> BM25 rebuilt)

_NORM_PREFIX_RE = re.compile(r"fa-norm-\d+")


def parse_embed_version(emb: str) -> tuple[str, str, int]:
    """manifest embed version -> (model, backend, max_length).
    Accepts 'fa-norm-3|BAAI/bge-m3|m3|512|hdr1' (norm-prefixed, current ingest),
    the older unprefixed 'BAAI/bge-m3|st|512|hdr1', and 'BAAI/bge-m3|N|…'
    without an explicit backend (defaults to 'st')."""
    parts = emb.split("|")
    i = 1 if _NORM_PREFIX_RE.match(parts[0]) else 0   # optional norm-version prefix
    model = parts[i] if i < len(parts) else ""
    tail = parts[i + 1:]
    backend = "st"
    if tail and tail[0] in ("st", "m3"):              # model|backend|N|…
        backend = tail[0]
        tail = tail[1:]
    maxlen = 512
    if tail and tail[0].isdigit():                    # …|N|…
        maxlen = int(tail[0])
    elif len(tail) > 1 and tail[1].isdigit():         # model|N|… (backend غایب)
        maxlen = int(tail[1])
    return model, backend, maxlen

# ----------------------------------------------------------------------------
# Config (booksearch.toml via stdlib tomllib)
# ----------------------------------------------------------------------------
@dataclass
class EmbedConfig:
    model: str = "BAAI/bge-m3"
    backend: str = "st"    # st (sentence-transformers) | m3 (FlagEmbedding BGEM3؛ با خروجی sparse)
    sparse: bool = True     # فقط با backend=m3: ذخیرهٔ sparse.npz کنار dense
    device: str = "auto"
    fp16: bool = True
    max_length: int = 512
    batch_size: int = 32
    num_threads: int = 0   # 0 = پیش‌فرض torch؛ با اندازه‌گیری تنظیم شود (روی CPU تأثیر زیادی دارد)
    query_instruction: str = ""


@dataclass
class ExtractConfig:
    """[extract] — استخراج، quality gate و OCR (README §1.1 و §13)."""
    ocr_engine: str = "none"         # tesseract | none (پیش‌فرض کد: خاموش؛ در booksearch.toml روشن می‌شود)
    ocr_lang: str = "fas+eng"
    ocr_dpi: int = 300
    workers: int = 0                 # 0 = سریال؛ N = ProcessPool برای OCR
    tesseract_cmd: str = ""          # مسیر tesseract.exe اگر روی PATH نیست
    tessdata_prefix: str = ""        # پوشهٔ traineddata (پیش‌فرض: ./tessdata پروژه)
    force_ocr: bool = False          # OCR همهٔ صفحات (per-book: library/books.toml)
    gate: dict = field(default_factory=lambda: {"min_chars": 150, "min_script_ratio": 0.6,
                                                "max_bad_ratio": 0.02, "min_lex_hit": 0.3})


@dataclass
class SearchConfig:
    depth: int = 100
    rrf_k: int = 60
    weights: dict = field(default_factory=lambda: {"bm25": 1.0, "dense": 1.0})
    fusion: str = "cc"               # cc | rrf  (CC برندهٔ eval روی dev: README §8)
    cc_alpha: float = 0.5            # وزن dense در CC؛ α∈[0.4,0.7] روی dev تقریباً تخت بود
    cc_beta: float = 0.25            # وزن لیست سوم (sparse bge-m3) در CC؛ 0 = خاموش
    fused_top: int = 30
    final_chunks: int = 6
    neighbors: int = 1
    context_tokens: int = 3000
    # آستانهٔ شباهت کسینوسی برای «hit معتبرِ» dense؛ روی slice بی‌جوابِ eval تنظیم شود
    min_dense_sim: float = 0.5
    # C2: rerank مرحلهٔ دوم — کلیدهای RerankConfig؛ enabled=false یعنی هزینهٔ صفر
    rerank: dict = field(default_factory=lambda: {"enabled": False})


@dataclass
class Config:
    paths: dict = field(default_factory=lambda: {"library": "library", "index": "index"})
    chunk: dict = field(default_factory=lambda: {"target_tokens": 300, "max_tokens": 420,
                                                 "overlap_sentences": 1, "header": True})
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    bm25: dict = field(default_factory=lambda: {"k1": 1.2, "b": 0.75})
    search: SearchConfig = field(default_factory=SearchConfig)


def load_config(path: str | Path = "booksearch.toml") -> Config:
    """Read booksearch.toml (if present) into the Config dataclasses. Missing keys keep defaults."""
    cfg = Config()
    p = Path(path)
    if not p.exists():
        return cfg
    import tomllib  # Python 3.11+
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    cfg.paths.update(raw.get("paths", {}))
    cfg.chunk.update(raw.get("chunk", {}))
    ex = raw.get("extract", {})
    for f in ("ocr_engine", "ocr_lang", "ocr_dpi", "workers", "tesseract_cmd",
              "tessdata_prefix", "force_ocr"):
        if f in ex:
            setattr(cfg.extract, f, ex[f])
    if isinstance(ex.get("gate"), dict):
        cfg.extract.gate.update(ex["gate"])
    e = raw.get("embed", {})
    for f in ("model", "backend", "sparse", "device", "fp16", "max_length", "batch_size",
              "num_threads", "query_instruction"):
        if f in e:
            setattr(cfg.embed, f, e[f])
    cfg.bm25.update(raw.get("bm25", {}))
    s = raw.get("search", {})
    if "weights" in s:
        cfg.search.weights = dict(s["weights"])
    if isinstance(s.get("rerank"), dict):          # [search.rerank] به‌صورت جدول تو در تو
        cfg.search.rerank = dict(s["rerank"])
    for f in ("depth", "rrf_k", "fusion", "cc_alpha", "cc_beta", "fused_top",
              "final_chunks", "neighbors", "context_tokens", "min_dense_sim"):
        if f in s:
            setattr(cfg.search, f, s[f])
    if isinstance(raw.get("rerank"), dict):        # جدول مستقل [rerank] در toml
        cfg.search.rerank = {**cfg.search.rerank, **raw["rerank"]}
    return cfg


# ----------------------------------------------------------------------------
# Persian normalization (one deterministic implementation, no hazm)
# ----------------------------------------------------------------------------
# chars that must not survive display/search text: bidi marks, controls, tags
_CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f\u200b\u200e\u200f\u202a-\u202e"
                         r"\u2066-\u2069\ufeff]")
_SPACES_RE = re.compile(r"[ \t\u00a0\u2000-\u200a\u3000]+")
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DIACRITICS_RE = re.compile(r"[\u0640\u064b-\u0652\u0670]")  # tatweel + harakat

_ZWNJ = "\u200c"


def normalize_display(text: str) -> str:
    """For stored/quoted text: NFKC, Arabic->Persian yeh/kaf, strip control chars, unify spaces.

    ZWNJ is preserved here (it is meaningful in Persian).
    fa-norm-3: ؀ (U+06C0) and «هٔ» (ه + U+0654) both become plain «ه»."""
    import unicodedata
    t = unicodedata.normalize("NFKC", text)          # opens presentation forms from PDFs
    t = t.replace("\u064a", "\u06cc").replace("\u0649", "\u06cc")  # ي/ى -> ی
    t = t.replace("\u0643", "\u06a9")                 # ك -> ک
    t = _CONTROL_RE.sub("", t)                        # LRM/RLM/ZWJ/controls
    t = t.replace("\u0629", "\u0647")                 # ة -> ه
    t = t.replace("\u0654", "").replace("\u0655", "")  # leftover combining hamza (fa-norm-3)
    t = t.replace("\u06c0", "\u0647").replace("\u06c1", "\u0647")  # ؀/ۀ/ہ -> ه (NFKC composes ه+U+0654 to ؀)
    t = _SPACES_RE.sub(" ", t)
    return t.strip()


def normalize_search(text: str) -> str:
    """For BOTH documents and queries before BM25/embedding (on top of normalize_display):
    digits->ASCII, remove diacritics/tatweel, lowercase latin. Idempotent."""
    t = normalize_display(text)
    t = t.translate(_ARABIC_DIGITS)
    t = _DIACRITICS_RE.sub("", t)
    return t.lower()


# tiny Persian stopword list (~40 most frequent function words; negation words kept)
# + colloquial question/boilerplate words (چنده، میخوام، …): they are OOV in the
# corpus and only dilute BM25 matches. Negation words are kept.
_STOPWORDS = frozenset("""
و در به از که را با این برای است بود های ها می شود کرد خود آن تا هم بر یا
یک گی برخی دیگر همه هر چون اما اگر چه بسیار باید دارد دارند داشته خواهد
ی چطور چقدر چنده چیه چرا کجا کجاست کی میشه میخوام خوام میخوای هست هستش
باشه دنبال لطفا طوری جوری بگم
""".split())

# Arabic-script punctuation lives INSIDE \u0600-\u06ff, so the tokenizer's char
# class keeps it glued to words («اکسپلویت،» became one OOV token). Split on it.
_ARABIC_PUNCT_MAP = str.maketrans({"،": " ", "؛": " ", "؟": " "})


def tokenize_bm25(text: str) -> list[str]:
    """normalize_search + split on non-letters + ZWNJ->space (Lucene PersianAnalyzer behavior)
    + drop short stopwords. Same function for docs and queries."""
    t = normalize_search(text).replace(_ZWNJ, " ")
    t = t.translate(_ARABIC_PUNCT_MAP)
    # keep unicode letters/digits, split on everything else
    toks = re.findall(r"[\w\u0600-\u06ff]+", t, flags=re.UNICODE)
    return [w for w in toks if w not in _STOPWORDS]


# ----------------------------------------------------------------------------
# Embedder (optional: only needed when dense vectors exist / are built)
# ----------------------------------------------------------------------------
class Embedder:
    """Wraps sentence-transformers. Loads the model once; fp16 on GPU."""

    def __init__(self, cfg: EmbedConfig):
        if not (HAVE_TORCH and HAVE_ST):
            raise RuntimeError("torch + sentence-transformers are required for the Embedder "
                               "(pip install torch sentence-transformers)")
        import torch
        dev = cfg.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = dev
        self.fp16 = bool(cfg.fp16) and dev == "cuda"
        try:    # مدل کش‌شده: بدون هیچ تماس شبکه‌ای → سریع‌تر و بدون هشدارهای HF Hub
            self.model = SentenceTransformer(cfg.model, device=dev, local_files_only=True)
        except Exception:                       # اولین اجرا: اجازهٔ دانلود بده
            self.model = SentenceTransformer(cfg.model, device=dev)
        if self.fp16:                          # fp16 weights on GPU: ~2x encode, half VRAM (README §12)
            self.model.half()
        # نسخهٔ embed در manifest همیشه max_length را ادعا می‌کرد ولی مدل به آن بی‌اعتنا بود
        # (bge-m3 پیش‌فرض 8192). حالا واقعاً اعمال می‌شود: سقف بدترین حالت کوئری‌های طولانی،
        # و برای تکه‌ها (≤420 توکن) هیچ تغییری در بردارها ایجاد نمی‌کند.
        self.model.max_seq_length = int(cfg.max_length)
        if cfg.num_threads > 0:                # اندازه‌گیری کنید؛ هستهٔ فیزیکی همیشه برنده نیست
            torch.set_num_threads(int(cfg.num_threads))
        self.cfg = cfg

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        """[n, dim] float32, L2-normalized (batches sorted by length for speed)."""
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        sorted_texts = [texts[i] for i in order]
        vecs = self.model.encode(sorted_texts, batch_size=self.cfg.batch_size,
                                 normalize_embeddings=True, show_progress_bar=len(texts) > 64)
        vecs = np.asarray(vecs, dtype=np.float32)
        out = np.empty_like(vecs)
        out[order] = vecs                      # restore original order
        return out

    @lru_cache(maxsize=256)
    def _cached_query(self, q: str) -> np.ndarray:
        text = (self.cfg.query_instruction + q) if self.cfg.query_instruction else q
        v = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)[0]

    def encode_query(self, q: str) -> np.ndarray:
        """[dim] float32 for one query (LRU-cached for repeated eval runs)."""
        return self._cached_query(q)


class M3Embedder:
    """C1: همان مدل bge-m3 با FlagEmbedding (BGEM3FlagModel) به‌جای sentence-transformers؛
    علاوه بر dense، خروجی sparse (lexical weights، سبک SPLADE) هم می‌دهد — لیست سوم جستجو.
    امضای API مثل Embedder است (encode_docs/encode_query) + encode_docs_sparse."""

    def __init__(self, cfg: EmbedConfig):
        if not HAVE_TORCH:
            raise RuntimeError("torch is required for the m3 backend")
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise RuntimeError("backend=m3 به FlagEmbedding نیاز دارد: "
                               "pip install FlagEmbedding") from e
        import torch
        dev = cfg.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = dev
        self.fp16 = bool(cfg.fp16) and dev == "cuda"
        # نکته: BGEM3FlagModel با local_files_only=True اگر یک فایل حاشیه‌ایِ snapshot
        # (مثلاً onnx) در کش نباشد استثنا می‌دهد؛ پس اول آفلاین امتحان و در خطا، لِینت
        # شبکه فقط برای meta. مدل اصلی bge-m3 در کش کامل است (safetensors 2.27GB).
        import os as _os
        try:
            _os.environ["HF_HUB_OFFLINE"] = "1"
            self.model = BGEM3FlagModel(cfg.model, use_fp16=self.fp16, devices=dev)
        except Exception:
            _os.environ.pop("HF_HUB_OFFLINE", None)
            self.model = BGEM3FlagModel(cfg.model, use_fp16=self.fp16, devices=dev)
        finally:
            _os.environ.pop("HF_HUB_OFFLINE", None)
        if cfg.num_threads > 0:
            torch.set_num_threads(int(cfg.num_threads))
        self.cfg = cfg

    def _run_encode(self, texts: list[str], return_dense: bool, return_sparse: bool,
                    ckpt_path: str | None = None):
        """داخلی: encode با پیشرفت زنده (هر 256 متن یک خط برای ایندکس‌های بزرگ).
        خروجی: (dense_vecs|None, list lexical_weights|None).
        اگر ckpt_path داده شود، پیشرفت هر بلوک 256تایی به‌صورت fp16+pickle ذخیره می‌شود
        و اگر فایل موجود باشد، ادامه از همان‌جا (چک‌پوینت embed — sync مکرر بعد از قطعی)."""
        n = len(texts)
        step = 256
        dense_parts, sparse_parts = [], []
        done = 0
        ck = None
        if ckpt_path:
            ck_p = Path(ckpt_path)
            if ck_p.exists():
                try:
                    ck = np.load(ckpt_path, allow_pickle=True)
                    if int(ck["n_texts"]) == n and list(ck["texts_sha"]) == \
                            [hashlib.sha1(t.encode("utf-8")).hexdigest()[:12] for t in texts[:64]] + \
                            [hashlib.sha1(t.encode("utf-8")).hexdigest()[:12] for t in texts[64::97]]:
                        done = int(ck["done"])
                        d = ck["dense"] if return_dense else None
                        s = list(ck["sparse"]) if return_sparse else None
                        if d is not None:
                            dense_parts = [np.asarray(d, dtype=np.float32)]
                        if s is not None:
                            sparse_parts = s
                        print(f"  [embed] checkpoint: resuming at {done}/{n}", flush=True)
                    else:
                        ck = None                       # متن‌ها عوض شده — چک‌پوینت بی‌اعتبار
                except Exception:
                    ck = None
        for s in range(done, n, step):
            part = texts[s:s + step]
            out = self.model.encode(part, batch_size=self.cfg.batch_size,
                                    max_length=int(self.cfg.max_length),
                                    return_dense=return_dense, return_sparse=return_sparse,
                                    return_colbert_vecs=False)
            if return_dense:
                dense_parts.append(out["dense_vecs"])
            if return_sparse:
                sparse_parts.extend(out["lexical_weights"])
            if n > 300:
                print(f"  [embed] {min(s + step, n)}/{n}", flush=True)
            if ckpt_path and ((s + step) % 1024 < step or (s + step) >= n):
                # چک‌پوینت هر ~۱۰۲۴ متن (4 بلوک): fp16 + pickle اتمیک
                try:
                    dck = (np.concatenate(dense_parts, axis=0).astype(np.float16)
                           if return_dense else np.zeros((0,), dtype=np.float16))
                    sp = (np.array(sparse_parts, dtype=object)
                          if return_sparse else np.zeros((0,), dtype=object))
                    sha = [hashlib.sha1(t.encode("utf-8")).hexdigest()[:12]
                           for t in texts[:64]] + \
                          [hashlib.sha1(t.encode("utf-8")).hexdigest()[:12]
                           for t in texts[64::97]]
                    tmp = ckpt_path + ".tmp"
                    with open(tmp, "wb") as f:
                        np.savez_compressed(f, dense=dck, sparse=sp, done=min(s + step, n),
                                            n_texts=n, texts_sha=sha)
                    os.replace(tmp, ckpt_path)
                except Exception as e:              # noqa: BLE001 — چک‌پوینت نباید sync را بکشد
                    print(f"  [embed] checkpoint write failed (ignored): {e}", flush=True)
        dense = None
        if return_dense:
            dense = np.concatenate(dense_parts, axis=0) if len(dense_parts) > 1 \
                else np.asarray(dense_parts[0], dtype=np.float32)
        return dense, (sparse_parts if return_sparse else None)

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        dense, _ = self._run_encode(texts, True, False)
        vecs = np.asarray(dense, dtype=np.float32)
        n = np.linalg.norm(vecs, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return vecs / n                        # L2-normalized، هم‌خوان با شاخهٔ st

    def encode_docs_sparse(self, texts: list[str]) -> list[dict[str, float]]:
        """lexical weights هر متن: {token_id(str): weight} — برای sparse.npz.
        جدا از encode_docs تا مسیر بدون sparse (sync --no-sparse) سبک بماند."""
        _, lw = self._run_encode(texts, False, True)
        return [{str(k): float(v) for k, v in d.items()} for d in lw]

    def encode_docs_both(self, texts: list[str], ckpt_path: str | None = None) -> \
            tuple[np.ndarray, list[dict[str, float]]]:
        """C1: یک پاسِ مشترک dense+sparse — نصفِ زمانِ دو فراخوانی جدا.
        فقط با backend=m3؛ خروجی: (dense L2-normalized، lexical weights).
        ckpt_path: مسیر فایل چک‌پوینت (اگر موجود باشد ادامه از آن — resumable embed)."""
        dense, lw = self._run_encode(texts, True, True, ckpt_path=ckpt_path)
        vecs = np.asarray(dense, dtype=np.float32)
        n = np.linalg.norm(vecs, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return vecs / n, [{str(k): float(v) for k, v in d.items()} for d in lw]

    @lru_cache(maxsize=256)
    def _cached_query(self, q: str) -> np.ndarray:
        text = (self.cfg.query_instruction + q) if self.cfg.query_instruction else q
        out = self.model.encode_queries([text], max_length=int(self.cfg.max_length),
                                        return_dense=True, return_sparse=False,
                                        return_colbert_vecs=False)
        v = np.asarray(out["dense_vecs"], dtype=np.float32)[0]
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def encode_query(self, q: str) -> np.ndarray:
        return self._cached_query(q)

    def encode_query_sparse(self, q: str) -> dict[str, float]:
        text = (self.cfg.query_instruction + q) if self.cfg.query_instruction else q
        out = self.model.encode_queries([text], max_length=int(self.cfg.max_length),
                                        return_dense=False, return_sparse=True,
                                        return_colbert_vecs=False)
        return {str(k): float(v) for k, v in out["lexical_weights"][0].items()}


class Reranker:
    """C2: rerank مرحلهٔ دوم — cross-encoder bge-reranker-v2-m3 روی کاندیدهای fused.
    فقط زخم کاندیدها را دوباره می‌چیند؛ تعداد hitها را عوض نمی‌کند. fp16 فقط روی GPU
    (روی CPU fp32 کند اما درست است). lazy: مدل فقط با اولین rerank بالا می‌آید."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is not None:
                return
            try:
                from FlagEmbedding import FlagAutoReranker
            except ImportError as e:
                raise RuntimeError("rerank به FlagEmbedding نیاز دارد: "
                                   "pip install FlagEmbedding") from e
            import torch
            dev = str(self.cfg.get("device", "auto"))
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            use_fp16 = bool(self.cfg.get("fp16", True)) and dev == "cuda"
            model = str(self.cfg.get("model", "BAAI/bge-reranker-v2-m3"))
            try:
                self._model = FlagAutoReranker.from_finetuned(
                    model, use_fp16=use_fp16, devices=dev, batch_size=int(self.cfg.get("batch_size", 16)),
                    max_length=int(self.cfg.get("max_length", 512)), local_files_only=True)
            except Exception:
                self._model = FlagAutoReranker.from_finetuned(
                    model, use_fp16=use_fp16, devices=dev,
                    batch_size=int(self.cfg.get("batch_size", 16)),
                    max_length=int(self.cfg.get("max_length", 512)))

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """score برای هر متن (هم‌تراز با ورودی)؛ ترتیب ورودی حفظ می‌شود."""
        self._load()
        scores = self._model.compute_score(list(zip([query] * len(texts), texts)),
                                           normalize=bool(self.cfg.get("normalize", True)))
        if isinstance(scores, (int, float)):
            scores = [scores]
        return [float(s) for s in scores]

    @classmethod
    def enabled(cls, rerank_cfg: dict | None) -> bool:
        return bool(rerank_cfg and rerank_cfg.get("enabled"))


# ----------------------------------------------------------------------------
# Weighted RRF fusion
# ----------------------------------------------------------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.؟!؛])\s+")


def _split_sents(text: str) -> list[str]:
    """Sentence split on Persian/latin enders — used to drop the chunker's
    overlap sentence when merging adjacent chunks (README §7)."""
    return [s for s in _SENT_SPLIT_RE.split(text) if s]


def cc_fuse(bm25_ids: np.ndarray, bm25_sc: np.ndarray,
            dense_ids: np.ndarray, dense_sc: np.ndarray,
            alpha: float = 0.5, top: int = 30,
            sparse_ids: np.ndarray | None = None,
            sparse_sc: np.ndarray | None = None,
            beta: float = 0.0) -> list[tuple[int, float]]:
    """Convex Combination (README §8): α·minmax(dense) + (1−α)·minmax(bm25) روی
    اجتماع top-N. لیست سوم اختیاری (sparse bge-m3) با وزن β وارد ترکیب می‌شود:
    α·dense + (1−α−β)·bm25 + β·sparse (وقتی sparse نیست، (1−α)·bm25 مثل قبل).
    امتیاز واقعی هر لیست را دارد، برخلاف RRF که فقط رتبه می‌بیند.
    خروجی [(row, score)] بهترین اول، tie-break روی row. اگر یک لیست خالی باشد،
    بقیه به‌تنهایی رتبه‌بندی می‌کنند؛ اگر همه خالی، خروجی خالی (found=False)."""
    def _mm(ids: np.ndarray, sc: np.ndarray) -> dict[int, float]:
        if len(ids) == 0:
            return {}
        sc = np.asarray(sc, dtype=np.float32)
        lo, hi = float(sc.min()), float(sc.max())
        if hi - lo < 1e-9:
            return {int(r): 1.0 for r in ids}
        return {int(r): float((s - lo) / (hi - lo)) for r, s in zip(ids, sc)}

    nb = _mm(bm25_ids, bm25_sc)
    nd = _mm(dense_ids, dense_sc)
    ns = _mm(sparse_ids if sparse_ids is not None else np.empty(0, np.int64),
             sparse_sc if sparse_sc is not None else np.empty(0, np.float32))
    beta = float(beta) if ns else 0.0
    cand = set(nb) | set(nd) | set(ns)
    w_bm = 1.0 - alpha - beta
    scored = sorted(((alpha * nd.get(r, 0.0) + w_bm * nb.get(r, 0.0) + beta * ns.get(r, 0.0), r)
                     for r in cand), key=lambda t: (-t[0], t[1]))
    return [(r, s) for s, r in scored[:top]]


class _SparseIndex:
    """C1: sparse.npz یک کتاب — vocab رشته‌ای (token id به‌عنوان متن) + ماتریس CSR
    وزن‌ها به ترتیب ردیف تکه‌ها. کوئری: تبدیل lexical weights کوئری به بردار sparse
    روی همان vocab و یک جدول‌ضربی. پارامترهای خواندن npz مالکیت خصوصی این کلاس‌اند."""

    def __init__(self, path: Path):
        with np.load(str(path)) as z:
            self.vocab: list[str] = list(z["vocab"])
            indptr = z["indptr"]
            indices = z["indices"]
            data = z["data"]
            shape = z["shape"]
        from scipy.sparse import csr_matrix
        self.n_rows, self.n_cols = (int(shape[0]), int(shape[1]))
        self.mat = csr_matrix((data, indices, indptr), shape=(self.n_rows, self.n_cols))
        self.pos = {t: i for i, t in enumerate(self.vocab)}

    def query_vec(self, lexical: dict[str, float]) -> np.ndarray:
        v = np.zeros(self.n_cols, dtype=np.float32)
        for tok, w in lexical.items():
            i = self.pos.get(tok)
            if i is not None:
                v[i] = max(v[i], float(w))     # lexical weights بدون تکرار توکن‌اند؛ max برای اطمینان
        return v

    def top(self, lexical: dict[str, float], depth: int) -> tuple[np.ndarray, np.ndarray]:
        """top-depth ردیف بر اساس امتیاز sparse (dot) — شبیه _dense_top.
        ردیف‌های امتیاز-صفر کاندید نیستند (لیست بی‌سیگنال خالی است، نه دلخواه)."""
        v = self.query_vec(lexical)
        if not v.any():
            return np.empty(0, np.int64), np.empty(0, np.float32)
        sims = self.mat @ v
        nz = np.nonzero(sims)[0]
        if len(nz) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        depth = min(depth, len(nz))
        part = nz[np.argsort(-sims[nz])][:depth]
        order = part[np.argsort(-sims[part], kind="stable")]
        return order.astype(np.int64), sims[order].astype(np.float32)


def valid_hit(bm25_rank: int | None, dense_sim: float | None,
              min_dense_sim: float = 0.5) -> bool:
    """آیا این hit شواهد واقعی دارد؟ BM25 فقط وقتی می‌شمارد که خودش رتبه‌اش کرده
    باشد (rank >= 1؛ لیست خالیِ zero-signal چیزی تولید نمی‌کند)؛ وگرنه شباهت
    dense باید از آستانهٔ min_dense_sim رد شود (RRF صرف قابل‌اتکا نیست — README §8)."""
    if bm25_rank:
        return True
    if dense_sim is not None:
        return dense_sim >= min_dense_sim
    return False


def rrf(ranked: dict[str, np.ndarray], k: int = 60,
        weights: dict[str, float] | None = None, top: int = 30,
        rows_keep: set[int] | None = None) -> list[tuple[int, float]]:
    """ranked: list-name -> array of row-ids in rank order (rank = position + 1).
    score(d) = sum_r w_r / (k + rank_r(d)); returns [(row, score)] best-first,
    tie-broken on row id for determinism.
    rows_keep: if given, only these rows are scored (intersection mode); the
    caller falls back to the plain union when the intersection is empty."""
    weights = weights or {}
    scores: dict[int, float] = {}
    for name, ids in ranked.items():
        w = float(weights.get(name, 1.0))
        for rank, row in enumerate(ids, start=1):
            r = int(row)
            if rows_keep is not None and r not in rows_keep:
                continue
            scores[r] = scores.get(r, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top]


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------
@dataclass
class Hit:
    row: int
    chunk_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # rank داخل هر لیست کاندید
    valid_hit: bool = False                              # شواهد واقعی دارد؟ (BM25 یا dense ≥ آستانه)


@dataclass
class Passage:
    n: int
    book_title: str
    pages: str
    pdf_pages: tuple[int, int]
    text: str
    rows: list[int]
    score: float


@dataclass
class Result:
    query: str
    passages: list[Passage]
    hits: list[Hit]
    timings_ms: dict[str, float]
    found: bool = True   # حداقل یک hit معتبر (BM25 یا dense ≥ min_dense_sim) هست؟


# ----------------------------------------------------------------------------
# Searcher
# ----------------------------------------------------------------------------
class Searcher:
    """Loads the merged index once, then answers queries with BM25 + dense (+ sparse) fusion."""

    def __init__(self, index_dir: str | Path = "index", cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.index_dir = Path(index_dir)
        self._dense: np.ndarray | None = None  # [n_rows, dim] float32, L2-normalized
        self._sparse: list | None = None       # per-book _SparseIndex | None (C1)
        self._bm25 = None
        self._chunks: list[dict] = []          # all chunks in manifest row order
        self._gen_dir: Path | None = None
        self._load()

    # ---- loading -----------------------------------------------------------
    def _load(self) -> None:
        t0 = time.perf_counter()
        merged = self.index_dir / "merged"
        current = merged / "CURRENT"
        if not current.exists():
            raise FileNotFoundError(
                f"{current} not found — run `python ingest.py sync` first to build the index.")
        gen = current.read_text(encoding="utf-8").strip()
        gdir = merged / gen
        manifest = json.loads((gdir / "manifest.json").read_text(encoding="utf-8"))

        import bm25s
        self._bm25 = bm25s.BM25.load(gdir / "bm25")

        chunks: list[dict] = []
        dense_parts: list[np.ndarray] = []
        sparse_on = bool(manifest.get("sparse"))
        sparse_parts: list = []
        for book in manifest["books"]:
            key = book["key"]
            with open(self.index_dir / "books" / key / "chunks.jsonl", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            if len(rows) != book["n"]:
                raise RuntimeError(f"manifest mismatch for {key}: {len(rows)} chunks != {book['n']}")
            chunks.extend(rows)
            dpath = self.index_dir / "books" / key / "dense.npy"
            if dpath.exists():
                d = np.load(dpath)                       # fp16 [n, dim], already normalized
                if len(d) != book["n"]:
                    raise RuntimeError(f"dense.npy row count mismatch for {key}")
                dense_parts.append(d)
            elif dense_parts:
                raise RuntimeError(f"book {key} has no dense.npy but earlier books do — "
                                   f"run `python ingest.py sync` again")
            spath = self.index_dir / "books" / key / "sparse.npz"
            if sparse_on and spath.exists():
                try:
                    sparse_parts.append(_SparseIndex(spath))
                except Exception as e:                  # scipy غایب یا فایل خراب → sparse بی‌صدا خاموش
                    print(f"[note] sparse.npz for {key} unusable ({e}) — sparse list off")
                    sparse_parts.append(None)
            else:
                sparse_parts.append(None)
        self._chunks = chunks
        self.manifest = manifest
        self._sparse = sparse_parts if any(s is not None for s in sparse_parts) else None

        if dense_parts:
            # one C-contiguous float32 matrix: fast matmul, no per-query copies
            self._dense = np.ascontiguousarray(np.concatenate(dense_parts).astype(np.float32))
        else:
            self._dense = None

        self._gen_dir = gdir
        self.load_ms = (time.perf_counter() - t0) * 1000

        # the query encoder MUST be the model the index was built with;
        # the manifest (not the local config) is the source of truth
        emb = manifest.get("embed", "none")
        if emb and emb != "none":
            self._embed_model, self._embed_backend, self._embed_maxlen = parse_embed_version(emb)
            if self._embed_model != self.cfg.embed.model:
                print(f"[note] index was built with '{self._embed_model}'; "
                      f"using it for queries instead of '{self.cfg.embed.model}'")
        else:
            self._embed_model, self._embed_backend = None, "st"

    def reload_if_changed(self) -> bool:
        """Cheap stat() on CURRENT; swap index if a new generation was built."""
        current = self.index_dir / "merged" / "CURRENT"
        if not current.exists():
            return False
        gen = current.read_text(encoding="utf-8").strip()
        if self._gen_dir is not None and gen == self._gen_dir.name:
            return False
        self._load()
        return True

    def _make_embedder(self):
        """Build the query encoder from the manifest's model/backend (never local config)."""
        cls = M3Embedder if getattr(self, "_embed_backend", "st") == "m3" else Embedder
        return cls(replace(self.cfg.embed, model=self._embed_model,
                           max_length=self._embed_maxlen))

    # ---- internals ----------------------------------------------------------
    def _chunk_index_text(self, ch: dict) -> str:
        """Indexed text = static header (title/chapter) + chunk text (built, never stored)."""
        header = ""
        if self.cfg.chunk.get("header", True):
            bits = [self._book_title(ch.get("key", ""))]
            if ch.get("chapter"):
                bits.append(ch["chapter"])
            if bits and bits[0]:
                header = " › ".join(bits) + "\n"
        return header + ch["text"]

    def _book_title(self, key: str) -> str:
        for b in self.manifest["books"]:
            if b["key"] == key:
                return b.get("title") or Path(b["file"]).stem
        return ""

    def _bm25_top(self, toks: list[str], depth: int) -> tuple[np.ndarray, np.ndarray]:
        """(top-depth row ids, bm25 scores) from bm25s; unknown/empty query tokens
        handled. bm25s silently ranks the WHOLE corpus when no query token is in
        the vocab, so drop OOV tokens first and return no candidates instead (then
        dense, or nothing, wins — never a random BM25 order)."""
        if not toks:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        vocab = getattr(self._bm25, "vocab_dict", None)
        if vocab is not None:
            toks = [t for t in toks if t in vocab]
        if not toks:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        res, sc = self._bm25.retrieve([toks], k=min(depth, len(self._chunks)), show_progress=False)
        return res[0].astype(np.int64), np.asarray(sc[0], dtype=np.float32)

    def _dense_top(self, qvec: np.ndarray, depth: int) -> tuple[np.ndarray, np.ndarray]:
        """(top-depth row ids, cosine sims) — exact search = one matmul."""
        sims = self._dense @ qvec
        depth = min(depth, len(sims))
        part = np.argpartition(-sims, depth - 1)[:depth]
        order = part[np.argsort(-sims[part])]
        return order.astype(np.int64), sims[order].astype(np.float32)

    def _sparse_top(self, lexical: dict[str, float], depth: int,
                    row_filter: set[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """C1: top-depth ردیف‌ها از sparse bge-m3 (lexical weights dot CSR). خروجی
        global row id (book.row0 + local). هر کتاب sparse خودش را دارد؛ نتیجه‌ها
        بین کتاب‌ها بر اساس امتیاز ادغام می‌شوند."""
        if not lexical or self._sparse is None:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        all_ids, all_sc = [], []
        for book, sp in zip(self.manifest["books"], self._sparse):
            if sp is None:
                continue
            lo = int(book["row0"])
            if row_filter is not None and not any(lo <= r < lo + int(book["n"]) for r in row_filter):
                continue                      # این کتاب اصلاً در فیلتر نیست
            ids, scs = sp.top(lexical, depth)   # هر ایندکس sparse فقط ردیف‌های خود کتابش را دارد
            if len(ids):
                all_ids.append(ids.astype(np.int64) + lo)
                all_sc.append(scs)
        if not all_ids:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        ids = np.concatenate(all_ids)
        scs = np.concatenate(all_sc)
        order = np.argsort(-scs, kind="stable")[:depth]
        return ids[order], scs[order]

    # ---- public API ----------------------------------------------------------
    def retrieve(self, query: str, books: list[str] | None = None,
                 overrides: dict | None = None) -> list[Hit]:
        """BM25 + dense (+ sparse bge-m3) → CC fusion → optional cross-encoder rerank.
        rerank فقط ترتیب کاندیدهای fused را دوباره می‌چیند؛ تعداد hitها ثابت می‌ماند."""
        ov = overrides or {}
        sc = self.cfg.search
        depth = int(ov.get("depth", sc.depth))
        k = int(ov.get("rrf_k", sc.rrf_k))
        weights = {**sc.weights, **ov.get("weights", {})}
        use = ov.get("lists") or (["bm25", "dense"] if self._dense is not None else ["bm25"])
        fused_top = int(ov.get("fused_top", sc.fused_top))
        # C2: rerank با override صریح (False خاموش می‌کند) یا [rerank].enabled در کانفیگ
        if "rerank" in ov:
            do_rerank = bool(ov["rerank"])
        else:
            do_rerank = Reranker.enabled(getattr(sc, "rerank", {}))
        if do_rerank and not hasattr(self, "reranker"):
            self.reranker = Reranker(dict(sc.rerank or {}))     # یک‌بار ساخته می‌شود؛ lazy

        row_filter: set[int] | None = None
        if books:
            row_filter = set()
            for b in self.manifest["books"]:
                if b["key"] in books or b["file"] in books or b.get("title") in books:
                    row_filter.update(range(b["row0"], b["row0"] + b["n"]))

        t0 = time.perf_counter()
        toks = tokenize_bm25(query)
        if "dense" in use and self._dense is not None and not hasattr(self, "embedder"):
            self.embedder = self._make_embedder()        # lazy: loads the model once
        if "sparse" in use and self._sparse is not None and not hasattr(self, "embedder"):
            self.embedder = self._make_embedder()        # لیست sparse هم به encoder نیاز دارد
        # C1: sparse فقط وقتی معنا دارد که embedder از backend m3 باشد (lexical weights)
        use_sparse = (self._sparse is not None and hasattr(self, "embedder")
                      and isinstance(self.embedder, M3Embedder))
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as ex:
            f_bm25 = ex.submit(self._bm25_top, toks, depth * 3)  # over-fetch, filter shrinks it
            f_dense = (ex.submit(self._dense_top, self.embedder.encode_query(query), depth)
                       if ("dense" in use and self._dense is not None) else None)
            f_sparse = (ex.submit(self._sparse_top, self.embedder.encode_query_sparse(query), depth)
                        if use_sparse else None)
            bm25_ids, bm25_sc = f_bm25.result()
            dense_ids, dense_sc = f_dense.result() if f_dense else (None, None)
            sparse_ids, sparse_sc = (f_sparse.result() if f_sparse
                                     else (np.empty(0, np.int64), np.empty(0, np.float32)))
        t_lists = (time.perf_counter() - t0) * 1000

        if row_filter is not None:
            bm25_ids = np.array([r for r in bm25_ids if int(r) in row_filter], dtype=np.int64)
            if dense_ids is not None:
                dense_ids = np.array([r for r in dense_ids if int(r) in row_filter], dtype=np.int64)
            if len(sparse_ids):
                keep = [i for i, r in enumerate(sparse_ids) if int(r) in row_filter]
                sparse_ids, sparse_sc = sparse_ids[keep], sparse_sc[keep]

        ranked = {}
        if "bm25" in use:
            ranked["bm25"] = bm25_ids
        if dense_ids is not None and "dense" in use:
            ranked["dense"] = dense_ids
        if use_sparse and len(sparse_ids):
            ranked["sparse"] = sparse_ids
        # هر لیست pre-filter شده است، پس لیست BM25 بی‌سیگنال حالا خالی است نه
        # دلخواه؛ شاهد تک‌لیستی (مثلاً hit کلیدواژه‌ای که dense از دست داده) دیده می‌شود.
        fusion = str(ov.get("fusion", sc.fusion)).lower()
        if fusion == "cc" and dense_ids is not None:
            # CC امتیاز واقعی هر لیست را ترکیب می‌کند (برندهٔ eval روی dev)؛
            # لیست خالیِ BM25 یعنی رتبه‌بندی صرفاً dense — بدون نیاز به حالت خاص.
            alpha = float(ov.get("alpha", sc.cc_alpha))
            beta = float(ov.get("beta", sc.cc_beta)) if (use_sparse and len(sparse_ids)) else 0.0
            fused = cc_fuse(bm25_ids, bm25_sc, dense_ids, dense_sc, alpha=alpha, top=fused_top,
                            sparse_ids=(sparse_ids if use_sparse else None),
                            sparse_sc=(sparse_sc if use_sparse else None), beta=beta)
        else:
            # weighted-RRF union (README §8)؛ fallback بدون dense یا fusion=rrf
            fused = rrf(ranked, k=k, weights=weights, top=fused_top)
        t_after_fuse = time.perf_counter()

        # hit معتبر: شواهد درون‌لیستی. RRF فارغ از کیفیت همیشه کاندید می‌دهد، پس
        # «در کتاب‌ها نیست» را از رتبهٔ BM25 یا شباهت dense تعیین می‌کنیم (README §8 بی‌جواب).
        min_sim = float(ov.get("min_dense_sim", self.cfg.search.min_dense_sim))
        qvec = (self.embedder.encode_query(query)
                if ("dense" in ranked and hasattr(self, "embedder")) else None)

        hits = []
        for row, score in fused:
            ranks = {}
            for name, ids in ranked.items():
                pos = np.where(ids == row)[0]
                if len(pos):
                    ranks[name] = int(pos[0]) + 1
            ch = self._chunks[row]
            d_sim = float(self._dense[row] @ qvec) if qvec is not None else None
            hits.append(Hit(row=row, chunk_id=ch["id"], score=round(score, 6), ranks=ranks,
                            valid_hit=valid_hit(ranks.get("bm25"), d_sim, min_sim)))
        rerank_ms = 0.0
        if do_rerank and hits:
            t1 = time.perf_counter()
            try:
                texts = [self._chunk_index_text(self._chunks[h.row]) for h in hits]
                rs = self.reranker.rerank(query, texts)
                order = sorted(range(len(hits)), key=lambda i: -rs[i])   # امتیاز نزولی
                hits = [hits[i] for i in order]
                for pos, h in enumerate(hits, start=1):
                    h.score = round(float(rs[order[pos - 1]]), 6)
                    h.ranks["rerank"] = pos              # رتبهٔ جدید؛ رتبه‌های bm25/dense/sparse می‌مانند
            except Exception as e:                       # noqa: BLE001 — rerank هرگز جستجو را نشکند
                print(f"[note] rerank skipped: {e}")
            rerank_ms = (time.perf_counter() - t1) * 1000
        self.last_timings = {"lists_ms": round(t_lists, 1),
                             "fuse_ms": round((t_after_fuse - t0) * 1000 - t_lists, 1)}
        if rerank_ms:
            self.last_timings["rerank_ms"] = round(rerank_ms, 1)
        return hits

    def search(self, query: str, **kw) -> Result:
        """retrieve + assemble context passages (neighbors merged, dedup, token budget)."""
        t0 = time.perf_counter()
        hits = self.retrieve(query, **kw)
        passages = self._assemble(hits)
        timings = dict(self.last_timings)
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return Result(query=query, passages=passages, hits=hits, timings_ms=timings,
                      found=any(h.valid_hit for h in hits))

    def _assemble(self, hits: list[Hit]) -> list[Passage]:
        """Expand each hit by `neighbors` chunks in the same section (sec), merge
        adjacent/overlapping rows of the same book, dedup, cap at context_tokens."""
        sc = self.cfg.search
        nb = int(sc.neighbors)
        budget = int(sc.context_tokens)
        chosen: list[int] = []
        for h in hits[: sc.final_chunks]:
            rows = [h.row]
            for d in range(1, nb + 1):
                for r in (h.row - d, h.row + d):
                    if 0 <= r < len(self._chunks) and self._same_sec(h.row, r):
                        rows.append(r)
            chosen.extend(rows)

        chosen = sorted(set(chosen))
        # merge consecutive rows (same book) into ranges
        ranges: list[list[int]] = []
        for r in chosen:
            if ranges and r == ranges[-1][-1] + 1 and self._chunks[r].get("key") == self._chunks[ranges[-1][-1]].get("key"):
                ranges[-1].append(r)
            else:
                ranges.append([r])

        # بودجهٔ توکن به ترتیب امتیاز مصرف شود، نه ترتیب row (README §7: «hitهای
        # نهایی به ترتیب امتیاز»). وگرنه hit رتبهٔ ۱ از context می‌افتد فقط چون
        # ردیفش بعد از hitهای ضعیف‌تر است و بودجه قبل از رسیدن به او تمام شده.
        best_of_row: dict[int, float] = {}
        for h in hits:
            if h.score > best_of_row.get(h.row, 0.0):
                best_of_row[h.row] = h.score
        ranges.sort(key=lambda rng: (-max(best_of_row.get(r, 0.0) for r in rng), rng[0]))

        passages: list[Passage] = []
        used = 0
        seen_sha1: set[str] = set()            # exact-duplicate chunks collapse (README §7)
        ov = int(self.cfg.chunk.get("overlap_sentences", 1))
        for rng in ranges:
            texts: list[str] = []
            rows: list[int] = []
            prev_sents: list[str] = []
            for r in rng:
                ch = self._chunks[r]
                sha = ch.get("sha1")
                if sha and sha in seen_sha1:
                    continue
                if sha:
                    seen_sha1.add(sha)
                body = self._chunk_index_text(ch).split("\n", 1)[-1]
                sents = _split_sents(body)
                if prev_sents and ov:
                    # chunks carry `overlap_sentences` carried sentences from the
                    # previous chunk -> drop the repeated prefix when merging
                    for j in range(min(ov, len(prev_sents)), 0, -1):
                        if len(sents) > j and prev_sents[-j:] == sents[:j]:
                            sents = sents[j:]
                            break
                if not sents:
                    continue
                texts.append(" ".join(sents))
                rows.append(r)
                prev_sents = sents
            if not rows:
                continue
            text = " ".join(texts)
            ntok = sum(self._chunks[r].get("ntok", 0) for r in rows)
            if used + ntok > budget and passages:
                continue        # جای نیست؛ range ضعیف‌تر بعدی (شاید کوچک‌تر) امتحان شود
            used += ntok
            first = self._chunks[rows[0]]
            last = self._chunks[rows[-1]]
            pages = first["l0"] if first["l0"] == last["l1"] else f"{first['l0']}–{last['l1']}"
            best = max((h for h in hits if h.row in rows), key=lambda h: h.score, default=None)
            passages.append(Passage(n=0, book_title=self._book_title(first.get("key", "")),
                                    pages=pages, pdf_pages=(first["p0"], last["p1"]),
                                    text=text, rows=rows, score=best.score if best else 0.0))
        # most-relevant passage first (row order would bury the answer, since
        # neighbors expand rows downward)
        passages.sort(key=lambda p: -p.score)
        for n, p in enumerate(passages, start=1):
            p.n = n
        return passages

    def _same_sec(self, a: int, b: int) -> bool:
        ca, cb = self._chunks[a], self._chunks[b]
        return ca.get("key") == cb.get("key") and ca.get("sec") == cb.get("sec")

    def format_context(self, passages: list[Passage]) -> str:
        """Numbered prompt context: '[n] Title, pp. X–Y' + text (Qwen should cite [n])."""
        blocks = []
        for p in passages:
            blocks.append(f"[{p.n}] «{p.book_title}»، ص {p.pages}\n{p.text}")
        return "\n\n".join(blocks)

    def warmup(self) -> None:
        """One dummy query: builds caches (CUDA kernels, LRU) so first real query is fast."""
        if self._dense is not None:
            self.retrieve("آماده‌سازی")


# ----------------------------------------------------------------------------
# CLI:  python search.py "سؤال"   |   --repl   |   --json
# ----------------------------------------------------------------------------
def main() -> None:
    # وقتی stdout به pipe وصل است، ویندوز روی cp1252 می‌افتد و چاپ فارسی کرش می‌کند
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="BookSearch-FA hybrid search (stage 1)")
    ap.add_argument("query", nargs="*", help="the question, in quotes")
    ap.add_argument("--index", default="index")
    ap.add_argument("--top", type=int, default=5, help="how many hits to print (default 5)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--repl", action="store_true", help="interactive loop")
    args = ap.parse_args()

    searcher = Searcher(index_dir=args.index)

    def one(q: str) -> None:
        # a sync that ran while this process was open swaps in the new generation
        searcher.reload_if_changed()
        # attach the query encoder on first use (lazy: Searcher works BM25-only without it)
        if searcher._dense is not None and not hasattr(searcher, "embedder"):
            searcher.embedder = searcher._make_embedder()  # loads the index's model once
        res = searcher.search(q)
        if args.json:
            print(json.dumps({"query": res.query,
                              "found": res.found,
                              "timings_ms": res.timings_ms,
                              "hits": [h.__dict__ for h in res.hits[:args.top]],
                              "passages": [p.__dict__ for p in res.passages]},
                             ensure_ascii=False, indent=2, default=str))
            return
        print(f"\nسؤال: {q}")
        print(f"زمان: {res.timings_ms}")
        if not res.found:
            print("نتیجه‌ای در کتاب‌ها یافت نشد.")
            return
        for p in res.passages[:args.top]:
            preview = p.text[:220].replace("\n", " ")
            print(f"\n{p.n}. «{p.book_title}» ص {p.pages}  (score={p.score:.4f}, rows={p.rows})")
            print(f"   {preview}…")
        if not res.passages:
            print("نتیجه‌ای پیدا نشد.")

    if args.repl:
        print("REPL — سؤال بنویسید، Ctrl+C برای خروج")
        searcher.warmup()
        while True:
            try:
                q = input("\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if q:
                one(q)
    else:
        q = " ".join(args.query)
        if not q:
            ap.error("a query is required (or use --repl)")
        one(q)


if __name__ == "__main__":
    main()
