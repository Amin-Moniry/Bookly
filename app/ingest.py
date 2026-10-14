"""
ingest.py — index-side of BookSearch-FA (stage 1: one or more books, no OCR).

Pipeline per book:
  PyMuPDF extract -> per-page quality stats -> repeated header/footer removal
  -> printed page labels -> normalize_display -> chunking (~300 tokens, never
  crossing a chapter) -> embedding (needs torch+sentence-transformers; with
  `sync --no-embed` you get a BM25-only index without them).

Incremental: each book is cached under index/books/<sha256[:16]>/ keyed by file
content hash; BM25 is global and rebuilt in seconds; CURRENT swaps atomically.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# پیشرفت/خطاها متن فارسی دارند؛ خروجی redirect‌شده روی ویندوز cp1252 است و بدون این،
# همان یک print وسط sync کل پروسه را می‌کشد (UnicodeEncodeError)
for _s in (sys.stdout, sys.stderr):
    if getattr(_s, "encoding", "") and _s.encoding.lower() not in ("utf-8", "utf8"):
        try:
            _s.reconfigure(errors="replace")
        except Exception:
            pass

try:                       # فقط برای OCR لازم است؛ query process هرگز آن را import نمی‌کند
    from PIL import Image
except ImportError:
    Image = None

from search import (BM25_TOKEN_VERSION, NORM_VERSION, load_config,
                    normalize_display, tokenize_bm25)

EXTRACT_VERSION = "ex-2"      # bump if extraction changes (invalidates the extract stage)
CHUNK_VERSION = "ch-1"        # bump if chunking changes (invalidates chunk+embed)

try:                       # new name since PyMuPDF 1.24, old alias before
    import pymupdf as fitz
except ImportError:
    import fitz


def _utf8_console() -> None:
    """Windows consoles default to cp1256/cp1252; force UTF-8 so Persian prints."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_atomic(path: Path, data: str | bytes) -> None:
    """Write to a .tmp file first, then os.replace -> readers never see half files."""
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(data, bytes):
        tmp.write_bytes(data)
    else:
        tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    _write_atomic(path, "\n".join(json.dumps(r, ensure_ascii=False) for r in rows))


# ----------------------------------------------------------------------------
# Library scan (incremental by size/mtime, sha256 only when those changed)
# ----------------------------------------------------------------------------
@dataclass
class BookFile:
    rel: str          # path relative to library/, e.g. "history/tarikh.pdf"
    path: Path
    size: int
    mtime_ns: int
    sha256: str
    key: str          # sha256[:16] -> cache dir name


def scan_library(lib: Path, catalog: dict) -> list[BookFile]:
    """Every PDF under library/ (subfolders allowed). sha256 is reused from the
    catalog when size+mtime are unchanged, so a second sync hashes nothing."""
    out: list[BookFile] = []
    for p in sorted(lib.rglob("*.pdf")):
        if p.is_file():
            rel = p.relative_to(lib).as_posix()
            st = p.stat()
            old = catalog.get(rel)
            if old and old.get("size") == st.st_size and old.get("mtime_ns") == st.st_mtime_ns:
                sha = old["sha256"]
            else:
                sha = sha256_file(p)
            out.append(BookFile(rel, p, st.st_size, st.st_mtime_ns, sha, sha[:16]))
    return out


# ----------------------------------------------------------------------------
# Extraction, quality, labels
# ----------------------------------------------------------------------------
@dataclass
class Page:
    i: int                      # 0-based PDF page index
    label: str                  # printed page number (for citations)
    method: str = "text"        # text | ocr (ocr = stage 4)
    q: dict = field(default_factory=dict)
    head: str = ""
    foot: str = ""
    foot_raw: str = ""          # آخرین خط لایهٔ متنیِ «قبل از OCR» — شاهد شمارهٔ صفحه
    text: str = ""              # normalize_display text, header/footer removed


def page_quality(text: str, fa_words: frozenset[str] | None = None) -> dict:
    """Cheap per-page stats. With fa_words, also lex_hit (fraction of Persian
    tokens present in the dictionary) and reversed_hit (visual-order detection,
    README §13.2) — the two signals the OCR gate needs beyond chars/bad."""
    text = text or ""
    letters = re.findall(r"[A-Za-z\u0600-\u06ff]", text)
    arabic = sum(1 for c in letters if "\u0600" <= c <= "\u06ff")
    script_ratio = arabic / len(letters) if letters else 0.0
    bad = sum(1 for c in text if "\ufffd" == c or "\u4e00" <= c <= "\u9fff"
              or "\ue000" <= c <= "\uf8ff" or "\ufb50" <= c <= "\ufeff")
    q = {"chars": len(text), "script": round(script_ratio, 3),
         "bad": round(bad / max(1, len(text)), 4)}
    if fa_words:
        fa_toks = [t for t in tokenize_bm25(text)
                   if any("\u0600" <= c <= "\u06ff" for c in t)]
        rev_toks = [w[::-1] for w in fa_toks]
        if fa_toks:
            q["lex"] = round(sum(1 for t in fa_toks if t in fa_words) / len(fa_toks), 3)
            q["rev"] = round(sum(1 for w in rev_toks if w in fa_words) / len(rev_toks), 3)
        else:
            q["lex"], q["rev"] = 1.0, 0.0      # no Persian tokens: nothing to fail on
        q["fa_toks"] = len(fa_toks)
    return q


# ----------------------------------------------------------------------------
# OCR (README §1.1: only pages the quality gate rejects; README §11/§12: a
# process per core with OMP_THREAD_LIMIT=1, 300 DPI render, tesseract_best)
# ----------------------------------------------------------------------------
_FA_WORDS_PATH = Path(__file__).parent / "fa_words.txt"
_FA_WORDS: frozenset[str] | None = None


def _fa_words() -> frozenset[str]:
    global _FA_WORDS
    if _FA_WORDS is None:
        try:
            _FA_WORDS = frozenset(line.strip() for line in
                                  _FA_WORDS_PATH.read_text(encoding="utf-8").splitlines()
                                  if line.strip() and not line.startswith("#"))
        except OSError:
            _FA_WORDS = frozenset()          # no dictionary -> lex/rev stay neutral
    return _FA_WORDS


def _parse_page_ranges(spec: str | None) -> set[int]:
    """"pages = "5,9-12"" (PDF indices) -> {5, 9, 10, 11, 12}."""
    out: set[int] = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


_TESSERACT_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
)


def _resolve_tesseract_cmd(cfg) -> str:
    """مسیر باینری tesseract؛ اولویت: config، مسیرهای استاندارد ویندوز، در نهایت PATH."""
    cmd = (getattr(cfg.extract, "tesseract_cmd", "") or "").strip()
    if not cmd:                       # مسیرهای استاندارد ویندوز را امتحان کن
        for cand in _TESSERACT_CANDIDATES:
            if Path(cand).exists():
                cmd = cand
                break
    if cmd and not Path(cmd).exists():
        cmd = ""                      # config کهنه/غلط → به PATH تکیه کن
    return cmd


def _resolve_tessdata_prefix(cfg) -> str:
    """مسیر «خام» پوشهٔ tessdata (بدون هیچ پرچمی).
    TESSDATA_PREFIX باید دقیقاً همین مسیر باشد؛ رشتهٔ پرچم (--tessdata-dir …)
    داخل env var باعث می‌شود tesseract دنبال «--tessdata-dir X/fas.traineddata» بگردد."""
    prefix = (getattr(cfg.extract, "tessdata_prefix", "") or "").strip()
    if not prefix:
        default_dir = Path(__file__).parent / "tessdata"
        prefix = str(default_dir) if default_dir.exists() else ""
    if prefix and not Path(prefix).is_absolute():
        prefix = str((Path(__file__).parent / prefix).resolve())   # tesseract مسیر نسبی نمی‌فهمد
    return prefix


def _setup_tesseract(cfg) -> object:
    """Import pytesseract lazily and configure binary/tessdata paths.
    Returns the pytesseract module. Raises RuntimeError with install hints."""
    try:
        import pytesseract
    except ImportError as e:
        raise RuntimeError("pytesseract نصب نیست: pip install pytesseract pillow") from e
    cmd = _resolve_tesseract_cmd(cfg)
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    elif not _tesseract_on_path(pytesseract):
        raise RuntimeError(
            "tesseract پیدا نشد. نصب: winget install UB-Mannheim.TesseractOCR "
            "(و یا [extract].tesseract_cmd را در booksearch.toml تنظیم کنید)")
    prefix = _resolve_tessdata_prefix(cfg)
    if prefix:
        os.environ["TESSDATA_PREFIX"] = prefix      # env var با مسیر خام = راه مطمئن در ویندوز
        pytesseract.pytesseract.tessdata_dir_config = f"--tessdata-dir {prefix}"
    return pytesseract


def _tesseract_on_path(pytesseract) -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_one_page(args: tuple) -> tuple[int, str]:
    """Render one page at cfg-dpi and OCR it. Module-level so ProcessPool can pickle it.
    Every import happens inside the worker: the query process must never pay for them."""
    (pdf_path, idx, dpi, lang, cmd, prefix, cache_path) = args
    import pytesseract
    try:
        import pymupdf as _fitz
    except ImportError:
        import fitz as _fitz
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    # TESSDATA_PREFIX = مسیر «خام» (نه رشتهٔ پرچم — ببین _resolve_tessdata_prefix)؛
    # بدون آن tesseract فایل fas را از مسیر نصب خودش می‌خواهد نه از tessdata پروژه
    if prefix:
        os.environ["TESSDATA_PREFIX"] = prefix
        pytesseract.pytesseract.tessdata_dir_config = f"--tessdata-dir {prefix}"
    os.environ["OMP_THREAD_LIMIT"] = "1"      # one process per core, no thread thrash (README §12)
    # کش دیسکی: اگر این صفحه قبلاً OCR شده، از کش بخوان (ری‌استارت sync از دست نمی‌دهد)
    if cache_path:
        try:
            cached = Path(cache_path).read_text(encoding="utf-8")
            if cached:
                return idx, cached
        except OSError:
            pass
    doc = _fitz.open(pdf_path)
    try:
        pix = doc[idx].get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang=lang)
    finally:
        doc.close()
    if cache_path and text:
        try:
            tmp = cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, cache_path)
        except OSError:
            pass
    return idx, text


def ocr_pages(pdf: Path, idxs: list[int], cfg) -> dict[int, str]:
    """OCR only the pages that failed the quality gate. Serial, or a process per
    core when [extract].workers > 1 (each worker: OMP_THREAD_LIMIT=1)."""
    if not idxs:
        return {}
    pytesseract = _setup_tesseract(cfg)
    # workerها مسیر خام می‌گیرند؛ خواندن از tessdata_dir_config رشتهٔ پرچم را هم می‌آورد
    cmd = _resolve_tesseract_cmd(cfg) or str(pytesseract.pytesseract.tesseract_cmd or "")
    prefix = _resolve_tessdata_prefix(cfg)
    # کش صفحه‌ای OCR: کلید = sha1 محتوای PDF + شماره صفحه؛ ری‌استارت از وسط کار ادامه می‌دهد
    cache_dir = None
    try:
        import hashlib as _hl
        _h = _hl.sha1()
        with open(pdf, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                _h.update(chunk)
        cache_dir = Path(cfg.paths["index"]) / "ocr_cache" / _h.hexdigest()[:16]
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        cache_dir = None
    jobs = [(str(pdf), i, int(cfg.extract.ocr_dpi), cfg.extract.ocr_lang, cmd, prefix,
             str(cache_dir / f"{i}.txt") if cache_dir is not None else None)
            for i in idxs]
    workers = int(getattr(cfg.extract, "workers", 0) or 0)
    t0 = time.perf_counter()
    if workers and workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_ocr_one_page, j) for j in jobs]
            results = []
            for n, fut in enumerate(as_completed(futs)):
                results.append(fut.result())
                if (n + 1) % 20 == 0 or (n + 1) == len(jobs):
                    print(f"  [ocr] {pdf.name}: {n + 1}/{len(jobs)} صفحه ({workers} worker)", flush=True)
    else:
        # پیشرفت زنده: هر ۲۰ صفحه یک خط (OCR روی CPU کند است)
        results = []
        for j, job in enumerate(jobs):
            if j and j % 20 == 0:
                print(f"  [ocr] {Path(job[0]).name}: {j}/{len(jobs)} صفحه", flush=True)
            results.append(_ocr_one_page(job))
    print(f"  OCR: {len(idxs)} صفحه در {time.perf_counter() - t0:.1f}s "
          f"({workers if workers > 1 else 1} worker)")
    return dict(results)


def needs_ocr(q: dict, gate: dict) -> bool:
    """README §3.2: a page goes to OCR when its text layer fails the gate.
    زبان‌آگاه: صفحهٔ انگلیسیِ سالم (هیچ توکن فارسی، حروف عربی نزدیک صفر) OCR
    نمی‌خواهد؛ صفحهٔ فارسیِ خراب (حروف عربی زیاد اما صفر واژهٔ معتبر) می‌خواهد."""
    if not q or q.get("chars", 0) < int(gate.get("min_chars", 150)):
        return True
    if q.get("bad", 0.0) > float(gate.get("max_bad_ratio", 0.02)):
        return True
    script = q.get("script", 0.0)
    lex, rev = q.get("lex"), q.get("rev")
    if lex is None:                     # بدون واژگان: رفتار قدیمی روی script_ratio
        return script < float(gate.get("min_script_ratio", 0.6))
    if q.get("fa_toks", 0) == 0:
        # حروف عربی زیاد اما هیچ واژهٔ معتبری → متن خراب → OCR؛
        # صفحهٔ انگلیسی (script≈0) → سالم، بدون OCR
        return script >= float(gate.get("min_script_ratio", 0.6))
    if lex < float(gate.get("min_lex_hit", 0.3)):
        return True
    if rev is not None and rev > lex + 0.15:
        return True                     # visual order: reverse beats dictionary
    return False


def _strip_running_lines(pages: list[Page]) -> None:
    """A line that repeats on >=30% of pages near top/bottom is a running
    header/footer -> remove from text, remember as head/foot."""
    n = len(pages)
    if n < 8:
        return
    def _mask(s: str) -> str:            # "Copyright ... 101" -> "Copyright ... #"
        return re.sub(r"\d{1,4}\s*$", "#", s)

    tops, bots, bots_masked = Counter(), Counter(), Counter()
    for pg in pages:
        lines = [l.strip() for l in pg.text.split("\n") if l.strip()]
        # look at the 3 outermost lines on each side: running headers/footers
        # (and page numbers) are not always the single first/last line
        for l in lines[:3]:
            if len(l) < 80:
                tops[l] += 1
        for l in lines[-3:]:
            if len(l) < 80:
                bots[l] += 1
                bots_masked[_mask(l)] += 1   # same footer, different page number
        # صفحهٔ OCRشده متنش عوض شده ولی لایهٔ متنی قبلی‌اش اغلب شمارهٔ footer را
        # دارد — به‌عنوان رأی اضافه وارد می‌شود (لایهٔ بد از رأی غالب زنده می‌ماند)
        if pg.method == "ocr" and pg.foot_raw and pg.foot_raw not in lines[-3:]:
            bots[pg.foot_raw] += 1
            bots_masked[_mask(pg.foot_raw)] += 1
    thr = max(3, int(0.3 * n))
    heads = {s for s, c in tops.items() if c >= thr}
    feet = {s for s, c in bots.items() if c >= thr}
    feet_masked = {m for m, c in bots_masked.items() if c >= thr}
    feet |= {s for s in bots if re.fullmatch(r"\d{1,4}", _to_ascii_digits(s))}   # bare page numbers
    heads |= {s for s in tops if re.fullmatch(r"\d{1,4}", _to_ascii_digits(s))}
    for pg in pages:
        lines = [l.strip() for l in pg.text.split("\n") if l.strip()]
        keep = lines
        for _ in range(2):                   # e.g. decoration line + copyright line
            if keep and (keep[0] in heads
                         or re.fullmatch(r"\d{1,4}", _to_ascii_digits(keep[0]))):
                pg.head = pg.head or keep.pop(0)
                continue
            break
        for _ in range(2):
            if keep and (keep[-1] in feet or _mask(keep[-1]) in feet_masked):
                pg.foot = keep.pop(-1)       # keep the last one: usually has the number
                continue
            break
        pg.text = "\n".join(keep)


_DIGITS_TR = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_NUM_RE = re.compile(r"(\d{1,4})")


def _to_ascii_digits(s: str) -> str:
    return s.translate(_DIGITS_TR)


def detect_printed_labels(doc, pages: list[Page], opts: dict) -> str:
    """Fill page.label. Priority (README §7): explicit PDF labels -> number found
    in the repeated footer line (dominant offset) -> books.toml page_offset ->
    i+1 (marked as pdf_index, never silently trusted)."""
    n = len(pages)

    # 1) PDF page labels, if the file defines non-trivial ones
    try:
        labels = [doc[i].get_label() for i in range(n)]
        if any(l for l in labels) and not all(l == str(i + 1) for i, l in enumerate(labels)):
            for pg, l in zip(pages, labels):
                pg.label = _to_ascii_digits(l or "")
            return "pdf_labels"
    except Exception:
        pass

    if "page_offset" in opts:                                   # manual fallback
        off = int(opts["page_offset"])
        for pg in pages:
            pg.label = str(pg.i + 1 + off)
        return "books.toml"

    # 2) dominant offset from footer numbers  (label = i + 1 + offset)
    offs = []
    for pg in pages:
        foot = pg.foot or (pg.foot_raw if pg.method == "ocr" else "")
        if foot:
            nums = _NUM_RE.findall(_to_ascii_digits(foot))
            if nums:
                v = int(nums[-1])        # LAST number: footers often carry a year first
                if 0 < v < 3000:
                    offs.append(v - (pg.i + 1))
    if len(offs) >= max(5, n // 5):                             # need real evidence
        off = Counter(offs).most_common(1)[0][0]
        for pg in pages:
            pg.label = str(pg.i + 1 + off)
        return "footer"

    for pg in pages:                                            # last resort
        pg.label = str(pg.i + 1)
    return "pdf_index"


def _parse_skip(spec: str) -> set[int]:
    """"skip_pages = "0-13,40"" -> {0..13, 40} (covers, imprint, TOC)."""
    out: set[int] = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def extract_pages(pdf: Path, opts: dict, cfg=None) -> tuple[list[Page], dict]:
    """Text of every page with get_text(sort=True) (geometric order, best for
    two-column Persian), then normalize_display + header/footer strip + labels.
    Pages that fail the quality gate go through OCR (ex-2); pages listed in
    opts["skip_pages"] (books.toml) are left out entirely."""
    doc = fitz.open(pdf)
    t0 = time.perf_counter()
    skip = _parse_skip(opts.get("skip_pages", ""))
    fa_words = _fa_words() or None
    pages: list[Page] = []
    index_map: list[int] = []               # pages[idx_in_pages] = PDF page index
    _n_all = doc.page_count
    for i in range(_n_all):
        if i in skip:
            continue
        if _n_all > 200 and i and i % 100 == 0:
            print(f"  [extract] {pdf.name}: صفحه {i}/{_n_all}", flush=True)
        raw = doc[i].get_text("text", sort=True)
        text = normalize_display(raw)
        pages.append(Page(i=i, label="", q=page_quality(text, fa_words), text=text))
        index_map.append(i)
    doc.close()

    ex_cfg = getattr(cfg, "extract", None) if cfg is not None else None
    engine = getattr(ex_cfg, "ocr_engine", "none") if ex_cfg else "none"
    force_all = bool(getattr(ex_cfg, "force_ocr", False)) or bool(opts.get("force_ocr", False))
    force_pages = set() if force_all else _parse_page_ranges(opts.get("force_pages", ""))
    if engine == "tesseract":
        gate = dict(getattr(ex_cfg, "gate", {}) or {})
        bad = [pg for pg in pages
               if force_all or pg.i in force_pages or needs_ocr(pg.q, gate)]
        if bad:
            texts = ocr_pages(pdf, [pg.i for pg in bad], cfg)
            for pg in pages:
                if pg.i not in texts:
                    continue
                t = normalize_display(texts[pg.i])
                if t:
                    # هر صفحه‌ای که به OCR رسیده یا force است یا gate را رد کرده
                    # (لایهٔ متنی‌اش نامعتبر است)؛ پس هر خروجی غیرخالی OCR جایگزین
                    # می‌شود — حتی اگر کوتاه‌تر باشد. guard «طول بیشتر» فقط برای
                    # لایهٔ سالم معنا می‌داشت که هرگز OCR نمی‌گیرد.
                    # آخرین خط لایهٔ قبلی معمولاً شمارهٔ صفحه است؛ برای رأی footer ذخیره
                    raw_lines = [l.strip() for l in pg.text.split("\n") if l.strip()]
                    if raw_lines:
                        pg.foot_raw = raw_lines[-1]
                    pg.text = t
                    pg.q = page_quality(t, fa_words)
                    pg.method = "ocr"
                elif not pg.text.strip():
                    pg.method = "ocr"            # هیچ‌کدام متنی ندارند (صفحهٔ واقعاً خالی)
    _strip_running_lines(pages)
    doc = fitz.open(pdf)
    source = detect_printed_labels(doc, pages, opts)
    seconds = time.perf_counter() - t0
    doc.close()
    return pages, {"source": source, "seconds": seconds, "skipped": len(skip),
                   "ocr": [pg.i for pg in pages if pg.method == "ocr"]}


# ----------------------------------------------------------------------------
# Structure + chunking
# ----------------------------------------------------------------------------
@dataclass
class Heading:
    title: str
    page: int          # 0-based PDF page where the chapter starts


_CHAPTER_RE = re.compile(r"^\s*(فصل|بخش|گفتار|مبحث|chapter|section)\b\S?\s*.{0,80}$",
                         flags=re.IGNORECASE)


def read_structure(pdf: Path, n_pages: int) -> list[Heading]:
    """PDF outline (TOC) if present, else regex on short lines like «فصل چهارم: …»."""
    doc = fitz.open(pdf)
    heads: list[Heading] = []
    try:
        for lvl, title, page in doc.get_toc(simple=True):
            if page >= 1:
                heads.append(Heading(normalize_display(title), page - 1))
    except Exception:
        heads = []
    doc.close()
    if heads:
        return heads
    doc = fitz.open(pdf)
    for i in range(n_pages):
        for line in doc[i].get_text("text", sort=True).splitlines()[:6]:
            line = line.strip()
            if 4 < len(line) < 90 and _CHAPTER_RE.match(normalize_display(line)):
                heads.append(Heading(normalize_display(line), i))
                break
    doc.close()
    return heads


def _split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences on Persian/latin enders (. ؟ ! ؛)."""
    parts = re.split(r"(?<=[.؟!؛])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _ntok(text: str) -> int:
    """Cheap token estimate: ~4 chars per token, rounded up. One estimator is
    used everywhere (chunk sizing); the real embedding tokenizer is only needed
    in the heavy pipeline."""
    return max(1, (len(text) + 3) // 4)


def _split_long(sentence: str, max_tokens: int) -> list[str]:
    words, out, cur, n = sentence.split(), [], [], 0
    for w in words:
        wl = (len(w) + 1 + 3) // 4          # word + following space, ceil
        if cur and n + wl > max_tokens:
            out.append(" ".join(cur)); cur, n = [], 0
        cur.append(w); n += wl
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_book(pages: list[Page], headings: list[Heading], cfg) -> list[dict]:
    """Pack paragraphs into ~300-token chunks (max 420), 1-sentence overlap,
    never crossing a chapter border. `sec` = running section number, used at
    query time to expand neighbors safely."""
    target = int(cfg.chunk.get("target_tokens", 300))
    max_tok = int(cfg.chunk.get("max_tokens", 420))
    overlap = int(cfg.chunk.get("overlap_sentences", 1))

    heads = sorted(headings, key=lambda h: h.page)
    if not heads:  # no structure found -> artificial section every 10 pages
        heads = [Heading("", p) for p in range(0, len(pages), 10)]

    # split pages into (chapter_title, [pages]) blocks
    blocks: list[tuple[str, list[Page]]] = []
    for idx, h in enumerate(heads):
        start = h.page
        end = heads[idx + 1].page if idx + 1 < len(heads) else len(pages)
        if start >= len(pages):
            continue
        pages_in = [pg for pg in pages if start <= pg.i < max(end, start + 1)]
        if pages_in and any(pg.text.strip() for pg in pages_in):
            blocks.append((h.title, pages_in))

    label_of = {pg.i: pg.label for pg in pages}
    chunks: list[dict] = []
    sec = 0
    for title, pg_list in blocks:
        sents: list[tuple[str, int]] = []          # (sentence, pdf_page)
        for pg in pg_list:
            for para in pg.text.split("\n"):
                para = para.strip()
                if len(para) < 2:
                    continue
                for s in _split_sentences(para):
                    for piece in ([s] if _ntok(s) <= max_tok else _split_long(s, max_tok)):
                        sents.append((piece, pg.i))
        sec += 1
        cur: list[tuple[str, int]] = []
        cur_n = 0

        def flush() -> None:
            nonlocal cur, cur_n
            if not cur:
                return
            text = " ".join(s for s, _ in cur)
            chunks.append({
                "seq": len(chunks), "p0": cur[0][1], "p1": cur[-1][1],
                "l0": label_of[cur[0][1]], "l1": label_of[cur[-1][1]],
                "chapter": title, "sec": sec, "ntok": _ntok(text), "text": text,
            })
            carry = cur[-overlap:] if overlap and len(cur) > overlap else []
            cur, cur_n = list(carry), sum(_ntok(s) for s, _ in carry)

        for s, pg_i in sents:
            st = _ntok(s)
            if cur and cur_n + st > target:
                flush()
            cur.append((s, pg_i)); cur_n += st
        flush()
    return chunks


# ----------------------------------------------------------------------------
# Per-book processing + cache
# ----------------------------------------------------------------------------
def _opts_for(rel: str, lib: Path) -> dict:
    """Optional library/books.toml exceptions: title, page_offset."""
    p = lib / "books.toml"
    if p.exists():
        import tomllib
        try:
            return tomllib.loads(p.read_text(encoding="utf-8")).get(rel, {})
        except Exception:
            pass
    return {}


def _extract_version(cfg=None) -> str:
    # نرمال‌سازی و gate/OCR در مرحلهٔ extract اعمال می‌شوند → هر دو نسخه را شامل می‌شود.
    # بدون config یعنی config پیش‌فرض پروژه (همان که sync بدون override می‌سازد) —
    # نه «هیچ OCR»؛ وگرنه نسخهٔ meta هرگز با مقایسهٔ بدون‌آرگومان جور نمی‌شود.
    if cfg is None:
        cfg = load_config()
    ex = getattr(cfg, "extract", None)
    gate = getattr(ex, "gate", None)
    ocr = getattr(ex, "ocr_engine", "none")
    g = ""
    if gate and ocr != "none":       # gate فقط وقتی OCR فعال است معنا دارد
        g = (f"|{gate.get('min_chars')}c/{gate.get('min_script_ratio')}s/"
             f"{gate.get('max_bad_ratio')}b/{gate.get('min_lex_hit')}l")
    return f"{EXTRACT_VERSION}|{NORM_VERSION}|ocr-{ocr}{g}"


def _embed_version(cfg, has_embed: bool) -> str:
    if not has_embed:
        return "none"
    backend = "m3" if str(getattr(cfg.embed, "backend", "st")).lower() == "m3" else "st"
    return (f"{NORM_VERSION}|{cfg.embed.model}|{backend}|{cfg.embed.max_length}"
            f"|hdr{int(cfg.chunk.get('header', True))}")


def _chunk_version(cfg) -> str:
    return (f"{NORM_VERSION}|{CHUNK_VERSION}|{cfg.chunk.get('target_tokens')}/{cfg.chunk.get('max_tokens')}"
            f"/{cfg.chunk.get('overlap_sentences')}|hdr{int(cfg.chunk.get('header', True))}")


def _index_texts(chunks: list[dict], meta_title: str, cfg) -> list[str]:
    """متن index‌شدهٔ هر تکه: header (عنوان کتاب + فصل) + متن — مشترک بین dense و sparse."""
    texts = []
    for c in chunks:
        header = ""
        if cfg.chunk.get("header", True):
            header = meta_title + ("\u203a " + c["chapter"] if c.get("chapter") else "") + "\n"
        texts.append(header + c["text"])
    return texts


def _embed_chunks(chunks: list[dict], meta_title: str, cfg, embedder) -> np.ndarray:
    return embedder.encode_docs(_index_texts(chunks, meta_title, cfg))   # [n, dim] float32, L2-normalized


def _sparse_rows(chunks: list[dict], meta_title: str, cfg, embedder) -> list[dict[str, float]]:
    """C1: lexical weights هر تکه روی همان متن index‌شده (header + text)."""
    return embedder.encode_docs_sparse(_index_texts(chunks, meta_title, cfg))


def _write_sparse_npz(path: Path, rows: list[dict[str, float]]) -> None:
    """CSR از lexical weights. فرمت: vocab (رشته‌ای، مرتب برای پایداری)، indptr/indices/data
    (float32)، shape [n_rows, n_cols]. خواندن‌اش مالک _SparseIndex در search.py است."""
    from scipy.sparse import csr_matrix
    vocab = sorted({t for lw in rows for t in lw})
    pos = {t: i for i, t in enumerate(vocab)}
    mat = csr_matrix((len(rows), len(vocab)), dtype=np.float32)
    for r, lw in enumerate(rows):
        if not lw:
            continue
        cols = [pos[t] for t in lw if t in pos]
        vals = [float(lw[t]) for t in lw if t in pos]
        mat[r, cols] = vals
    mat.sort_indices()
    buf = path.with_name(path.name + ".tmp")
    with open(buf, "wb") as f:               # file handle → np.savez پسوند اضافه نمی‌کند
        np.savez(f, vocab=np.array(vocab, dtype=np.str_),
                 indptr=mat.indptr.astype(np.int64), indices=mat.indices.astype(np.int32),
                 data=mat.data.astype(np.float32), shape=np.array(mat.shape, dtype=np.int64))
    os.replace(buf, path)


def _finish_book(bf: BookFile, cfg, pages: list[Page], chunks: list[dict],
                 embedder, title: str, extract_info: dict, t_chunk: float,
                 keep_dense: bool = False) -> dict:
    """Write the cache dir for one book (meta.json LAST as the commit marker).
    Shared by the full pipeline and the stage-wise `--force chunk|embed` path.
    keep_dense=True: dense.npy سالم از اجرای قبل می‌ماند (re-embed نشده) — فقط
    وقتی که تماس‌دهنده مطمئن است متن تکه‌ها دقیقاً همان قبلی است."""
    index = Path(cfg.paths["index"])
    kdir = index / "books" / bf.key
    kdir.mkdir(parents=True, exist_ok=True)

    # C1: sparse (lexical weights بge-m3) — فقط با backend m3؛ یک‌پاسه با dense در بلوک embed
    _sparse_wanted = (embedder is not None and chunks
                      and str(getattr(cfg.embed, "backend", "st")).lower() == "m3"
                      and bool(getattr(cfg.embed, "sparse", True))
                      and hasattr(embedder, "encode_docs_sparse"))
    sparse_rows: list[dict[str, float]] | None = None

    for c in chunks:
        c["key"] = bf.key
        c["id"] = f"{bf.key}:{c['seq']:05d}"
        c["sha1"] = hashlib.sha1(c["text"].encode("utf-8")).hexdigest()[:12]

    meta = {
        "key": bf.key, "file": bf.rel, "sha256": bf.sha256,
        "title": title,
        "lang": "fa" if np.mean([pg.q.get("script", 0) for pg in pages] or [0]) > 0.3 else "en",
        "n_pages": len(pages), "n_chunks": len(chunks),
        "label_source": extract_info["source"], "skipped_pages": extract_info.get("skipped", 0),
        "ocr_pages": extract_info.get("ocr_pages", []),
        "stages": {
            "extract": {"version": _extract_version(cfg), "ocr_pages": extract_info.get("ocr_pages", []),
                        "seconds": round(extract_info["seconds"], 1)},
            "chunk": {"version": _chunk_version(cfg), "seconds": round(t_chunk, 1)},
            "embed": {"version": _embed_version(cfg, embedder is not None),
                      "dim": 0, "sparse": bool(sparse_rows is not None), "seconds": 0.0},
        },
    }

    write_jsonl(kdir / "pages.jsonl", [
        {"i": pg.i, "label": pg.label, "method": pg.method, "q": pg.q,
         "head": pg.head, "foot": pg.foot, "foot_raw": pg.foot_raw,
         "text": pg.text} for pg in pages])

    if embedder is not None and chunks:
        t0 = time.perf_counter()
        if _sparse_wanted and hasattr(embedder, "encode_docs_both"):
            # یک پاس مشترک dense+sparse (نصف زمان دو فراخوانی جدا) + چک‌پوینت resumable:
            # هر ~۱۰۲۴ متن ذخیره می‌شود؛ اگر sync قطع شد، دوباره از همان‌جا ادامه می‌دهد.
            vecs, sparse_rows = embedder.encode_docs_both(
                _index_texts(chunks, title, cfg), ckpt_path=str(kdir / "embed_ckpt.npz"))
        else:
            vecs = _embed_chunks(chunks, title, cfg, embedder)
            if _sparse_wanted:
                sparse_rows = _sparse_rows(chunks, title, cfg, embedder)
        meta["stages"]["embed"]["dim"] = int(vecs.shape[1])
        meta["stages"]["embed"]["seconds"] = round(time.perf_counter() - t0, 1)
        buf = kdir / "dense.npy.tmp"
        with open(buf, "wb") as f:
            np.save(f, vecs.astype(np.float16))     # fp16 on disk, fp32 in RAM at query
        os.replace(buf, kdir / "dense.npy")
        # C1: sparse.npz کنار dense؛ فقط وقتی sparse_rows واقعاً ساخته شده
        if sparse_rows is not None:
            _write_sparse_npz(kdir / "sparse.npz", sparse_rows)
        (kdir / "embed_ckpt.npz").unlink(missing_ok=True)   # چک‌پوینت موقت — نهایی شد
        meta["stages"]["embed"]["sparse"] = sparse_rows is not None
    elif not (keep_dense and (kdir / "dense.npy").exists()):
        (kdir / "dense.npy").unlink(missing_ok=True)   # stale dense from an earlier embed run
        (kdir / "sparse.npz").unlink(missing_ok=True)  # sparse از مود قبلی هم بی‌صاحب است

    write_jsonl(kdir / "chunks.jsonl", chunks)
    _write_atomic(kdir / "meta.json", json.dumps(meta, ensure_ascii=False, indent=1))
    return meta


def process_book(bf: BookFile, cfg, embedder) -> dict | None:
    """Run all stages for one book from the PDF; write the cache dir; return meta.json.
    Returns None on hard failure (no meta written, old cache untouched) — e.g. OCR
    errors when the extract must not silently downgrade to a text-only index."""
    opts = _opts_for(bf.rel, Path(cfg.paths["library"]))

    t0 = time.perf_counter()
    try:
        pages, ex = extract_pages(bf.path, opts, cfg)
    except Exception as e:                    # noqa: BLE001 — a book must never kill sync
        print(f"  [error] extract failed: {e}")
        return None
    heads = read_structure(bf.path, len(pages))
    t_extract = time.perf_counter() - t0

    t0 = time.perf_counter()
    chunks = chunk_book(pages, heads, cfg)
    t_chunk = time.perf_counter() - t0

    return _finish_book(bf, cfg, pages, chunks, embedder,
                        opts.get("title") or Path(bf.rel).stem,
                        {"source": ex["source"], "seconds": t_extract, "skipped": ex.get("skipped", 0),
                         "ocr_pages": ex.get("ocr", [])},
                        t_chunk)


def _pages_from_cache(index: Path, key: str) -> list[Page]:
    rows = read_jsonl(index / "books" / key / "pages.jsonl")
    return [Page(i=r["i"], label=r.get("label", ""), method=r.get("method", "text"),
                 q=r.get("q", {}), head=r.get("head", ""), foot=r.get("foot", ""),
                 foot_raw=r.get("foot_raw", ""), text=r.get("text", "")) for r in rows]


def old_meta_embed(meta: dict) -> dict:
    """آخرین وضعیت ثبت‌شدهٔ مرحلهٔ embed یک کش (برای حفظ dense در re-extract)."""
    st = (meta or {}).get("stages", {}).get("embed", {})
    return {"version": st.get("version", "none"), "dim": st.get("dim", 0),
            "sparse": st.get("sparse", False), "seconds": st.get("seconds", 0.0)}


def reprocess_book(bf: BookFile, cfg, embedder, old_meta: dict, stage: str) -> dict:
    """--force chunk|embed (or a stale chunk/embed version): reuse the cached
    pages, redo only the stages from `stage` onward. `extract` is handled by a
    full process_book() (it may OCR); chunk/embed never re-extract."""
    index = Path(cfg.paths["index"])
    pages = _pages_from_cache(index, bf.key)
    t0 = time.perf_counter()
    if stage == "chunk":
        heads = read_structure(bf.path, len(pages))
        chunks = chunk_book(pages, heads, cfg)
    else:                       # embed: keep the existing chunks, re-embed only
        chunks = read_jsonl(index / "books" / bf.key / "chunks.jsonl")
    t_chunk = time.perf_counter() - t0
    opts = _opts_for(bf.rel, Path(cfg.paths["library"]))
    return _finish_book(bf, cfg, pages, chunks, embedder,
                        opts.get("title") or Path(bf.rel).stem,
                        {"source": old_meta.get("label_source", "pdf_index"),
                         "seconds": old_meta.get("stages", {}).get("extract", {}).get("seconds", 0.0),
                         "skipped": old_meta.get("skipped_pages", 0),
                         "ocr_pages": old_meta.get("ocr_pages", []) or
                                      old_meta.get("stages", {}).get("extract", {}).get("ocr_pages", [])},
                        t_chunk)


# ----------------------------------------------------------------------------
# Global merge: BM25 over all books + manifest, atomic CURRENT swap
# ----------------------------------------------------------------------------
def build_merged(index: Path, metas: list[dict], cfg) -> str:
    import bm25s
    merged = index / "merged"
    gen_n = 0
    if (merged / "CURRENT").exists():
        try:
            gen_n = int((merged / "CURRENT").read_text(encoding="utf-8").strip().split("-")[1])
        except Exception:
            gen_n = 0
    gen = f"gen-{gen_n + 1:06d}"
    gdir = merged / gen / "bm25"
    gdir.mkdir(parents=True, exist_ok=True)

    corpus: list[list[str]] = []
    books, row0, dim = [], 0, 0
    any_sparse = False
    for m in metas:
        chunks = read_jsonl(index / "books" / m["key"] / "chunks.jsonl")
        title = m.get("title") or Path(m["file"]).stem
        hdr = bool(cfg.chunk.get("header", True))
        for c in chunks:
            header = (title + ("\u203a " + c["chapter"] if c.get("chapter") else "") + "\n") if hdr else ""
            corpus.append(tokenize_bm25(header + c["text"]))
        books.append({"key": m["key"], "file": m["file"], "title": title,
                      "row0": row0, "n": len(chunks)})
        row0 += len(chunks)
        d = index / "books" / m["key"] / "dense.npy"
        if d.exists():
            dim = int(np.load(d, mmap_mode="r").shape[1])
        if (index / "books" / m["key"] / "sparse.npz").exists():
            any_sparse = True

    bm25 = bm25s.BM25(method="lucene", **cfg.bm25)   # k1/b belong to the constructor
    bm25.index(corpus, show_progress=False)
    bm25.save(str(gdir))

    vocab = len(getattr(bm25, "vocab_dict", None) or {})
    manifest = {
        "generation": gen_n + 1,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "norm": NORM_VERSION, "chunk": _chunk_version(cfg),
        "embed": _embed_version(cfg, dim > 0), "dim": dim, "sparse": any_sparse,
        "n_rows": row0,
        "bm25": {**cfg.bm25, "method": "lucene", "tok": BM25_TOKEN_VERSION, "vocab": vocab},
        "books": books,
    }
    _write_atomic(merged / gen / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    _write_atomic(merged / "CURRENT.tmp", gen)
    os.replace(merged / "CURRENT.tmp", merged / "CURRENT")

    # keep only the last 2 generations
    for g in sorted(merged.glob("gen-*"))[:-2]:
        import shutil
        shutil.rmtree(g, ignore_errors=True)
    return gen


# ----------------------------------------------------------------------------
# sync — the only command you normally need
# ----------------------------------------------------------------------------
def _cache_complete(index: Path, key: str, meta: dict | None) -> bool:
    """meta.json is the commit marker, but a run interrupted at the wrong moment
    (یا یک پاک‌سازی دستی) می‌تواند meta را باقی بگذارد و فایل‌های همراه را نبرد.
    Sync فقط وقتی به کش اعتماد می‌کند که هرچه Searcher/reprocess لازم دارند موجود است:
    pages.jsonl + chunks.jsonl همیشه؛ dense.npy وقتی meta ادعای embedding دارد."""
    if not meta:
        return False
    kdir = index / "books" / key
    if not (kdir / "pages.jsonl").exists() or not (kdir / "chunks.jsonl").exists():
        return False
    if meta.get("stages", {}).get("embed", {}).get("dim", 0):
        return (kdir / "dense.npy").exists()
    return True


def _stage_plan(bf: BookFile, cfg, catalog: dict, index: Path,
                force: str | None, no_embed: bool, lib: Path) -> dict:
    """تصمیم «چه کاری برای این کتاب لازم است» — قبل از هر پردازشی و بدون embedder،
    تا B1 بتواند کل batch را بچیند. skip_embed: extract کهنه شده اما dense سالم است
    (فقط OCR/استخراج دوباره؛ بدون re-embed اگر متن تکه‌ها تغییر نکند)."""
    meta_path = index / "books" / bf.key / "meta.json"
    meta = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = None
        if meta is not None and not _cache_complete(index, bf.key, meta):
            print(f"  [warn] cache incomplete for key {bf.key} (interrupted run?) "
                  f"-> reprocessing {bf.rel}")
            meta = None
    was_known = any(v.get("sha256") == bf.sha256 for v in catalog.values())
    run_stage = None
    if meta is not None and force is None:
        st = meta.get("stages", {})
        # earliest stale stage wins; everything after it is redone too
        if st.get("extract", {}).get("version") != _extract_version(cfg):
            run_stage = "extract"
        elif st.get("chunk", {}).get("version") != _chunk_version(cfg):
            run_stage = "chunk"
        elif st.get("embed", {}).get("version") != _embed_version(cfg, not no_embed):
            run_stage = "embed"
    elif force:
        run_stage = force            # --force is stage-wise: cached pages/chunks are reused
    skip_embed = False
    if run_stage == "extract" and not no_embed and meta is not None:
        old_embed_ok = (meta.get("stages", {}).get("embed", {}).get("version")
                        == _embed_version(cfg, True))
        if old_embed_ok and (index / "books" / bf.key / "dense.npy").exists():
            print(f"reprocessing {bf.rel} from stage 'extract' (dense cached, no re-embed)…")
            skip_embed = True
    return {"bf": bf, "meta_path": meta_path, "meta": meta, "was_known": was_known,
            "run_stage": run_stage, "skip_embed": skip_embed,
            "opts": _opts_for(bf.rel, lib)}


def _extract_stage_work(args: tuple) -> tuple:
    """بخش CPU-سنگینِ extract/chunk برای یک کتاب — module-level تا ProcessPool pickle کند (B1).
    خروجی: (pages, chunks, extract_info, t_chunk) یا («__error__», msg). embed اینجا نیست:
    مدل بین پروسه‌ها منتقل نمی‌شود و در پروسهٔ اصلی برای همهٔ کتاب‌ها مشترک است."""
    bf, cfg, opts, stage, old_meta = args
    if stage != "embed":
        # OCR/tesseract داخل worker تک‌رشته‌ای (README §12)؛ env والد دست نمی‌خورد
        os.environ["OMP_THREAD_LIMIT"] = "1"
    try:
        if stage == "embed" and old_meta is not None:
            # فقط re-embed: صفحات و تکه‌ها از کش، بدون هیچ extract/chunk دوباره
            index = Path(cfg.paths["index"])
            pages = _pages_from_cache(index, bf.key)
            chunks = read_jsonl(index / "books" / bf.key / "chunks.jsonl")
            st = old_meta.get("stages", {})
            info = {"source": old_meta.get("label_source", "pdf_index"),
                    "seconds": st.get("extract", {}).get("seconds", 0.0),
                    "skipped": old_meta.get("skipped_pages", 0),
                    "ocr_pages": old_meta.get("ocr_pages", []) or
                                 st.get("extract", {}).get("ocr_pages", [])}
            return (pages, chunks, info, 0.0)
        if stage == "chunk" and old_meta is not None:
            index = Path(cfg.paths["index"])
            pages = _pages_from_cache(index, bf.key)
            t0 = time.perf_counter()
            heads = read_structure(bf.path, len(pages))
            chunks = chunk_book(pages, heads, cfg)
            t_chunk = time.perf_counter() - t0
            info = {"source": old_meta.get("label_source", "pdf_index"),
                    "seconds": old_meta.get("stages", {}).get("extract", {}).get("seconds", 0.0),
                    "skipped": old_meta.get("skipped_pages", 0),
                    "ocr_pages": old_meta.get("ocr_pages", []) or
                                 old_meta.get("stages", {}).get("extract", {}).get("ocr_pages", [])}
            return (pages, chunks, info, t_chunk)
        t0 = time.perf_counter()
        pages, ex = extract_pages(bf.path, opts, cfg)
        heads = read_structure(bf.path, len(pages))
        t_extract = time.perf_counter() - t0
        t1 = time.perf_counter()
        chunks = chunk_book(pages, heads, cfg)
        t_chunk = time.perf_counter() - t1
        info = {"source": ex["source"], "seconds": t_extract,
                "skipped": ex.get("skipped", 0), "ocr_pages": ex.get("ocr", [])}
        return (pages, chunks, info, t_chunk)
    except Exception as e:                    # noqa: BLE001 — a book must never kill sync
        return ("__error__", str(e))


def sync(cfg, workers: int = 0, no_embed: bool = False, force: str | None = None) -> dict:
    lib = Path(cfg.paths["library"])
    index = Path(cfg.paths["index"])
    if not lib.exists():
        print(f"library/ not found: {lib.resolve()} — create it and drop a PDF inside.")
        return
    (index / "books").mkdir(parents=True, exist_ok=True)
    catalog_path = index / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {}

    files = scan_library(lib, catalog)
    # if the BM25 tokenizer changed, the cached chunks are fine but every BM25
    # vector is stale -> pretend everything changed so a fresh merge is built
    # (chunks/pages/dense are reused — only the cheap merge re-runs)
    # (opts برای هر کتاب در حلقهٔ زیر با _opts_for خوانده می‌شود)
    tok_stale = False
    current = index / "merged" / "CURRENT"
    if current.exists():
        try:
            gen = current.read_text(encoding="utf-8").strip()
            man = json.loads((index / "merged" / gen / "manifest.json").read_text(encoding="utf-8"))
            tok_stale = man.get("bm25", {}).get("tok") != BM25_TOKEN_VERSION
            if tok_stale:
                print(f"BM25 tokenizer changed -> rebuilding merge (tok={BM25_TOKEN_VERSION})")
        except Exception:
            tok_stale = False

    # ---- B1: فاز ۱ — برنامه‌ریزی همهٔ کتاب‌ها (بدون embedder، فقط تصمیم) ----
    metas, changed = [], False
    stats = Counter()
    stage_secs = Counter()
    plans = [_stage_plan(bf, cfg, catalog, index, force, no_embed, lib) for bf in files]
    todo = [p for p in plans if not (p["meta"] is not None and p["run_stage"] is None)]
    for p in plans:
        if p not in todo:
            meta_path, meta = p["meta_path"], p["meta"]
            renamed = meta.get("file") != p["bf"].rel
            if renamed:                                  # renamed/moved: just update path
                meta["file"] = p["bf"].rel
                _write_atomic(meta_path, json.dumps(meta, ensure_ascii=False, indent=1))
            stats["renamed" if renamed else "unchanged"] += 1
            metas.append(meta)
            catalog[p["bf"].rel] = {"size": p["bf"].size, "mtime_ns": p["bf"].mtime_ns,
                                    "sha256": p["bf"].sha256}

    # ---- فاز ۲: کار CPU-سنگین (extract/chunk) — موازی وقتی --workers > 1 ----
    work_args = [(p["bf"], cfg, p["opts"],
                  "chunk" if p["run_stage"] == "chunk" else "extract",
                  p["meta"] if p["run_stage"] == "chunk" else None) for p in todo]
    # «embed» یعنی فقط re-embed؛ extract/chunk از کش خوانده می‌شود — پارالل نمی‌خواهد
    results: list = [None] * len(todo)
    heavy = [(i, a) for i, a in enumerate(work_args) if a[3] != "embed"]
    if workers and workers > 1 and len(heavy) > 1:
        from concurrent.futures import ProcessPoolExecutor
        # OMP_THREAD_LIMIT داخل خود worker تنظیم می‌شود تا embed پروسهٔ اصلی محدود نشود
        print(f"parallel extract/chunk: {len(heavy)} کتاب روی {workers} worker…")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_extract_stage_work, a) for _, a in heavy]
            for (i, _), f in zip(heavy, futures):
                results[i] = f.result()
    else:
        for i, a in heavy:
            results[i] = _extract_stage_work(a)
    # مرحلهٔ embed-فقط (بدون extract/chunk دوباره) همان مسیر reprocess_book قدیمی است
    for i, p in enumerate(todo):
        if work_args[i][3] == "embed":
            results[i] = _extract_stage_work((p["bf"], cfg, p["opts"], "embed", p["meta"]))

    # ---- فاز ۳: embed در پروسهٔ اصلی + نوشتن کش‌ها ----
    embedder = None
    for p, res in zip(todo, results):
        bf, meta_path, opts = p["bf"], p["meta_path"], p["opts"]
        run_stage = p["run_stage"]
        if isinstance(res, tuple) and len(res) == 2 and res[0] == "__error__":
            print(f"  [error] {bf.rel}: extract/chunk failed: {res[1]}")
            old = p["meta"]
            if old is not None:
                metas.append(old)
                catalog[bf.rel] = {"size": bf.size, "mtime_ns": bf.mtime_ns, "sha256": bf.sha256}
                stats["skipped_failed"] += 1
            else:
                stats["failed"] += 1
            continue
        pages, chunks, info, t_chunk = res
        # _extract_stage_work برای stage='embed' فقط تکه‌های کش‌شده را برمی‌گرداند
        need_embedder = not no_embed and not p["skip_embed"]
        if need_embedder and embedder is None:
            from search import Embedder
            try:
                if str(getattr(cfg.embed, "backend", "st")).lower() == "m3":
                    from search import M3Embedder
                    print("loading embedding model (backend=m3، اولین بار ~2 GB دانلود)…")
                    embedder = M3Embedder(cfg.embed)
                else:
                    print("loading embedding model (first time downloads ~2 GB)…")
                    embedder = Embedder(cfg.embed)
            except RuntimeError as e:
                print(f"  [warn] {e}\n  -> building BM25-only index (no dense search)")
        keep_dense = False
        if p["skip_embed"]:
            # extract دوباره از PDF (با OCR)؛ چون dense سالم است اگر متن تکه‌ها
            # بدون تغییر بماند، dense.npy حفظ می‌شود؛ وگرنه صادقانه حذف و BM25-only
            old_chunks = read_jsonl(index / "books" / bf.key / "chunks.jsonl")
            old_texts = [(c.get("seq"), c.get("text")) for c in old_chunks]
            keep_dense = [(c.get("seq"), c.get("text")) for c in chunks] == old_texts
        meta = _finish_book(bf, cfg, pages, chunks,
                            None if (no_embed or p["skip_embed"]) else embedder,
                            opts.get("title") or Path(bf.rel).stem, info, t_chunk,
                            keep_dense=keep_dense)
        if p["skip_embed"]:
            if keep_dense:
                meta["stages"]["embed"] = {**old_meta_embed(meta), "reused": True}
                _write_atomic(meta_path, json.dumps(meta, ensure_ascii=False, indent=1))
            else:
                print("  [warn] متن تکه‌ها بعد از extract عوض شد — dense حذف شد؛ "
                      "دوباره `sync` اجرا کنید تا فقط embed دوباره ساخته شود")
                stats["needs_reembed"] += 1
        elif run_stage == "embed":
            pass                                  # reprocess مسیر معمول را رفته است
        print(f"  pages={meta['n_pages']} chunks={meta['n_chunks']} "
              f"labels={meta['label_source']} embed={'yes' if (embedder and not no_embed and not p['skip_embed']) else 'no'}")
        stats["added" if not p["was_known"] else "changed"] += 1
        changed = True
        for k, v in meta.get("stages", {}).items():
            stage_secs[k] += v.get("seconds", 0.0)
        metas.append(meta)
        catalog[bf.rel] = {"size": bf.size, "mtime_ns": bf.mtime_ns, "sha256": bf.sha256}

    # removals: cached books whose PDF no longer exists
    current_rels = {bf.rel for bf in files}
    for rel in list(catalog):
        if rel not in current_rels:
            removed_key = catalog[rel]["sha256"][:16]
            import shutil
            shutil.rmtree(index / "books" / removed_key, ignore_errors=True)
            metas = [m for m in metas if m.get("file") != rel]
            del catalog[rel]
            changed = True
            stats["removed"] += 1

    if not metas:
        current = index / "merged" / "CURRENT"
        if (changed or tok_stale) and current.exists():
            current.unlink()                     # Searcher will say: run sync
            print("library is empty — index cleared")
    elif changed or tok_stale:
        gen = build_merged(index, metas, cfg)
        n_rows = sum(m["n_chunks"] for m in metas)
        print(f"merged index -> {gen} (rows={n_rows})")
    else:
        print("nothing changed; index is up to date")
    _write_atomic(catalog_path, json.dumps(catalog, ensure_ascii=False, indent=1))
    report = {"stats": dict(stats),
              "stage_seconds": {k: round(v, 1) for k, v in stage_secs.items()},
              "changed": changed}
    print(f"sync summary: {report['stats']} | "
          f"stage seconds: {report['stage_seconds']}")
    return report


# ----------------------------------------------------------------------------
# inspect — look at what the pipeline actually extracted (eye check vs the PDF)
# ----------------------------------------------------------------------------
def inspect(book: str, n: int = 10, cfg=None) -> None:
    cfg = cfg or load_config()
    lib, index = Path(cfg.paths["library"]), Path(cfg.paths["index"])
    key = None
    for cand in (Path(book), lib / book, lib / Path(book).name):   # cwd, lib-relative, name
        if cand.exists() and cand.suffix.lower() == ".pdf":
            key = sha256_file(cand)[:16]
            break
    else:
        catalog = json.loads((index / "catalog.json").read_text(encoding="utf-8")) \
            if (index / "catalog.json").exists() else {}
        if book in catalog:
            key = catalog[book]["sha256"][:16]
        elif (index / "books" / book).exists():
            key = book
    if not key or not (index / "books" / key / "pages.jsonl").exists():
        print(f"book not found: {book}")
        return
    meta = json.loads((index / "books" / key / "meta.json").read_text(encoding="utf-8"))
    pages = read_jsonl(index / "books" / key / "pages.jsonl")
    sample = random.sample(pages, min(n, len(pages)))
    print(f"«{meta['title']}» key={key} pages={meta['n_pages']} "
          f"label_source={meta['label_source']}")
    for pg in sorted(sample, key=lambda x: x["i"]):
        print(f"\n--- PDF page {pg['i']} (label ص {pg['label']}, method={pg['method']}, "
              f"chars={pg['q']['chars']}, script={pg['q']['script']}, bad={pg['q']['bad']})")
        print(pg["text"][:180].replace("\n", " ⏎ "))


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    _utf8_console()
    ap = argparse.ArgumentParser(description="BookSearch-FA indexing (stage 1)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync", help="scan library/, process new/changed PDFs, rebuild index")
    s.add_argument("--workers", type=int, default=0,
                   help="N>1 = extract/chunk موازی سطح کتاب (پروسه به ازای هر هسته)؛ embed همیشه در پروسهٔ اصلی")
    s.add_argument("--no-embed", action="store_true", help="BM25-only, no torch needed")
    s.add_argument("--force", choices=["extract", "chunk", "embed"], help="reprocess everything")

    i = sub.add_parser("inspect", help="print random extracted pages for eye-checking")
    i.add_argument("book", help="pdf path (relative to library/), catalog rel path, or key")
    i.add_argument("--pages", type=int, default=10)

    args = ap.parse_args()
    cfg = load_config()
    if args.cmd == "sync":
        sync(cfg, workers=args.workers, no_embed=args.no_embed, force=args.force)
    elif args.cmd == "inspect":
        inspect(args.book, n=args.pages, cfg=cfg)


if __name__ == "__main__":
    main()
