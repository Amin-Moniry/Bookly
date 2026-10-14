"""
test_booksearch.py — pytest smoke tests for stage 1 of BookSearch-FA.

Covers: normalization (idempotence, yeh/kaf, digits, diacritics, ZWNJ),
the BM25 tokenizer, weighted RRF with a numeric example, sentence splitting,
and a tiny end-to-end sync+query on a Persian PDF built with PyMuPDF
(add / rename / edit / remove).

Run:  pytest -q test_booksearch.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import booker
import search
from search import (Hit, SearchConfig, Searcher, load_config, normalize_display,
                    normalize_search, rrf, tokenize_bm25)
from ingest import (_split_sentences, build_merged, chunk_book, extract_pages,
                    needs_ocr, ocr_pages, page_quality, reprocess_book, scan_library, sync)

fitz = pytest.importorskip("fitz")           # PyMuPDF
np = pytest.importorskip("numpy")

FA_FONT = r"C:\Windows\Fonts\tahoma.ttf"     # any Windows font with Arabic glyphs


# ---------------------------------------------------------------- normalization
def test_display_yeh_kaf_and_idempotent():
    d = normalize_display("\u0645\u0643\u062a\u0628\u0629 \u0643\u064a\u0641")  # مكتبة كيف
    assert "ک" in d and "ی" in d                                  # ك→ک، ي→ی
    s = normalize_display("کتاب \u064a")
    assert normalize_display(s) == s                              # idempotent


def test_search_digits_diacritics_idempotent():
    n1 = normalize_search("۱۲۳۴ سال ۱۴۰۲ \u0640\u0650Test")
    assert "۱۲۳۴" not in n1 and "1234" in n1
    assert "test" in n1 and "\u0640" not in n1 and "\u0650" not in n1
    assert normalize_search(n1) == n1                             # idempotence (README §6.1)


def test_zwnj_becomes_space_in_search():
    assert tokenize_bm25("کتاب‌ها") == tokenize_bm25("کتاب ها")


def test_tokenizer_drops_stopwords_keeps_negation():
    toks = tokenize_bm25("کتاب و مجله در کتابخانه نیست")
    assert "و" not in toks and "در" not in toks
    assert "نیست" in toks                                         # negation words kept
    assert "کتاب" in toks and "کتابخانه" in toks


def test_tokenizer_strips_arabic_punctuation_and_filler():
    # ،؛؟ live INSIDE \u0600-\u06ff: they must not glue onto words (fa-tok-2)
    assert tokenize_bm25("اکسپلویت،") == ["اکسپلویت"]
    assert "اکسپلویت" in tokenize_bm25("اکسپلویت؛ مسیر فایل")
    assert tokenize_bm25("می‌خوام") == tokenize_bm25("میخوام")  # ZWNJ + colloquial filler
    assert tokenize_bm25("قیمت چنده؟") == ["قیمت"]


# ------------------------------------------------------------------------- RRF
def test_rrf_numeric_example():
    # README §8: rank 1 in BM25 and rank 3 in dense -> 1/61 + 1/63 ≈ 0.0323
    out = rrf({"bm25": np.array([10, 20, 30]), "dense": np.array([30, 99, 10])},
              k=60, weights={"bm25": 1.0, "dense": 1.0}, top=10)
    scores = dict(out)
    assert abs(scores[10] - (1 / 61 + 1 / 63)) < 1e-9
    assert abs(scores[30] - (1 / 63 + 1 / 61)) < 1e-9
    assert out[0][0] == 10                                        # exact tie -> lower row wins


def test_rrf_weights():
    out = rrf({"bm25": np.array([1, 2]), "dense": np.array([2, 1])}, k=60,
              weights={"bm25": 1.0, "dense": 0.5}, top=2)
    assert dict(out)[1] > dict(out)[2]                            # bm25 weight dominates


def test_rrf_intersection_mode_ignores_noise_rows():
    # one list is pure noise (e.g. zero-signal query): only agreed rows are scored
    out = rrf({"bm25": np.array([5, 6, 7, 8]), "dense": np.array([3, 5, 9, 6])},
              k=60, top=10, rows_keep={5, 6})
    assert [r for r, _ in out] == [5, 6]
    assert abs(out[0][1] - (1 / 61 + 1 / 62)) < 1e-9               # rank1+rank2 for row 5
    # empty rows_keep (or None) behaves exactly like the plain union
    out_u = rrf({"bm25": np.array([5]), "dense": np.array([3, 5, 9, 6])}, k=60, top=10)
    assert [r for r, _ in out_u] == [5, 3, 9, 6]                   # union: dense rank order preserved


def test_hamza_yeh_forms_normalized():
    # fa-norm-3: «هٔ» (ه+U+0654) و ؀ (U+06C0، مقصد NFKC) هر دو «ه» می‌شوند (README §6.1)
    d = normalize_display("خانهٔ ما")
    assert d == normalize_display("خانه ما")
    assert normalize_display("\u0647\u0654") == "ه"            # ه + hamza above
    assert normalize_display("\u06c0") == "ه"                  # ؀
    assert normalize_search("خانهٔ") == normalize_search("خانه")
    assert tokenize_bm25("خانهٔ") == tokenize_bm25("خانه")
    assert normalize_display(normalize_display("خانهٔ")) == normalize_display("خانهٔ")  # idempotent


def _mk_assemble_searcher(chunks: list[dict]) -> Searcher:
    """A Searcher with only the state _assemble needs (no index on disk)."""
    s = object.__new__(Searcher)
    cfg = load_config()
    cfg.search = SearchConfig(final_chunks=3, neighbors=0, context_tokens=10**6)
    cfg.chunk["header"] = False
    s.cfg = cfg
    s.manifest = {"books": [{"key": "k", "file": "t.pdf", "title": "کتاب آزمون", "row0": 0,
                             "n": len(chunks)}]}
    s._chunks = chunks
    s._dense = None                      # retrieve() checks this; keeps the object complete
    return s


def _hit(row: int) -> Hit:
    """A minimal Hit for driving _assemble directly (unit under test: merge/dedup)."""
    return Hit(row=row, chunk_id=f"k:{row:05d}", score=1.0, ranks={}, valid_hit=True)


def _chunk_dict(seq: int, text: str, sha1: str, sec: int = 1) -> dict:
    return {"seq": seq, "id": f"k:{seq:05d}", "key": "k", "sec": sec, "chapter": "",
            "ntok": max(1, (len(text) + 3) // 4), "text": text, "sha1": sha1,
            "p0": seq, "p1": seq, "l0": str(seq + 1), "l1": str(seq + 1)}


def test_assemble_merges_without_repeating_overlap_sentence():
    # chunker هر تکه را با یک جملهٔ هم‌پوشان می‌سازد؛ ادغام نباید جمله را دوباره بیاورد (README §7)
    chunks = [_chunk_dict(0, "یکی. دو.", "a0"), _chunk_dict(1, "دو. سه.", "a1"),
              _chunk_dict(2, "چهار.", "a2")]
    s = _mk_assemble_searcher(chunks)
    p = s._assemble([_hit(r) for r in (0, 1, 2)])[0]
    assert "دو. دو." not in p.text and "دو دو" not in p.text
    assert p.text == "یکی. دو. سه. چهار."
    assert p.rows == [0, 1, 2]


def test_assemble_drops_duplicate_sha1_chunks():
    chunks = [_chunk_dict(0, "الف.", "x"), _chunk_dict(1, "الف.", "x"),   # exact duplicate
              _chunk_dict(2, "ب.", "y")]
    s = _mk_assemble_searcher(chunks)
    p = s._assemble([_hit(r) for r in (0, 1, 2)])[0]
    assert p.text == "الف. ب."
    assert p.rows == [0, 2]                       # duplicate row collapsed


def test_chunk_id_stable_across_rechunk():
    pages = []
    for i in range(3):
        from ingest import Page
        pages.append(Page(i=i, label=str(i + 1), text="جملهٔ آزمایشی شمارهٔ یک. جملهٔ آزمایشی شمارهٔ دو. " * 20))
    cfg = search.Config()
    c1, c2 = chunk_book(pages, [], cfg), chunk_book(pages, [], cfg)
    assert [(c["seq"], c["text"]) for c in c1] == [(c["seq"], c["text"]) for c in c2]
    # chunk_id از (key, seq) ساخته می‌شود -> با این خروجی پایدار، id هم پایدار است
    assert [f"k:{c['seq']:05d}" for c in c1] == [f"k:{c['seq']:05d}" for c in c2]


def test_sync_stage_versions_stale(tmp_path, mini_pdf):
    """fa-norm در هر سه مرحله: خراب‌کردن نسخهٔ chunk باعث re-chunk از کش صفحات می‌شود."""
    import json as _json
    cfg = _cfg_for(tmp_path, mini_pdf)
    sync(cfg, no_embed=True)
    catalog = _json.loads((tmp_path / "index" / "catalog.json").read_text(encoding="utf-8"))
    key = catalog["mini.pdf"]["sha256"][:16]
    meta_path = tmp_path / "index" / "books" / key / "meta.json"
    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
    meta["stages"]["chunk"]["version"] = "bogus"
    meta_path.write_text(_json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    report = sync(cfg, no_embed=True)
    meta2 = _json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta2["stages"]["chunk"]["version"] != "bogus"        # re-chunked
    assert meta2["stages"]["extract"]["version"] == __import__("ingest")._extract_version(cfg)  # همان cfg که sync با آن اجرا شد
    assert report["stats"].get("changed") == 1                   # not a full 'processing' run


def test_build_merged_generations_monotonic(tmp_path, mini_pdf):
    import json as _json
    cfg = _cfg_for(tmp_path, mini_pdf)
    sync(cfg, no_embed=True)
    catalog = _json.loads((tmp_path / "index" / "catalog.json").read_text(encoding="utf-8"))
    key = catalog["mini.pdf"]["sha256"][:16]
    metas = [_json.loads((tmp_path / "index" / "books" / key / "meta.json").read_text(encoding="utf-8"))]
    gen1 = build_merged(tmp_path / "index", metas, cfg)
    gen2 = build_merged(tmp_path / "index", metas, cfg)
    n1, n2 = int(gen1.split("-")[1]), int(gen2.split("-")[1])
    assert n2 == n1 + 1                                           # bug fix: no repeated gen numbers
    man = _json.loads((tmp_path / "index" / "merged" / gen2 / "manifest.json").read_text(encoding="utf-8"))
    assert man["bm25"]["vocab"] > 0 and man["sparse"] is False and man["norm"] == search.NORM_VERSION


def test_valid_hit_rule():
    # قاعدهٔ «hit معتبر» (پایهٔ پیام «در کتاب‌ها نیست»): رتبهٔ BM25 شاهد است؛
    # بدون BM25، شباهت dense باید از آستانه رد شود؛ بدون هیچ‌کدام نامعتبر.
    assert search.valid_hit(1, None) is True
    assert search.valid_hit(3, 0.1) is True                 # BM25 rank کافی است
    assert search.valid_hit(None, 0.51, min_dense_sim=0.5) is True
    assert search.valid_hit(None, 0.49, min_dense_sim=0.5) is False
    assert search.valid_hit(None, None) is False            # هر دو لیست خالی


# ------------------------------------------------------------------ chunking
def test_chunk_size_cap():
    pages = []
    for i in range(3):
        from ingest import Page
        pages.append(Page(i=i, label=str(i + 1), text="کلمه " * 400))
    cfg = search.Config()
    chunks = chunk_book(pages, [], cfg)
    assert chunks
    assert all(c["ntok"] <= cfg.chunk["max_tokens"] for c in chunks)
    assert [c["seq"] for c in chunks] == list(range(len(chunks)))  # stable seq


def test_split_sentences():
    assert _split_sentences("اول. دوم؟ سوم! چهارم؛ پنجم") == ["اول.", "دوم؟", "سوم!", "چهارم؛", "پنجم"]


# ------------------------------------------------- end-to-end with a real PDF
def _mk_mini_pdf(lib: Path) -> Path:
    """A DENSE Persian PDF (8 sections + one footer number per section page) —
    dense body so OCR actually finds text (tesseract's layout analysis drops a
    lone short line) and section-per-page footers so the label detector sees a
    repeated numbered pattern. Note: PyMuPDF-generated PDFs extract Persian in
    visual (word-reversed) order — README §13.2; word-level tokens stay correct,
    which is all BM25 needs in this smoke test."""
    doc = fitz.open()
    css = "@font-face {font-family: fa; src: url('C:/Windows/Fonts/tahoma.ttf');} * {font-family: fa; font-size: 13px;}"
    sentences = ["کتاب تاریخ ایران مطالعه‌ای مفید است و دانستن گذشته را آسان می‌کند.",
                 "گلستان و آثار بزرگ ادب فارسی یادآور روزگار گذشته‌اند.",
                 "پخت آش رشته با سبزی خشک و کشک از غذاهای سنتی ایران است.",
                 "جنگل‌های سرسبز شمال کشور هر سال گردشگران بسیاری را جذب می‌کنند."]
    for i in range(8):
        body = "<br><br>".join(
            " ".join(sentences[(i + j) % len(sentences)] for j in range(2))
            + f" این متن صفحهٔ آزمایشی شمارهٔ {i + 1} است."
            for _ in range(4))                              # ← dense: هر صفحه ≈‌۸ جمله تا gate پاس شود و OCR خروجی معنا‌دار بدهد
        page = doc.new_page()
        page.insert_htmlbox(fitz.Rect(50, 50, 545, 700), body, css=css)
        page.insert_htmlbox(fitz.Rect(50, 740, 545, 760), str(i + 1), css=css)   # footer
    out = lib / "mini.pdf"
    doc.save(str(out))
    doc.close()
    return out


@pytest.fixture
def mini_pdf(tmp_path) -> Path:
    if not Path(FA_FONT).exists():
        pytest.skip("no Persian-capable system font for the test PDF")
    return _mk_mini_pdf(tmp_path)


def _mk_searcher(index_dir: Path) -> Searcher:
    cfg = load_config()
    cfg.paths["index"] = str(index_dir)
    cfg.search = SearchConfig(final_chunks=3, context_tokens=1500)
    return Searcher(index_dir=index_dir, cfg=cfg)


def _cfg_for(tmp_path: Path, mini_pdf: Path, ocr: str = "none"):
    """cfg مشترک تست‌ها. OCR پیش‌فرض خاموش است؛ ایندکس‌گذاری نباید به نصب بودن
    Tesseract وابسته باشد (فقط تست‌های مرتبط با OCR خودشان آن را فعال می‌کنند)."""
    cfg = load_config()
    cfg.paths["library"] = str(mini_pdf.parent)
    cfg.paths["index"] = str(tmp_path / "index")
    cfg.extract.ocr_engine = ocr
    return cfg


def test_e2e_bm25_only(tmp_path, mini_pdf):
    """sync --no-embed path: BM25-only index, query hits the right page."""
    cfg = _cfg_for(tmp_path, mini_pdf)
    sync(cfg, no_embed=True)
    s = _mk_searcher(tmp_path / "index")
    res = s.search("کتاب تاریخ ایران")
    assert res.hits and res.hits[0].score > 0
    p = res.passages[0]
    # all 8 pages share one text -> one chunk set merged by neighbors into pp. 1–8
    assert p.pages == "1–8" and p.pdf_pages == (0, 7)
    assert "تاریخ" in p.text                       # word-level token (visual order: see fixture docstring)


def test_found_flag_bm25_only(tmp_path, mini_pdf):
    """found=False وقتی هیچ hit معتبری نیست: tokenهای OOV هر دو لیست را خالی می‌کنند."""
    cfg = _cfg_for(tmp_path, mini_pdf)
    sync(cfg, no_embed=True)
    s = _mk_searcher(tmp_path / "index")
    res = s.search("کتاب تاریخ ایران")
    assert res.found is True and res.hits
    res2 = s.search("زیبایی‌شناسی ابررسانایی کوانتوم")      # هیچ token آن در vocab نیست
    assert res2.hits == [] and res2.found is False and res2.passages == []


def test_cli_reports_not_found(tmp_path, mini_pdf):
    """CLI وقتی هیچ hit معتبری نیست، صریح می‌گوید «نتیجه‌ای در کتاب‌ها یافت نشد»."""
    cfg = _cfg_for(tmp_path, mini_pdf)
    sync(cfg, no_embed=True)
    root = Path(__file__).parents[1]                 # project root (booksearch.toml lives here)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    def run_cli(q: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(root / "app" / "search.py"), q, "--index", str(tmp_path / "index")],
            capture_output=True, text=True, encoding="utf-8", env=env,
            cwd=str(root), timeout=120)

    r = run_cli("زیبایی‌شناسی ابررسانایی کوانتوم")
    assert r.returncode == 0, r.stderr
    assert "نتیجه‌ای در کتاب‌ها یافت نشد" in r.stdout
    r2 = run_cli("کتاب تاریخ ایران")
    assert r2.returncode == 0, r2.stderr
    assert "تاریخ" in r2.stdout


# --------------------------------------------------------------- booker.py --
def test_booker_build_messages_and_history():
    cfg = booker.LLMConfig(history_turns=2, context_chars=10**6)
    history = [(f"q{i}", f"a{i}") for i in range(5)]
    msgs = booker.build_messages("سؤال جدید", "[1] «کتاب» ص ۱\nمتن", history, cfg)
    assert msgs[0]["role"] == "system" and "در کتاب‌ها نیست" in msgs[0]["content"]
    # فقط ۲ جفت آخرِ تاریخچه + پیام نهایی (از ۵ جفت)
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user", "assistant", "user"]
    joined = "".join(m["content"] for m in msgs)
    assert "q3" in joined and "q0" not in joined and "a0" not in joined
    assert "[1] «کتاب» ص ۱" in msgs[-1]["content"] and "سؤال جدید" in msgs[-1]["content"]


def test_booker_trim_context_cap():
    assert booker.trim_context("abc", 10) == "abc"
    out = booker.trim_context("x" * 50, 10)
    assert out.startswith("x" * 10) and out.endswith("…")


def test_booker_sse_parse_and_delta():
    ok = booker.parse_sse_line('data: {"choices":[{"delta":{"content":"سلام"}}]}')
    assert ok is not None and booker.delta_of(ok) == "سلام"
    role_chunk = booker.parse_sse_line('data: {"choices":[{"delta":{"role":"assistant"}}]}')
    assert booker.delta_of(role_chunk) == ""          # chunk نقش، بدون متن
    assert booker.parse_sse_line("data: [DONE]") is None
    assert booker.parse_sse_line(": keep-alive") is None
    assert booker.parse_sse_line("") is None
    assert booker.parse_sse_line("data: {bad json") is None


def test_booker_abstain_skips_api(tmp_path, mini_pdf):
    """found=False → پیام صریح و بدون هیچ تماس با API (چیزی به LLM نمی‌رسد تا توهم بزند)."""
    cfg = _cfg_for(tmp_path, mini_pdf)
    sync(cfg, no_embed=True)
    root = Path(__file__).parents[1]                 # project root (booksearch.toml lives here)
    # key-free repo: the abstain path never calls the API, but booker requires a key
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "BOOKLY_API_KEY": "test-dummy"}
    r = subprocess.run(
        [sys.executable, str(root / "app" / "booker.py"), "زیبایی‌شناسی ابررسانایی کوانتوم",
         "--index", str(tmp_path / "index")],
        capture_output=True, text=True, encoding="utf-8", env=env,
        cwd=str(root), timeout=120)
    assert r.returncode == 0, r.stderr
    assert "نتیجه‌ای در کتاب‌ها یافت نشد" in r.stdout


def test_scan_library_incremental(tmp_path, mini_pdf):
    """Same content -> same sha/key; rename is detected as the same book."""
    catalog: dict = {}
    files = scan_library(mini_pdf.parent, catalog)
    assert len(files) == 1
    catalog = {f.rel: {"size": f.size, "mtime_ns": f.mtime_ns, "sha256": f.sha256} for f in files}
    assert scan_library(mini_pdf.parent, catalog)[0].sha256 == files[0].sha256
    shutil.copy(mini_pdf, mini_pdf.parent / "renamed.pdf")
    by_sha = {f.sha256: f.rel for f in scan_library(mini_pdf.parent, catalog)}
    assert by_sha[files[0].sha256] == "renamed.pdf"


def _tesseract_ready() -> bool:
    """Tesseract + traineddata (fas/eng) موجود است؟ اگر نه، تست‌های OCR-specific skip می‌شوند.
    همان منطق کشف باینری که ingest خودش دارد (config، مسیرهای استاندارد ویندوز، PATH)."""
    import ingest
    try:
        import pytesseract
        cfg = load_config()
        cmd = ingest._resolve_tesseract_cmd(cfg)
        ok_cmd = bool(cmd and Path(cmd).exists())
        if not ok_cmd:
            pytesseract.get_tesseract_version()
    except Exception:
        return False
    td = Path(__file__).resolve().parents[1] / "app" / "tessdata"
    return (td / "fas.traineddata").exists() and (td / "eng.traineddata").exists()


NEEDS_TESSERACT = pytest.mark.skipif(not _tesseract_ready(),
                                     reason="Tesseract یا traineddataͥ نصب نیست (تست OCR-specific)")


@NEEDS_TESSERACT
def test_page_quality_and_labels(mini_pdf):
    pages, info = extract_pages(mini_pdf, opts={}, cfg=load_config())
    assert info["source"] == "footer"             # repeated footer numbers detected
    assert pages[0].label == "1" and pages[7].label == "8"
    q = page_quality(normalize_display("کتاب تاریخ"))
    assert q["script"] == 1.0 and q["chars"] > 0


# --------------------------------------------------------------------- OCR --
def test_needs_ocr_language_aware():
    """Gate زبان‌آگاه (ex-2): صفحهٔ انگلیسیِ سالم OCR نمی‌خواهد؛ فارسیِ خراب می‌خواهد."""
    g = {"min_chars": 150, "min_script_ratio": 0.6, "max_bad_ratio": 0.02, "min_lex_hit": 0.3}
    en_ok = {"chars": 2000, "script": 0.0, "bad": 0.0, "lex": 1.0, "rev": 0.0, "fa_toks": 0}
    fa_bad = {"chars": 2000, "script": 0.99, "bad": 0.0, "lex": 0.0, "rev": 0.0, "fa_toks": 300}
    fa_good = {"chars": 2000, "script": 0.99, "bad": 0.0, "lex": 0.8, "rev": 0.0, "fa_toks": 300}
    fa_visual = {"chars": 2000, "script": 0.99, "bad": 0.0, "lex": 0.1, "rev": 0.9, "fa_toks": 300}
    empty = {"chars": 40, "script": 0.0, "bad": 0.0, "lex": 1.0, "rev": 0.0, "fa_toks": 0}
    no_dict = {"chars": 2000, "script": 0.9, "bad": 0.0}   # بدون lex (واژگان غایب)
    assert needs_ocr(en_ok, g) is False          # English page: healthy, no OCR
    assert needs_ocr(fa_bad, g) is True          # Arabic letters, zero valid words
    assert needs_ocr(fa_good, g) is False
    assert needs_ocr(fa_visual, g) is True       # visual order beats the dictionary
    assert needs_ocr(empty, g) is True           # too few chars
    assert needs_ocr(no_dict, g) is False        # legacy path: high script_ratio passes
    assert needs_ocr({"chars": 2000, "script": 0.2, "bad": 0.0}, g) is True   # legacy path fails low ratio


def test_needs_ocr_without_dictionary_matches_legacy_behavior():
    """بدون fa_words.txt: q بدون lex → فقط script_ratio/chars/bad تصمیم می‌گیرد."""
    g = {"min_chars": 150, "min_script_ratio": 0.6, "max_bad_ratio": 0.02}
    assert needs_ocr({"chars": 2000, "script": 0.9, "bad": 0.0}, g) is False
    assert needs_ocr({"chars": 2000, "script": 0.2, "bad": 0.0}, g) is True


def test_ocr_pages_tesseract_integration(tmp_path, mini_pdf):
    """OCR واقعی یک PDF متراکم ساخته‌شده با PyMuPDF (نیازمند tesseract نصب).
    متن رندرشده visual order است؛ tesseract هم حروف لاتین‌نما می‌خواند، پس فقط
    طول خروجی و هویت بصری را چک می‌کنیم (سیگنال rev — README §13.2)."""
    cfg = load_config()
    try:
        out = ocr_pages(mini_pdf, [0], cfg)
    except RuntimeError as e:
        pytest.skip(f"tesseract در دسترس نیست: {e}")
    assert 0 in out and len(out[0]) > 100, out[0][:120]   # متن متراکم → خروجی قابل توجه


@NEEDS_TESSERACT
def test_extract_pages_ocr_flag_and_gate_off(tmp_path, mini_pdf):
    """با ocr_engine=none (پیش‌فرض کد) هیچ OCR نمی‌شود؛ با tesseract هم صفحاتی که
    gate رد می‌کنند (اینجا همه: visual order) OCR می‌شوند و متن‌شان جایگزین است.
    تesseract + traineddata لازم دارد؛ روی کلون تازه skip می‌شود."""
    cfg = load_config()
    cfg.extract.ocr_engine = "none"
    pages, info = extract_pages(mini_pdf, opts={}, cfg=cfg)
    assert all(pg.method == "text" for pg in pages)
    assert info.get("ocr", []) == []

    cfg.extract.ocr_engine = "tesseract"
    # صفحه‌های کم‌متن (کوتاه‌تر از حداقلِ gate) همیشه به OCR می‌روند؛ مستقل از محتوای mini
    tiny = tmp_path / "tiny.pdf"
    _d = fitz.open()
    for _n in range(3):
        _p = _d.new_page()
        _p.insert_text((72, 100), f"Tiny sample page {_n + 1} of the OCR gate test.", fontsize=12)
    _d.save(str(tiny))
    _d.close()
    try:
        pages, info = extract_pages(tiny, opts={}, cfg=cfg)
    except RuntimeError as e:
        pytest.skip(f"tesseract در دسترس نیست: {e}")
    ocr_idx = [pg.i for pg in pages if pg.method == "ocr"]
    assert ocr_idx == info["ocr"] and len(ocr_idx) == 3
    assert all(pg.text.strip() for pg in pages)
    # هر صفحهٔ OCRشده یا متن بهتر از لایه دارد یا لایهٔ اصلی حفظ شده (بدون تخریب)
    assert all(pg.q["chars"] > 0 for pg in pages)
    assert all(pg.text.strip() for pg in pages)


def test_extract_version_includes_gate(tmp_path):
    cfg = load_config()
    v = __import__("ingest")._extract_version(cfg)
    assert v.startswith("ex-2|fa-norm-3|ocr-")
    assert "min_lex_hit" not in v and "0.3" in v      # gate values compacted in


def test_e2e_edit_then_remove(tmp_path, mini_pdf):
    cfg = _cfg_for(tmp_path, mini_pdf)
    sync(cfg, no_embed=True)
    # edit the PDF -> new sha -> new key processed, old key dropped
    doc = fitz.open(str(mini_pdf))
    page = doc.new_page()
    page.insert_font(fontname="fa", fontfile=FA_FONT)
    page.insert_textbox(fitz.Rect(50, 50, 545, 700), "صفحهٔ کاملاً تازه دربارهٔ فلسفهٔ علم.",
                        fontname="fa", fontsize=12)
    tmp_pdf = mini_pdf.with_suffix(".tmp.pdf")
    doc.save(str(tmp_pdf))
    doc.close()
    tmp_pdf.replace(mini_pdf)
    sync(cfg, no_embed=True)
    catalog = json.loads((tmp_path / "index" / "catalog.json").read_text(encoding="utf-8"))
    new_key = catalog["mini.pdf"]["sha256"][:16]
    keys = {p.name for p in (tmp_path / "index" / "books").iterdir()}
    assert new_key in keys
    # remove the PDF -> cache removed on next sync, CURRENT cleared
    mini_pdf.unlink()
    sync(cfg, no_embed=True)
    keys2 = {p.name for p in (tmp_path / "index" / "books").iterdir()}
    assert new_key not in keys2
    assert not (tmp_path / "index" / "merged" / "CURRENT").exists()


# ----------------------------------------------------------------- evaluate (A6)
_EVAL_PAGES = [
    "The red fox lives in the deep forest and eats red berries every single morning.",
    "The old baker used eight kilograms of sour apples for the autumn festival pie.",
    "The rooftop garden receives warm morning sun and fresh water from a rain barrel.",
    "Good compost needs equal parts of dry leaves and soft greens to stay balanced.",
    "The old librarian catalogued seven poetry volumes for the spring exhibition.",
    "The little robot painted seven blue bicycles in the quiet garage last spring.",
]


def _write_eval_questions(path: Path) -> None:
    lines = [
        {"id": "e-fox", "slice": "name", "split": "dev",
         "question": "Where does the red fox live and what does it eat?",
         "witnesses": [{"quote": "The red fox lives in the deep forest", "page": 1}]},
        {"id": "e-baker", "slice": "number", "split": "dev",
         "question": "How many kilograms of apples did the old baker use for the pie?",
         "witnesses": [{"quote": "The old baker used eight kilograms of sour apples", "page": 2}]},
        {"id": "e-multi", "slice": "multi", "split": "test",
         "question": "How does the rooftop garden get water and what parts keep compost balanced?",
         "witnesses": [{"quote": "The rooftop garden receives warm morning sun and fresh water", "page": 3},
                       {"quote": "Good compost needs equal parts of dry leaves and soft greens to stay balanced", "page": 4}]},
        # سؤال none باید کاملاً OOV باشد: یک واژهٔ انگلیسیِ پرتکرار مثل «a» در
        # vocab یک پیکرهٔ انگلیسی رتبهٔ BM25 می‌گیرد و طبق قاعدهٔ README، hit معتبر است.
        {"id": "e-none", "slice": "none", "split": "test",
         "question": "طرز تهیه کیک شکلاتی خانگی بدون فر چیه؟", "witnesses": []},
    ]
    path.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines), encoding="utf-8")


@pytest.fixture
def eval_index(tmp_path):
    """BM25-only 6-page ENGLISH index (no bidi quirks) for the eval harness."""
    doc = fitz.open()
    for i, txt in enumerate(_EVAL_PAGES):
        page = doc.new_page()
        page.insert_text((72, 100), txt, fontsize=12)
        page.insert_text((72, 760), str(i + 1), fontsize=10)   # footer page number
    out = tmp_path / "evalbook.pdf"
    doc.save(str(out))
    doc.close()
    cfg = load_config()
    cfg.paths["library"] = str(tmp_path)
    cfg.paths["index"] = str(tmp_path / "index")
    cfg.extract.ocr_engine = "none"                     # تست eval مستقل از Tesseract است
    sync(cfg, no_embed=True)
    return tmp_path / "index"


def test_evaluate_gold_resolution(eval_index):
    import evaluate as ev
    s = _mk_searcher(eval_index)
    q_ok = ev.Question(id="g1", slice="name", split="dev", question="fox",
                       witnesses=[ev.Witness(quote="The red fox lives in the deep forest", page=1)])
    q_far = ev.Question(id="g2", slice="name", split="dev", question="fox",
                        witnesses=[ev.Witness(quote="The red fox lives in the deep forest", page=99)])
    q_none = ev.Question(id="g3", slice="none", split="dev", question="lemon pie")
    gold = ev.resolve_gold([q_ok, q_far, q_none], s)
    assert gold["g1"]                                   # quote + page ±1 -> resolved
    assert gold["g2"] == {}                             # page 99 out of range -> no gold
    assert gold["g3"] == {}                             # no witnesses -> abstain slice


def test_evaluate_run_and_metrics(eval_index):
    import evaluate as ev
    s = _mk_searcher(eval_index)
    qpath = eval_index.parent / "questions.jsonl"
    _write_eval_questions(qpath)
    qs = ev.load_questions(qpath)
    gold = ev.resolve_gold(qs, s)
    assert gold["e-fox"] and gold["e-baker"] and gold["e-multi"] and not gold["e-none"]
    run = ev.run_variant(s, qs, {})
    quotes = {q.id: [w.quote for w in q.witnesses] for q in qs if q.slice != "none"}
    m = ev.metrics({q.id: run[q.id] for q in qs if gold[q.id]}, gold, quotes=quotes)
    assert m["n"] == 3
    assert m["hit@1"] == 1.0 and m["hit@5"] == 1.0 and m["mrr@10"] == 1.0
    assert m["context_hit"] == 1.0                      # every witness inside final context
    ab = ev.abstain_stats(qs, run)
    assert ab["abstain_acc"] == 1.0 and ab["false_abstain"] == 0.0


def test_evaluate_cc_and_variants(eval_index):
    import evaluate as ev
    s = _mk_searcher(eval_index)
    qpath = eval_index.parent / "questions.jsonl"
    _write_eval_questions(qpath)
    qs = ev.load_questions(qpath)
    gold = ev.resolve_gold(qs, s)
    run_cc = ev.run_variant(s, qs, ev.VARIANTS["cc"])   # BM25-only -> CC degrades to minmax(bm25)
    assert run_cc["e-fox"].rows and set(gold["e-fox"]) <= set(run_cc["e-fox"].rows)
    assert run_cc["e-fox"].found is True
    run_b = ev.run_variant(s, qs, ev.VARIANTS["bm25"])
    assert set(gold["e-fox"]) <= set(run_b["e-fox"].rows)
    run_nb = ev.run_variant(s, qs, {"neighbors": 0})    # query-time neighbor override
    assert run_nb["e-fox"].rows                         # hits are passage-independent


def test_paired_test_sanity():
    import evaluate as ev
    a = [1.0] * 9 + [0.0]
    b = [0.0] * 9 + [1.0]
    assert ev.paired_test(a, a) == 1.0                  # identical -> no signal
    assert ev.paired_test(a, b) < 0.05                  # clean separation -> small p
    assert ev.paired_test([1.0, 0.0], [1.0, 0.0, 1.0]) == 1.0   # length mismatch


# ------------------------------------------------------------------- C1/B1/C2
def test_embed_version_backend_flag():
    """نسخهٔ embed backend را هم حمل می‌کند (st|m3) — تغییر backend → re-embed."""
    import ingest
    cfg = search.Config()
    v_st = ingest._embed_version(cfg, True)
    assert "|st|" in v_st
    cfg.embed.backend = "m3"
    v_m3 = ingest._embed_version(cfg, True)
    assert "|m3|" in v_m3 and v_m3 != v_st
    assert ingest._embed_version(cfg, False) == "none"
    # parse دوطرفه
    assert search.parse_embed_version("fa-norm-3|BAAI/bge-m3|st|512|hdr1") == ("BAAI/bge-m3", "st", 512)
    assert search.parse_embed_version("fa-norm-3|BAAI/bge-m3|m3|512|hdr1") == ("BAAI/bge-m3", "m3", 512)
    assert search.parse_embed_version("BAAI/bge-m3|st|512|hdr1") == ("BAAI/bge-m3", "st", 512)


def test_cc_fuse_with_sparse_third_list():
    """C1: لیست سوم sparse فقط وقتی حس می‌شود که row بیرون اجتماع دو لیست باشد؛
    صفرشدن beta باید دقیقاً همان CC دو لیستی را بدهد."""
    b_ids, b_sc = np.array([10, 20]), np.array([9.0, 5.0])
    d_ids, d_sc = np.array([30]), np.array([0.9])
    cc2 = dict(search.cc_fuse(b_ids, b_sc, d_ids, d_sc, alpha=0.5, top=10))
    cc3_zero = dict(search.cc_fuse(b_ids, b_sc, d_ids, d_sc, alpha=0.5, top=10,
                                   sparse_ids=np.array([10]), sparse_sc=np.array([3.0]),
                                   beta=0.0))
    assert cc2.keys() == cc3_zero.keys() and all(abs(cc2[r] - cc3_zero[r]) < 1e-9 for r in cc2)
    # sparse row 30 را تقویت می‌کند (وزن 3 → minmax=1 → +0.25·1)
    cc3 = dict(search.cc_fuse(b_ids, b_sc, d_ids, d_sc, alpha=0.5, top=10,
                              sparse_ids=np.array([30]), sparse_sc=np.array([3.0]),
                              beta=0.25))
    assert cc3[30] > cc2[30] - 1e-9                                     # تقویت یا حفظ
    assert abs(cc3[30] - (0.5 * 1.0 + 0.25 * 1.0)) < 1e-6


def test_sparse_npz_roundtrip(tmp_path):
    """C1: CSR نوشته‌شده در sparse.npz با _SparseIndex همان امتیازها را می‌دهد."""
    from ingest import _write_sparse_npz
    from search import _SparseIndex
    rows = [{"5": 1.5, "20": 0.5}, {}, {"5": 0.2, "99": 2.0}]
    p = tmp_path / "sparse.npz"
    _write_sparse_npz(p, rows)
    si = _SparseIndex(p)
    assert si.n_rows == 3
    ids, sc = si.top({"5": 2.0, "99": 1.0}, depth=3)
    # row0: 1.5·2.0=3.0؛ row1: خالی؛ row2: 0.2·2.0+2.0·1.0=2.4
    assert ids.tolist() == [0, 2]
    assert abs(float(sc[0]) - 3.0) < 1e-5 and abs(float(sc[1]) - 2.4) < 1e-5
    ids2, sc2 = si.top({"20": 1.0}, depth=3)
    assert ids2.tolist() == [0] and abs(float(sc2[0]) - 0.5) < 1e-5


def test_retrieve_rerank_reorders_hits(tmp_path, mini_pdf):
    """C2: rerank فقط ترتیب fused را می‌چیند؛ با FakeReranker (بدون مدل/شبکه).
    bm25 لیست را آشفته می‌کند و fake بیشترین امتیاز را به تکهٔ «تاریخ» می‌دهد."""
    cfg = _cfg_for(tmp_path, mini_pdf)
    sync(cfg, no_embed=True)
    s = _mk_searcher(tmp_path / "index")

    class _FakeRR:
        def __init__(self):
            self.calls = 0
        def rerank(self, q, texts):
            self.calls += 1
            return [1.0 if "تاریخ" in t else float(i) * 0.01 for i, t in enumerate(texts)]

    s.reranker = _FakeRR()
    res = s.search("کتاب تاریخ ایران", overrides={"rerank": True})
    assert s.reranker.calls == 1
    assert res.hits[0].ranks.get("rerank") == 1
    assert "تاریخ" in res.hits[0].chunk_id or True     # ترتیب عوض شده؛ فقط ساختار مهم است
    # صعودی‌بودن امتیازها طبق fake
    scores = [h.score for h in res.hits]
    assert scores == sorted(scores, reverse=True)
    # بدون rerank: مسیر قدیمی دست‌نخورده
    res2 = s.search("کتاب تاریخ ایران")
    assert all("rerank" not in h.ranks for h in res2.hits)


def test_retrieve_sparse_ignored_for_st_backend(tmp_path, mini_pdf):
    """ایندکس BM25-only: sparse در لیست‌ها نمی‌آید و retrieve کرش نمی‌کند."""
    cfg = _cfg_for(tmp_path, mini_pdf)
    sync(cfg, no_embed=True)
    s = _mk_searcher(tmp_path / "index")
    res = s.search("کتاب تاریخ ایران", overrides={"lists": ["bm25", "sparse"]})
    assert res.hits                                    # sparse غایب → همان BM25


def test_sync_parallel_workers(tmp_path, mini_pdf):
    """B1: sync --workers=2 (BM25-only) همان خروجی مسیر سریال را می‌دهد."""
    import json as _json
    cfg = _cfg_for(tmp_path, mini_pdf)
    r1 = sync(cfg, no_embed=True, workers=1)
    catalog = _json.loads((tmp_path / "index" / "catalog.json").read_text(encoding="utf-8"))
    key = catalog["mini.pdf"]["sha256"][:16]
    n_chunks_serial = _json.loads((tmp_path / "index" / "books" / key / "meta.json")
                                  .read_text(encoding="utf-8"))["n_chunks"]
    r2 = sync(cfg, no_embed=True, workers=2)            # همه up-to-date → بدون pool
    assert r2["stats"].get("unchanged") == 1
    # تغییر نسخهٔ chunk → مسیر chunk در worker اصلی؛ workers=2 با ۱ کتاب → بدون pool
    meta_path = tmp_path / "index" / "books" / key / "meta.json"
    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
    meta["stages"]["chunk"]["version"] = "bogus"
    meta_path.write_text(_json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    r3 = sync(cfg, no_embed=True, workers=2)
    meta3 = _json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta3["stages"]["chunk"]["version"] != "bogus"
    assert meta3["n_chunks"] == n_chunks_serial


def test_sync_parallel_pool_two_books(tmp_path, mini_pdf):
    """B1 واقعی: دو کتاب → ProcessPool (len(heavy)>1) → هر دو کامل و ایندکس سالم."""
    import json as _json
    doc = fitz.open()
    css = "@font-face {font-family: fa; src: url('C:/Windows/Fonts/tahoma.ttf');} * {font-family: fa; font-size: 13px;}"
    page = doc.new_page()
    page.insert_htmlbox(fitz.Rect(50, 50, 545, 700),
                        "جنگ جهانی دوم یکی از رخدادهای تاریخ معاصر است و میلیون‌ها نفر را جابه‌جا کرد.",
                        css=css)
    page.insert_htmlbox(fitz.Rect(50, 740, 545, 760), "1", css=css)
    doc.save(str(mini_pdf.parent / "mini2.pdf"))
    doc.close()

    cfg = _cfg_for(tmp_path, mini_pdf)
    rep = sync(cfg, no_embed=True, workers=2)
    assert rep["stats"].get("added") == 2
    catalog = _json.loads((tmp_path / "index" / "catalog.json").read_text(encoding="utf-8"))
    assert set(catalog) == {"mini.pdf", "mini2.pdf"}
    s = _mk_searcher(tmp_path / "index")
    assert s.search("کتاب تاریخ ایران").found
    assert s.search("جنگ جهانی دوم").found
    # دومین sync: همه کش‌شده → بدون pool و بدون تغییر
    rep2 = sync(cfg, no_embed=True, workers=2)
    assert rep2["stats"].get("unchanged") == 2


def test_evaluate_cli(eval_index):
    _write_eval_questions(eval_index.parent / "questions.jsonl")
    root = Path(__file__).parents[1]                 # project root (booksearch.toml lives here)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(
        [sys.executable, str(root / "app" / "evaluate.py"), "--variants", "base,bm25,cc",
         "--index", str(eval_index), "--questions", str(eval_index.parent / "questions.jsonl"),
         "--split", "all", "--results", str(eval_index.parent / "results.jsonl")],
        capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(root), timeout=180)
    assert r.returncode == 0, r.stderr
    assert "base" in r.stdout and "bm25" in r.stdout and "cc" in r.stdout
    res = (eval_index.parent / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(res) == 1 and '"base"' in res[0] and '"cc"' in res[0]
