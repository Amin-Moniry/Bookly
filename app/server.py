"""
server.py — web UI for BookSearch-FA (the «run button» of booker.py).

    python server.py            # http://127.0.0.1:8765  → open the browser

Agentic loop: the model answers directly for greetings/small-talk, and may
issue `SEARCH: query` lines to pull real passages from the books mid-answer
(results are fed back with global [n] numbering, so citations stay correct).

  GET  /                 the single-page chat UI
  GET  /api/books        manifest → titles/page-counts
  GET  /api/models       model store: {active, models:[{name, base_url, api_key, model, ...}]}
  POST /api/models            add/update one model (OpenAI-compatible; local or API)
  POST /api/models/delete     {name}
  POST /api/models/activate   {name}
  POST /api/models/test       {name} یا inline cfg → ping کوچک به chat/completions
  POST /api/chat         SSE stream: model → stage → delta → sources → done
  POST /api/sync         background `sync`      GET /api/sync/status
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

import booker
from booker import LLMConfig, build_messages, chat_once, load_llm_config
from search import Searcher, load_config

ROOT = Path(__file__).parent
_INDEX_DIR = "index"
_searcher: Searcher | None = None
_history: list[tuple[str, str]] = []
MAX_SEARCH_ROUNDS = 3          # حداکثر SEARCH: هایی که مدل در یک پاسخ می‌زند

TOOL_RULES = (
    "\n\nابزار جستجو در کتاب‌ها:\n"
    "- اگر برای پاسخ به گزیدهٔ کتاب نیاز داری یا عبارت اول نتیجه نداد، فقط یک خط به شکل "
    "«SEARCH: عبارت جستجو» بنویس (کلیدواژه‌ای، فارسی یا انگلیسی). نتایج به‌صورت "
    "گزیده‌های شماره‌دار برمی‌گردند؛ بعد پاسخ کامل را با ارجاع [n] بده.\n"
    "- حداکثر ۳ بار جستجو کن؛ بعد از آخرین نتیجه حتماً پاسخ نهایی را بنویس.\n"
    "- برای سلام‌وعلیک، گپ یا سؤالی که به کتاب‌ها ربطی ندارد، SEARCH ننویس و مستقیم و "
    "کوتاه پاسخ بده.\n"
    "- اگر بعد از جستجوها جواب در کتاب‌ها نبود، صریح بگو «در کتاب‌ها نیست» و حدس نزن."
)

TOOL_RULES_EN = (
    "\n\nBook-search tool:\n"
    "- If you need book excerpts to answer, or the first wording found nothing, output a "
    "single line 'SEARCH: query' (keywords, Persian or English). Results come back as "
    "numbered excerpts; then give the full answer with [n] citations.\n"
    "- Search at most 3 times; after the last result always write the final answer.\n"
    "- For greetings, small talk, or questions unrelated to the books, do NOT search — "
    "answer directly and briefly.\n"
    "- If the answer is not in the books after searching, say so explicitly ('Not found "
    "in the books') and never guess."
)


def _get_searcher() -> Searcher:
    """Prefer the user's own index; fall back to the shipped demo index
    (index/sample, built from demo/sample-book.pdf) until the first real
    `app/ingest.py sync` produces index/merged/CURRENT."""
    global _searcher
    want = _INDEX_DIR
    if want == "index" and not (Path(want) / "merged" / "CURRENT").exists() \
            and (Path(want) / "sample" / "merged" / "CURRENT").exists():
        want = str(Path(want) / "sample")            # fresh clone → demo index
    if _searcher is None or str(_searcher.index_dir) != want:
        _searcher = Searcher(index_dir=want)
    return _searcher


# ---- model store: هر مدل (لوکال یا API) یک رکورد OpenAI-سازگار --------------
# فایل llm_models.json کنار server.py؛ ورود اول از [llm] booksearch.toml مهاجرت می‌کند.
_MODELS_PATH = ROOT / "llm_models.json"
_models_lock = threading.Lock()


def _default_store() -> dict:
    # هیچ پیش‌فرضی: کاربر خودش مدل (لوکال یا API) را از پنل UI اضافه می‌کند.
    return {"active": "", "models": []}


def load_models() -> dict:
    with _models_lock:
        if not _MODELS_PATH.exists():
            store = _default_store()
            _MODELS_PATH.write_text(json.dumps(store, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
            return store
        try:
            st = json.loads(_MODELS_PATH.read_text(encoding="utf-8"))
            if not isinstance(st.get("models"), list) or not st["models"]:
                return _default_store()
            return st
        except Exception:                             # noqa: BLE001 — فایل خراب → بازسازی
            return _default_store()


def save_models(store: dict) -> None:
    with _models_lock:
        tmp = str(_MODELS_PATH) + ".tmp"
        Path(tmp).write_text(json.dumps(store, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        os.replace(tmp, _MODELS_PATH)


def _llm_cfg(name: str | None = None) -> tuple[LLMConfig | None, str]:
    """LLMConfig برای مدل درخواستی (یا active). بدون مدلِ ثبت‌شده → (None, "").
    هیچ fallback ای به toml نیست — کاربر باید خودش مدل اضافه کند."""
    if name in ("None", "null", "undefined"):        # ورودی خراب از سمت کلاینت
        name = ""
    store = load_models()
    want = name or store.get("active") or ""
    for m in store["models"]:
        if m["name"] == want:
            cfg = LLMConfig(base_url=m.get("base_url", ""),
                            api_key=m.get("api_key", "") or os.environ.get("BOOKLY_API_KEY", ""),
                            model=m.get("model", ""),
                            temperature=float(m.get("temperature", 0.2)),
                            max_tokens=int(m.get("max_tokens", 700)))
            return cfg, m["name"]
    return None, ""


# ---- sync state ------------------------------------------------------------
_sync_lock = threading.Lock()
_sync_running = False
_sync_log: list[str] = []


def _slog(line: str) -> None:
    _sync_log.append(line)
    if len(_sync_log) > 400:
        del _sync_log[:200]


def _sync_worker() -> None:
    global _sync_running
    import io, contextlib
    _sync_running = True
    try:
        from ingest import sync as ingest_sync
        cfg = load_config()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report = ingest_sync(cfg)
        for line in buf.getvalue().splitlines()[-120:]:
            _slog(line)
        _slog(f"✓ sync finished — {report['stats']}")
    except Exception as e:                            # noqa: BLE001
        _slog(f"✗ sync failed: {e}")
    finally:
        _sync_running = False


# ---- agentic chat ----------------------------------------------------------
class SearchRequested(Exception):
    def __init__(self, query: str):
        self.query = query


def _make_gate(emit):
    """Stream gate: holds back text that may become a `SEARCH:` line, routes
    completed SEARCH lines to the caller, passes everything else through."""
    prefix = "SEARCH:"
    state = {"buf": ""}

    def feed(t: str) -> None:
        state["buf"] += t
        while True:
            nl = state["buf"].find("\n")
            if nl == -1:
                stripped = state["buf"].lstrip()
                # تا وقتی newline نیامده، هر بافری که «SEARCH:» است یا می‌تواند بشود، نگه داشته شود
                if stripped and (prefix.startswith(stripped) or stripped.startswith(prefix)):
                    return
                emit(state["buf"]); state["buf"] = ""
                return
            line, state["buf"] = state["buf"][:nl], state["buf"][nl + 1:]
            st = line.strip()
            if st.startswith(prefix):
                raise SearchRequested(st[len(prefix):].strip())
            emit(line + "\n")

    def flush() -> str | None:
        """Emit whatever is left. If the trailing (newline-less) buffer is a
        SEARCH line, return its query instead of leaking it into the answer."""
        st = state["buf"].strip()
        state["buf"] = ""
        if st.startswith(prefix):
            query = st[len(prefix):].strip()
            if query:
                return query
            return None
        if st:
            emit(st)
        return None

    return feed, flush


class _Agentic:
    """One Q/A turn with the SEARCH: tool loop. Collects the streamed answer,
    live stage events, and deduped sources across rounds."""

    def __init__(self, s: Searcher, llm: LLMConfig, history):
        self.s, self.llm, self.history = s, llm, history
        self.stages: list[dict] = []
        self.text = ""
        self.srcs: list[dict] = []
        self._seen_rows: set[int] = set()

    def emit(self, t: str) -> None:
        self.text += t

    def stage(self, name: str, msg: str, done: bool = False) -> None:
        self.stages.append({"name": name, "msg": msg, "done": done})

    def _search(self, query: str) -> bool:
        self.stage("search", f"SEARCH: {query}")
        res = self.s.search(query)
        if res.found and res.passages:
            added = 0
            for p in res.passages:
                if any(r in self._seen_rows for r in p.rows):
                    continue
                for r in p.rows:
                    self._seen_rows.add(r)
                p.n = len(self.srcs) + 1
                self.srcs.append(p)
                added += 1
            self.stage("read", f"{added} گزیدهٔ جدید از «{res.passages[0].book_title}» "
                               f"ص {res.passages[0].pages}")
            return True
        self.stage("search", f"برای «{query}» چیز مرتبطی پیدا نشد")
        return False

    def _context_block(self) -> str:
        return "\n\n".join(f"[{p.n}] «{p.book_title}»، ص {p.pages}\n{p.text}"
                           for p in self.srcs)

    def run(self, q: str) -> None:
        pl = getattr(self.llm, "prompt_lang", "auto")
        if pl == "en":
            system = booker.SYSTEM_PROMPT_EN
            rules = TOOL_RULES_EN
            L = lambda fa, en: en                          # noqa: E731
        elif pl == "fa":
            system = booker.SYSTEM_PROMPT
            rules = TOOL_RULES
            L = lambda fa, en: fa                          # noqa: E731
        else:
            system = booker.system_prompt_for(q)
            rules = TOOL_RULES if booker._FA_RE.search(q or "") else TOOL_RULES_EN
            L = lambda fa, en: (fa if booker._FA_RE.search(q or "") else en)  # noqa: E731
        msgs = [{"role": "system", "content": system + rules}]
        for old_q, old_a in self.history[-int(self.llm.history_turns):]:
            msgs += [{"role": "user", "content": old_q},
                     {"role": "assistant", "content": old_a}]

        self.stage("search", L("جستجوی اولیه با خود سؤال…", "Initial search with the question…"))
        first = self._search(q)
        if first:
            user = (L("گزیده‌های مرتبط از کتاب‌ها:", "Relevant excerpts from the books:")
                    + f"\n\n{self._context_block()}\n\n"
                      + L("سؤال: ", "Question: ") + q)
        else:
            user = (q + "\n\n"
                    + L("(جستجوی اولیه چیزی مرتبط پیدا نکرد. اگر عبارت دیگری احتمال "
                        "دارد در کتاب‌ها باشد، یک خط SEARCH: بنویس؛ اگر سؤال به کتاب‌ها "
                        "ربطی ندارد، مستقیم پاسخ بده.)",
                        "(The initial search found nothing relevant. If another wording might "
                        "be in the books, output a SEARCH: line; if the question is unrelated "
                        "to the books, answer directly.)"))
        msgs.append({"role": "user", "content": user})

        with httpx.Client(timeout=self.llm.timeout_s) as client:
            for _round in range(MAX_SEARCH_ROUNDS + 1):
                self.stage("llm", L(f"پرسش از {self.llm.model}…",
                                    f"Asking {self.llm.model}…"), done=True)
                gate_feed, gate_flush = _make_gate(self.emit)
                requested: str | None = None
                try:
                    chat_once(client, self.llm, msgs, echo=False, on_delta=gate_feed)
                    requested = gate_flush()      # SEARCH بدون newlineِ پایانی
                except SearchRequested as sr:
                    requested = sr.query          # SEARCH وسط استریم (با newline)
                if not requested:
                    break                         # پاسخ نهایی کامل شد
                partial = self.text.strip()
                msgs.append({"role": "assistant",
                             "content": (partial + f"\nSEARCH: {requested}").strip()})
                self._search(requested)
                msgs.append({"role": "user",
                             "content": (L("نتایج جستجو:", "Search results:")
                                         + "\n\n" + self._context_block() + "\n\n"
                                         + L(f"حالا سؤال اصلی کاربر را کامل پاسخ بده: «{q}»",
                                             f"Now answer the user's original question in full: '{q}'")
                                         + "\n"
                                         + L("پاسخ کامل، ساختارمند و مفید با ارجاع [n] به هر شاهد؛ "
                                             "فقط اگر بخشی از سؤال در هیچ گزیده‌ای نبود، همان بخش را صریح "
                                             "«در کتاب‌ها نیست» بنویس.",
                                             "Give a complete, structured, useful answer with [n] citations "
                                             "for every piece of evidence; for any part not present in any "
                                             "excerpt, explicitly write 'Not found in the books'."))})


# ----------------------------------------------------------------------------
def _sse(handler: BaseHTTPRequestHandler, event: str, data) -> None:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    handler.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
    handler.wfile.flush()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (ROOT / "web" / "index.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif self.path == "/api/books":
            try:
                s = _get_searcher()
                books = [{"title": b.get("title") or Path(b["file"]).stem,
                          "pages": b.get("n_pages") or b.get("n"), "chunks": b["n"]}
                         for b in s.manifest["books"]]
                self._json(200, {"books": books})
            except Exception as e:                    # noqa: BLE001
                self._json(200, {"books": [], "error": str(e)})
        elif self.path == "/api/sync/status":
            self._json(200, {"running": _sync_running, "log": _sync_log[-80:]})
        elif self.path == "/api/models":
            st = load_models()
            self._json(200, {"active": st.get("active", ""), "models": st["models"]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/chat":
            self._chat()
        elif self.path == "/api/sync":
            with _sync_lock:
                if _sync_running:
                    self._json(409, {"error": "sync already running"})
                    return
                threading.Thread(target=_sync_worker, daemon=True).start()
            self._json(200, {"ok": True})
        elif self.path in ("/api/models", "/api/models/delete",
                           "/api/models/activate", "/api/models/test"):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            self._models_api(self.path, body)
        else:
            self._json(404, {"error": "not found"})

    def _models_api(self, route: str, body: dict) -> None:
        st = load_models()
        if route == "/api/models":
            name = str(body.get("name", "")).strip()
            base_url = str(body.get("base_url", "")).strip().rstrip("/")
            model = str(body.get("model", "")).strip()
            if not name or not base_url or not model:
                self._json(400, {"error": "name، base_url و model لازم است"})
                return
            rec = {"name": name, "base_url": base_url, "api_key": str(body.get("api_key", "")),
                   "model": model,
                   "temperature": float(body.get("temperature", 0.2) or 0.2),
                   "max_tokens": int(body.get("max_tokens", 700) or 700)}
            for i, m in enumerate(st["models"]):
                if m["name"] == name:
                    st["models"][i] = rec
                    break
            else:
                st["models"].append(rec)
            if not st.get("active"):
                st["active"] = name
            save_models(st)
            self._json(200, {"ok": True, "active": st["active"], "models": st["models"]})
        elif route == "/api/models/delete":
            name = str(body.get("name", ""))
            before = len(st["models"])
            st["models"] = [m for m in st["models"] if m["name"] != name]
            if len(st["models"]) == before:
                self._json(404, {"error": f"model not found: {name}"})
                return
            if st.get("active") == name:
                st["active"] = st["models"][0]["name"] if st["models"] else ""
            save_models(st)
            self._json(200, {"ok": True, "active": st.get("active", ""), "models": st["models"]})
        elif route == "/api/models/activate":
            name = str(body.get("name", ""))
            if not any(m["name"] == name for m in st["models"]):
                self._json(404, {"error": f"model not found: {name}"})
                return
            st["active"] = name
            save_models(st)
            self._json(200, {"ok": True, "active": name})
        else:  # /api/models/test — ping کوچک به chat/completions
            name = str(body.get("name", ""))
            rec = next((m for m in st["models"] if m["name"] == name), None) or body
            base_url = str(rec.get("base_url", "")).strip().rstrip("/")
            if not base_url:
                self._json(400, {"ok": False, "error": "base_url خالی است"})
                return
            url = base_url + "/chat/completions"
            headers = {"Authorization": f"Bearer {rec.get('api_key', '')}"} \
                if rec.get("api_key") else {}
            payload = {"model": rec.get("model", ""), "stream": False, "max_tokens": 8,
                       "messages": [{"role": "user", "content": "ping"}]}
            try:
                r = httpx.post(url, headers=headers, json=payload, timeout=25.0)
                if r.status_code >= 400:
                    self._json(200, {"ok": False,
                                     "error": f"HTTP {r.status_code}: {r.text[:200]}"})
                    return
                msg = ((r.json()["choices"][0].get("message") or {}).get("content") or "").strip()
                self._json(200, {"ok": True, "reply": msg[:80] or "(خالی)"})
            except Exception as e:                    # noqa: BLE001
                self._json(200, {"ok": False, "error": f"{type(e).__name__}: {e}"})

    def _chat(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        q = str(req.get("q", "")).strip()
        if not q:
            self._json(400, {"error": "empty question"})
            return
        cfg, used_name = _llm_cfg(str(req.get("model", "") or "").strip() or None)
        ui_lang = str(req.get("lang", "auto") or "auto")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")   # SSE: کلاینت به EOF نیاز دارد
        self.end_headers()
        t0 = time.perf_counter()
        try:
            if cfg is None:
                msg = ("هیچ مدلی ثبت نشده — از دکمهٔ ✨ در هدر یک مدل (لوکال یا API) اضافه کن."
                       if ui_lang == "fa" else
                       "No model configured — click the ✨ button in the header to add one (local or API).")
                _sse(self, "error", {"msg": msg})
                return
            cfg.prompt_lang = ui_lang
            _sse(self, "model", {"name": used_name, "model": cfg.model})
            s = _get_searcher()
            s.reload_if_changed()
            if s._dense is not None and not hasattr(s, "embedder"):
                s.embedder = s._make_embedder()

            agent = _Agentic(s, cfg, _history)

            def stage_ev(name, msg, done=False):
                _sse(self, "stage", {"name": name, "msg": msg, "done": done})

            # replay the auto-search stages that already happened in run()? no —
            # run() emits nothing yet; bridge stages live via a thin wrapper:
            orig_stage = agent.stage
            agent.stage = lambda name, msg, done=False: (orig_stage(name, msg, done),
                                                         stage_ev(name, msg, done))
            orig_emit = agent.emit
            agent.emit = lambda t: (orig_emit(t), _sse(self, "delta", {"t": t}))

            agent.run(q)

            if agent.text.strip():
                _history.append((q, agent.text))
                if len(_history) > 12:
                    del _history[:6]
            _sse(self, "sources",
                 {"sources": [{"n": p.n, "title": p.book_title, "pages": p.pages,
                               "preview": p.text[:180]} for p in agent.srcs]})
            _sse(self, "done",
                 {"elapsed_ms": round((time.perf_counter() - t0) * 1000)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:                        # noqa: BLE001
            try:
                _sse(self, "error", {"msg": f"{type(e).__name__}: {e}"})
            except Exception:
                pass


def main() -> None:
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    srv = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Booker UI  →  http://127.0.0.1:8765   (Ctrl+C برای خروج)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
