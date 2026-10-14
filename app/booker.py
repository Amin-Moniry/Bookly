"""
booker.py — chat layer of BookSearch-FA (README §1.2: only the numbered context
reaches the LLM, the model must answer with citation [n] + book + printed page).

Terminal-first (the UI comes later):

    python booker.py                      # chat REPL — «Booker ❯ »
    python booker.py "سؤال"               # one question, one answer, exit
    python booker.py --index index ...    # same flags as search.py

Behavior:
  * every question first goes through Searcher (hybrid BM25+dense);
    if nothing valid is found (`Result.found == false`) we print «نتیجه‌ای در
    کتاب‌ها یافت نشد.» and do NOT call the API — the model never sees an empty
    context, so it cannot hallucinate one (README §8).
  * otherwise the passages are formatted with format_context() ([n] title, pp.)
    and sent as the ONLY user content; the system prompt forbids outside
    knowledge and demands citations.
  * chat history keeps the last N Q/A pairs (without their old contexts) so
    follow-up questions («و دربارهٔ X چی؟») work naturally.

The endpoint is any OpenAI-compatible API (`[llm]` in booksearch.toml) —
currently mistral-large via xkiro.com instead of the local Qwen3-4B planned in
the README; swapping later is a config change only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from search import Searcher, load_config


# ----------------------------------------------------------------------------
# Config: [llm] in booksearch.toml + env var BOOKLY_API_KEY (env wins).
# The toml shipped in the repo must stay key-free; keys live in env vars or in
# llm_models.json (which is git-ignored).
# ----------------------------------------------------------------------------
@dataclass
class LLMConfig:
    base_url: str = "https://api.xkiro.com/v1"
    api_key: str = ""
    model: str = "mistralai/mistral-large-2512"
    temperature: float = 0.2
    max_tokens: int = 700
    history_turns: int = 4          # تعداد جفت سؤال/جواب قبلی که در چت می‌ماند
    timeout_s: float = 90.0
    context_chars: int = 9000       # سقف نرم context شماره‌دار (کاراکتر)
    prompt_lang: str = "auto"       # fa | en | auto (تشخیص از سؤال) — UI صریح می‌فرستد


def load_llm_config(path: str | Path = "booksearch.toml") -> LLMConfig:
    cfg = LLMConfig()
    p = Path(path)
    if not p.exists():
        return cfg
    import tomllib  # Python 3.11+
    raw = tomllib.loads(p.read_text(encoding="utf-8")).get("llm", {})
    for f in ("base_url", "api_key", "model", "temperature", "max_tokens",
              "history_turns", "timeout_s", "context_chars"):
        if f in raw:
            setattr(cfg, f, raw[f])
    if os.environ.get("BOOKLY_API_KEY"):          # env var همیشه بر toml اولویت دارد
        cfg.api_key = os.environ["BOOKLY_API_KEY"]
    return cfg


# ----------------------------------------------------------------------------
# Prompt assembly (pure functions — unit-testable without any network)
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "شما دستیار فنی «Booker» هستید و فقط از گزیده‌های شماره‌دار «کتاب‌ها» پاسخ می‌دهید.\n"
    "\n"
    "قواعد پاسخ:\n"
    "۱. ساختار: اول جواب مستقیم در حداکثر ۲ خط؛ بعد در صورت نیاز چند bullet کوتاه."
    " مقدمه، تعارف و تکرار صورت سؤال ممنوع.\n"
    "۲. قالب‌بندی سبک: فقط bullet و **bold** کوتاه؛ از هدر (#)، جدول، بلوک کد و"
    " جداکننده (---) استفاده نکن؛ خروجی برای ترمینال است.\n"
    "۳. ارجاع: در پایانِ هر بند کلیدی مثل [1]؛ اما بخش جداگانهٔ «مرجع/منابع» در"
    " انتهای پاسخ ننویس — ابزار خودش منابع را چاپ می‌کند.\n"
    "۴. زبان: همان زبان سؤال (پیش‌فرض فارسی روان)؛ اصطلاحات فنی و فلگ‌های ابزارها"
    " به همان شکل انگلیسی بمانند.\n"
    "۵. فقط از گزیده‌ها؛ اگر جواب کامل در گزیده‌ها نیست، فقط بخش موجود را بگو و برای"
    " بخش غایب صریح بنویس «در کتاب‌ها نیست». هرگز حدس نزن و از دانش بیرونی استفاده نکن.\n"
    "۶. اگر سؤال چندبخشی است، هر بخش را جدا و کوتاه جواب بده."
)

SYSTEM_PROMPT_EN = (
    "You are 'Booker', a technical assistant that answers ONLY from the numbered"
    " excerpts labeled 'BOOKS'.\n"
    "\n"
    "Answer rules:\n"
    "1. Structure: direct answer in at most 2 lines first; then a few short bullets"
    " if needed. No preamble, no pleasantries, no restating the question.\n"
    "2. Light formatting: short bullets and **bold** only; no headers (#), tables,"
    " code blocks, or separators (---).\n"
    "3. Citations: end each key claim with [1]-style markers; do NOT write a separate"
    " 'Sources' section — the tool prints sources itself.\n"
    "4. Language: match the user's language. If the question is in Persian (Farsi),"
    " answer in fluent Persian; otherwise answer in English. Technical terms and tool"
    " flags stay in English as written.\n"
    "5. Excerpts only: if the full answer is not in the excerpts, say what is present"
    " and explicitly write 'Not found in the books' for the missing part. Never guess"
    " or use outside knowledge.\n"
    "6. Multi-part questions: answer each part separately and briefly."
)

# Persian-script detector: any letter in Arabic/Persian ranges -> Persian prompt
_FA_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def system_prompt_for(question: str, lang: str = "auto") -> str:
    """زبان پاسخ: «fa»/«en» صریح (انتخاب کاربر در UI) یا «auto» = تشخیص از خود سؤال."""
    if lang == "en":
        return SYSTEM_PROMPT_EN
    if lang == "fa":
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT if _FA_RE.search(question or "") else SYSTEM_PROMPT_EN


def trim_context(context: str, limit: int) -> str:
    """Hard cap on context chars (tokens ≈ chars/3 for these books)."""
    return context if len(context) <= limit else context[:limit] + "\n…"


def build_messages(question: str, context: str, history: list[tuple[str, str]],
                   cfg: LLMConfig) -> list[dict]:
    """system + (old Q/A turns) + (fresh numbered context + new question).

    Old contexts are NOT replayed — each turn carries only its own answer text,
    so history stays small and the newest evidence is always authoritative.
    """
    msgs: list[dict] = [{"role": "system",
                         "content": system_prompt_for(question, getattr(cfg, "prompt_lang", "auto"))}]
    for q_old, a_old in history[-int(cfg.history_turns):]:
        msgs.append({"role": "user", "content": q_old})
        msgs.append({"role": "assistant", "content": a_old})
    user = (f"گزیده‌های مرتبط از کتاب‌ها:\n\n{trim_context(context, cfg.context_chars)}\n\n"
            f"سؤال: {question}")
    msgs.append({"role": "user", "content": user})
    return msgs


def parse_sse_line(line: str) -> dict | None:
    """One SSE line from an OpenAI-compatible stream -> parsed chunk (or None)."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def delta_of(chunk: dict) -> str:
    """Extract the content delta from one streamed chunk, tolerating variants."""
    choices = chunk.get("choices") or [{}]
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""


# ----------------------------------------------------------------------------
# API call — stream first (claude-CLI feel), one non-stream retry as fallback
# ----------------------------------------------------------------------------
def chat_once(client: httpx.Client, cfg: LLMConfig, messages: list[dict],
              echo: bool = True, on_delta=None) -> str:
    """One streamed completion. `on_delta(str)` receives each text chunk live
    (used by the web UI in server.py); `echo=True` also prints to stdout."""
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}   # مدل‌های لوکال بدون کلید
    payload = {"model": cfg.model, "messages": messages, "stream": True,
               "temperature": cfg.temperature, "max_tokens": cfg.max_tokens}

    def _emit(text: str) -> None:
        if on_delta:
            on_delta(text)
        if echo:
            print(text, end="", flush=True)

    def _non_stream() -> str:
        payload["stream"] = False
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        text = (r.json()["choices"][0].get("message") or {}).get("content") or ""
        if echo:
            print(text, flush=True)
        if on_delta:
            on_delta(text)
        return text

    try:
        parts: list[str] = []
        with client.stream("POST", url, headers=headers, json=payload) as r:
            if r.status_code >= 400:
                body = r.read().decode("utf-8", "replace")[:300]
                raise RuntimeError(f"HTTP {r.status_code}: {body}")
            for line in r.iter_lines():
                chunk = parse_sse_line(line)
                if chunk is None:
                    continue
                delta = delta_of(chunk)
                if delta:
                    parts.append(delta)
                    _emit(delta)
        if echo:
            print(flush=True)
        text = "".join(parts)
        if text.strip():
            return text
        return _non_stream()          # some gateways ignore stream=True
    except (httpx.HTTPError, json.JSONDecodeError):
        return _non_stream()


# ----------------------------------------------------------------------------
# REPL
# ----------------------------------------------------------------------------
def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Booker — chat with your books (RAG)")
    ap.add_argument("query", nargs="*", help="optional single question (else REPL)")
    ap.add_argument("--index", default="index")
    args = ap.parse_args()

    llm = load_llm_config()
    if not llm.api_key:
        sys.exit("[llm] api_key is empty — set BOOKLY_API_KEY (env) or add a model in the web UI.")

    try:
        searcher = Searcher(index_dir=args.index)
    except FileNotFoundError as e:
        sys.exit(str(e))

    history: list[tuple[str, str]] = []

    def one(q: str) -> None:
        searcher.reload_if_changed()
        if searcher._dense is not None and not hasattr(searcher, "embedder"):
            searcher.embedder = searcher._make_embedder()
        res = searcher.search(q)
        if not res.found:
            print("نتیجه‌ای در کتاب‌ها یافت نشد.")     # no API call — README §8
            return
        print(f"🔎 جستجو: {len(res.passages)} passage در {res.timings_ms['total_ms']}ms")
        context = searcher.format_context(res.passages)
        messages = build_messages(q, context, history, llm)
        try:
            with httpx.Client(timeout=llm.timeout_s) as client:
                answer = chat_once(client, llm, messages)
        except RuntimeError as e:
            print(f"\n[llm error] {e}")
            return
        if answer.strip():
            history.append((q, answer))
        if res.passages:
            srcs = " · ".join(f"[{p.n}] «{p.book_title}» ص {p.pages}" for p in res.passages[:3])
            print(f"منابع: {srcs}")

    if args.query:
        one(" ".join(args.query))
        return

    print("Booker — از کتاب‌ها بپرسید. (/exit یا Ctrl+C برای خروج)")
    print(f"مدل: {llm.model}   ایندکس: {args.index}\n")
    while True:
        try:
            q = input("Booker ❯ ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not q:
            continue
        if q.lower() in ("/exit", "/quit", "exit", "quit"):
            break
        one(q)


if __name__ == "__main__":
    main()
