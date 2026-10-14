# eval/

Evaluation data for retrieval quality.

- **`questions.sample.jsonl`** — tracked sample of the format only. Contains
  generic, made-up sentences; no real book content.
- **`questions.jsonl`** — your own questions + page-quoted witnesses.
  **git-ignored on purpose**: it contains direct quotes from *your* books, so
  it must never be pushed to a public repo. Build it locally, run
  `python app/evaluate.py --questions eval/questions.jsonl ...`, keep it local.
- **`results.jsonl`** — append-only run logs (also git-ignored).

Don't rename the sample to `questions.jsonl` for real eval work: copy it, edit
it, and rely on the ignore rule so nothing accidental gets committed.
