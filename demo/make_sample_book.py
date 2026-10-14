"""Build demo/sample-book.pdf (neutral sample content) + BM25-only index at index/sample/.

Run once; the artifacts are tracked in git so fresh clones get a working demo.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "app"))

import fitz


PAGES = [
    "A sample-book made for the Bookly demo. It carries a few calm pages so you can "
    "try the whole app right away: search, reading-room chat and page citations, all "
    "before you index a single book of your own library on this machine.",

    "Page 1. The lighthouse keeper climbs the tower every evening and greets the "
    "harbour with three short flashes of the lamp. Calm weather keeps the lamp "
    "glass clean, so the keeper wipes it twice before sunset. The fog signal "
    "always runs one minute before dawn, and sailors say the gentle pattern of "
    "light feels like a promise that the shoreline is exactly where the chart "
    "promises it to be. On stormy nights the keeper doubles the oil reserve.",

    "Page 2. This chapter describes a small reading room beside the harbour with "
    "two wooden benches and a tall bookshelf filled with poetry volumes and "
    "sea charts. A brass kettle warms on the little stove, and paper borrow "
    "cards record who took which tale home for the evening. The catalogue was "
    "rebuilt last spring, and the librarian arranged the shelves chronologically "
    "so the oldest journal sits at eye level beside the warm window.",

    "Page 3. The final page lists the house rules of this sample-book. Keep "
    "voices low after nine bells. Return the borrowed tales before Sunday so "
    "the next traveller finds them waiting. Water the potted fern every third "
    "morning with rain collected in the old barrel. And each week write the "
    "favourite quote of the house on the slate board near the door, so visiting "
    "sailors leave knowing what the reading room loved most.",

    "Page 4. A short appendix collects handy numbers from around the harbour: "
    "twelve boats registered for the spring parade, seven poetry volumes on the "
    "top shelf, three rainy days last month, and one friendly dog that guards "
    "the pier gate. For demo purposes, ask Bookly a question like how many "
    "poetry volumes sit on the top shelf, and it will answer with page numbers.",
]


def main() -> None:
    root = Path(__file__).parent
    out = root / "demo" / "sample-book.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for i, txt in enumerate(PAGES):
        page = doc.new_page()
        page.insert_text((72, 100), txt[:95], fontsize=13)
        page.insert_text((72, 130), txt[95:190], fontsize=12)
        page.insert_text((72, 160), txt[190:285], fontsize=12)
        page.insert_text((72, 190), txt[285:380], fontsize=12)
        page.insert_text((72, 220), txt[380:], fontsize=12)
        page.insert_text((72, 760), str(i + 1), fontsize=10)  # printed page number (footer)
    doc.save(str(out))
    doc.close()
    print(f"wrote {out}")

    demo_idx = root / "index" / "sample"
    if demo_idx.exists():
        shutil.rmtree(demo_idx)               # clean rebuild after content change

    from ingest import sync
    from search import load_config

    cfg = load_config()
    cfg.paths["library"] = str(root / "demo")
    cfg.paths["index"] = str(demo_idx)
    report = sync(cfg, no_embed=True)         # BM25-only: fresh clone works without torch
    print("sync:", report["stats"])


if __name__ == "__main__":
    main()
