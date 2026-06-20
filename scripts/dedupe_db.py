"""One-time migration: collapse duplicate fixtures in an existing touchline.db.

The natural key changed from order-dependent (date|home|away) to order-independent
(date|sorted teams), and team-name normalization now folds "&" spellings. Rows
written under the old scheme keep their old keys, so `upsert` alone won't remove
the duplicates already on disk. This rewrites the matches table through
`dedupe_matches`, which also re-aligns goals when feeds disagreed on orientation.

Usage:
    python -m scripts.dedupe_db            # migrate the configured DB
    python -m scripts.dedupe_db --dry-run  # report what would change, write nothing
"""
from __future__ import annotations

import argparse

from touchline import config
from touchline.models import dedupe_matches
from touchline.storage.db import Database


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dedupe_db")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the duplicate count without modifying the DB")
    args = ap.parse_args(argv)

    db = Database(config.DB_PATH)
    db.init_schema()
    before = db.all_matches()
    deduped = dedupe_matches(before)
    removed = len(before) - len(deduped)
    print(f"DB: {config.DB_PATH}")
    print(f"  rows before:   {len(before)}")
    print(f"  rows after:    {len(deduped)}")
    print(f"  duplicates removed: {removed}")

    if args.dry_run:
        print("  (dry run — no changes written)")
        return 0

    db.replace_all_matches(deduped)
    after = db.all_matches()
    print(f"  rewritten:     {len(after)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
