from __future__ import annotations

import hashlib
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "Anthropic/hh-rlhf"
CONFIG = "default"
SPLIT = "train"
API = "https://datasets-server.huggingface.co/rows"


def _request_payload(offset: int, length: int, retries: int = 5) -> dict:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": CONFIG, "split": SPLIT, "offset": offset, "length": length}
    )
    url = f"{API}?{query}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "optimML/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            return payload
        except Exception as exc:  # pragma: no cover - network-dependent branch
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Unable to retrieve {url}") from last_error


def canonical_jsonl(rows: list[dict[str, str]]) -> bytes:
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def fetch_pairs(count: int, cache_dir: Path, sample_seed: int, page_size: int = 100) -> tuple[list[dict[str, str]], dict]:
    if count < 3:
        raise ValueError("At least three preference pairs are required")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"hh_rlhf_{count}_seed{sample_seed}.jsonl"
    offsets_file = cache_dir / f"hh_rlhf_{count}_seed{sample_seed}_offsets.json"
    if cache_file.exists():
        raw = cache_file.read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        if len(rows) != count:
            raise RuntimeError("Cached data count does not match the requested count")
    else:
        metadata = _request_payload(0, 1)
        total = int(metadata["num_rows_total"])
        pages_needed = (count + page_size - 1) // page_size
        page_count = total // page_size
        page_indices = random.Random(sample_seed).sample(range(page_count), pages_needed + 5)
        offsets = [page * page_size for page in page_indices]
        rows: list[dict[str, str]] = []
        for offset in offsets:
            payload = _request_payload(offset, min(page_size, count - len(rows)))
            rows.extend(item["row"] for item in payload["rows"] if item["row"]["chosen"] != item["row"]["rejected"])
            if len(rows) >= count:
                break
        rows = rows[:count]
        if len(rows) != count:
            raise RuntimeError("Not enough non-identical preference pairs were retrieved")
        raw = canonical_jsonl(rows)
        cache_file.write_bytes(raw)
        offsets_file.write_text(json.dumps({"total_rows": total, "offsets": offsets}, indent=2), encoding="utf-8")

    digest = hashlib.sha256(raw).hexdigest()
    if not all(isinstance(r.get("chosen"), str) and isinstance(r.get("rejected"), str) for r in rows):
        raise RuntimeError("Dataset schema validation failed")
    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "rows": len(rows),
        "sampling": "deterministic random non-overlapping API pages",
        "sample_seed": sample_seed,
        "source_api": API,
        "sha256_canonical_jsonl": digest,
        "cache_file": cache_file.as_posix(),
    }
    return rows, manifest
