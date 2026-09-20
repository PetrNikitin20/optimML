from __future__ import annotations

import hashlib
import json
import random
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

DATASET = "allenai/reward-bench"
CONFIG = "default"
SPLIT = "raw"
API = "https://datasets-server.huggingface.co/rows"
PARQUET_URL = "https://huggingface.co/datasets/allenai/reward-bench/resolve/refs%2Fconvert%2Fparquet/default/raw/0000.parquet"

CATEGORY_BY_SUBSET = {
    "alpacaeval-easy": "Chat",
    "alpacaeval-length": "Chat",
    "alpacaeval-hard": "Chat",
    "mt-bench-easy": "Chat",
    "mt-bench-med": "Chat",
    "mt-bench-hard": "Chat Hard",
    "llmbar-natural": "Chat Hard",
    "llmbar-adver-neighbor": "Chat Hard",
    "llmbar-adver-GPTInst": "Chat Hard",
    "llmbar-adver-GPTOut": "Chat Hard",
    "llmbar-adver-manual": "Chat Hard",
    "refusals-dangerous": "Safety",
    "refusals-offensive": "Safety",
    "xstest-should-refuse": "Safety",
    "xstest-should-respond": "Safety",
    "donotanswer": "Safety",
    "math-prm": "Reasoning",
    "hep-cpp": "Reasoning",
    "hep-go": "Reasoning",
    "hep-java": "Reasoning",
    "hep-js": "Reasoning",
    "hep-python": "Reasoning",
    "hep-rust": "Reasoning",
}


def _request(offset: int, length: int, retries: int = 5) -> dict:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": CONFIG, "split": SPLIT, "offset": offset, "length": length}
    )
    url = f"{API}?{query}"
    error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "optimML/2.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except Exception as exc:  # pragma: no cover
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Unable to retrieve {url}") from error


def fetch_all(cache_dir: Path, page_size: int = 100) -> tuple[list[dict], dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "rewardbench_raw_5123.jsonl"
    if cache.exists():
        raw = cache.read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    else:
        import pyarrow.parquet as parquet

        parquet_path = cache_dir / "rewardbench_raw_5123.parquet"
        if not parquet_path.exists():
            request = urllib.request.Request(PARQUET_URL, headers={"User-Agent": "optimML/2.0"})
            with urllib.request.urlopen(request, timeout=180) as response:
                parquet_path.write_bytes(response.read())
        rows = parquet.read_table(parquet_path).to_pylist()
        raw = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n").encode("utf-8")
        cache.write_bytes(raw)
    subsets = sorted({row["subset"] for row in rows})
    unknown = [subset for subset in subsets if subset not in CATEGORY_BY_SUBSET]
    if unknown:
        raise RuntimeError(f"Unmapped RewardBench subsets: {unknown}")
    return rows, {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "rows": len(rows),
        "subsets": subsets,
        "sha256_canonical_jsonl": hashlib.sha256(raw).hexdigest(),
        "source_api": API,
    }


def stratified_sample(rows: list[dict], per_subset: int, seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["subset"]].append(row)
    rng = random.Random(seed)
    sample = []
    for subset in sorted(groups):
        candidates = groups[subset]
        if len(candidates) < per_subset:
            raise ValueError(f"Subset {subset} has only {len(candidates)} rows")
        sample.extend(rng.sample(candidates, per_subset))
    return sorted(sample, key=lambda row: (row["subset"], row["id"]))


def as_dialogue(prompt: str, response: str) -> str:
    return f"\n\nHuman: {prompt}\n\nAssistant: {response}"
