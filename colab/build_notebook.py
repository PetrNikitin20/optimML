from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "factorial_preference_colab.py"
OUTPUT = ROOT / "factorial_preference_study.ipynb"


def parse_cells(text: str) -> list[dict]:
    cells: list[dict] = []
    marker = "# %%"
    current_kind = "code"
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        lines = current
        if current_kind == "markdown":
            cleaned = []
            for line in lines:
                if line.startswith("# "):
                    cleaned.append(line[2:])
                elif line.rstrip() == "#":
                    cleaned.append("")
                elif line.startswith("#"):
                    cleaned.append(line[1:].lstrip())
                else:
                    cleaned.append(line)
            source = "\n".join(cleaned).strip() + "\n"
        else:
            source = "\n".join(lines).strip() + "\n"
        cells.append(
            {
                "cell_type": current_kind,
                "metadata": {},
                "source": source.splitlines(keepends=True),
                **({"execution_count": None, "outputs": []} if current_kind == "code" else {}),
            }
        )
        current = []

    for line in text.splitlines():
        if line.startswith(marker):
            flush()
            current_kind = "markdown" if "[markdown]" in line else "code"
        else:
            current.append(line)
    flush()
    return cells


notebook = {
    "cells": parse_cells(SOURCE.read_text(encoding="utf-8")),
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUTPUT)
