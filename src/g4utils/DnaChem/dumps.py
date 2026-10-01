from __future__ import annotations

import re
from pathlib import Path

MANIFEST_NAME = "Manifest.json"

_NUMBER = re.compile(r"[+-]?\d+(\.\d*)?([eE][+-]?\d+)?")


def find_dumps(path: str | Path) -> list[Path]:
    """Return the Dump folders at ``path``.

    ``path`` itself if it holds a ``Manifest.json``; otherwise its direct
    subdirectories that do, sorted by name.
    """
    root = Path(path)
    if (root / MANIFEST_NAME).is_file():
        return [root]
    dumps = (
        sorted(
            p for p in root.iterdir() if p.is_dir() and (p / MANIFEST_NAME).is_file()
        )
        if root.is_dir()
        else []
    )
    if not dumps:
        raise FileNotFoundError(f"No Dump (Manifest.json) found in {root}")
    return dumps


def parse_dump_name(
    name: str, pattern: str | re.Pattern | None
) -> dict[str, float | str]:
    """Extract named-group values from a Dump folder name.

    ``p`` in a value stands for the decimal point; values that parse as
    numbers are returned as ``float``, others are kept as strings.
    """
    if pattern is None:
        return {}
    match = re.search(pattern, name)
    if match is None:
        pat = pattern.pattern if isinstance(pattern, re.Pattern) else pattern
        raise ValueError(f"Dump folder {name!r} does not match pattern {pat!r}")
    out: dict[str, float | str] = {}
    for key, value in match.groupdict().items():
        if value is not None and _NUMBER.fullmatch(value.replace("p", ".")):
            out[key] = float(value.replace("p", "."))
        else:
            out[key] = value
    return out
