from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

MANIFEST_NAME = "Manifest.json"

_NUMBER = re.compile(r"[+-]?\d+(\.\d*)?([eE][+-]?\d+)?")


@dataclass(frozen=True)
class DumpLocation:
    """One selected manifest, identified relative to its discovery root."""

    key: str
    directory: Path
    manifest_path: Path
    prefix: str


def _read_manifest_json(file: Path) -> dict:
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{file}: manifest must be a JSON object")
    if "schemaVersion" not in data:
        raise ValueError(f"{file}: missing schemaVersion")
    if type(data["schemaVersion"]) is not int or data["schemaVersion"] < 1:
        raise ValueError(f"{file}: schemaVersion must be a positive integer")
    return data


def _literal_prefix(value, context: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or any(c in value for c in ("/", "\\", ":", "\0")):
        raise ValueError(f"{context}: prefix must be a literal filename prefix")
    return value


def _relative_file(root: Path, value: str, context: str, *, allow_empty=False) -> Path:
    """Resolve portable relative paths, enforcing containment after symlinks."""
    if not isinstance(value, str) or (not value and not allow_empty) or "\0" in value:
        raise ValueError(f"{context}: expected a relative path, got {value!r}")
    portable = value.replace("\\", "/")
    if portable.startswith("/") or PureWindowsPath(portable).drive:
        raise ValueError(f"{context}: path must be relative: {value!r}")
    root = root.resolve()
    file = (root / portable).resolve()
    if not file.is_relative_to(root):
        raise ValueError(f"{context}: path escapes root {root}: {value!r}")
    return file


def discover_dumps(root: str | Path) -> list[DumpLocation]:
    """Discover dumps in index order, or by a shallow manifest scan.

    Keys are resolved root-relative manifest paths with POSIX separators.
    An ordinary root Manifest.json retains the legacy single-dump behavior.
    Otherwise the fallback scans *Manifest.json at the root and in direct
    children; it does not recursively walk unrelated output directories.
    """
    from g4utils.DnaChem.manifest import _parse_manifest_file

    root = Path(root).resolve()
    if root.is_file():
        manifest = _parse_manifest_file(root)
        return [DumpLocation(root.name, root.parent, root, manifest.prefix or "")]
    if not root.is_dir():
        raise FileNotFoundError(f"No Dump (*Manifest.json) found at {root}")

    locations: list[DumpLocation] = []
    seen: set[str] = set()

    def add(file: Path, expected_prefix: str | None = None, context: str = ""):
        file = _relative_file(root, file.relative_to(root).as_posix(), context)
        if not file.is_file():
            raise FileNotFoundError(f"{context}: missing referenced manifest {file}")
        key = file.relative_to(root).as_posix()
        if key in seen:
            raise ValueError(f"{context}: duplicate dump key {key!r}")
        manifest = _parse_manifest_file(file)
        prefix = manifest.prefix or ""
        if expected_prefix is not None and prefix != expected_prefix:
            raise ValueError(
                f"{context}: prefix {expected_prefix!r} does not match "
                f"{file} prefix {prefix!r}"
            )
        locations.append(DumpLocation(key, file.parent, file, prefix))
        seen.add(key)

    def scan(file: Path):
        # Validate containment before opening even fallback candidates.
        file = _relative_file(root, file.relative_to(root).as_posix(), str(file))
        data = _read_manifest_json(file)
        if "dumps" not in data and data.get("kind") != "resultsIndex":
            add(file, context=str(file))
            return
        entries = data.get("dumps")
        if not isinstance(entries, list):
            raise ValueError(f"{file}: dumps must be an array")
        for i, entry in enumerate(entries):
            context = f"{file}: dumps[{i}]"
            if not isinstance(entry, dict):
                raise ValueError(f"{context}: entry must be an object")
            for field in ("folder", "prefix", "manifest"):
                if not isinstance(entry.get(field), str):
                    raise ValueError(f"{context}: {field} must be a string")
            prefix = _literal_prefix(entry["prefix"], context)
            selected = _relative_file(file.parent, entry["manifest"], context)
            folder = _relative_file(file.parent, entry["folder"], context, allow_empty=True)
            if selected.parent != folder or selected.name != f"{prefix}{MANIFEST_NAME}":
                raise ValueError(f"{context}: folder/prefix do not match manifest path")
            add(selected, expected_prefix=prefix, context=context)

    plain = root / MANIFEST_NAME
    if plain.is_file():
        scan(plain)
    else:
        candidates = sorted(root.glob(f"*{MANIFEST_NAME}"))
        for child in sorted(root.iterdir()):
            if child.is_dir():
                child_plain = child / MANIFEST_NAME
                if child_plain.is_file():
                    candidates.append(child_plain)
                else:
                    candidates.extend(sorted(child.glob(f"*{MANIFEST_NAME}")))
        for file in candidates:
            if file.is_file():
                scan(file)
    if not locations:
        raise FileNotFoundError(f"No Dump (*Manifest.json) found in {root}")
    return locations


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
