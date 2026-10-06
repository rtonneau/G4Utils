from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("g4utils")
except PackageNotFoundError:  # source tree not installed
    __version__ = "0.0.0+unknown"
