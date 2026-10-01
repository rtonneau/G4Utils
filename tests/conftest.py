from __future__ import annotations

import json
from pathlib import Path

import pytest

_SPECIES_HEADER = [
    "#class tools::wcsv::ntuple",
    "#title species",
    "#separator 44",
    "#vector_separator 59",
    "#column int speciesID",
    "#column int number",
    "#column int nEvent",
    "#column string speciesName",
    "#column double time",
    "#column double sumG",
    "#column double sumG2",
]

_REACTIONS_HEADER = [
    "#class tools::wcsv::ntuple",
    "#title reactions",
    "#separator 44",
    "#vector_separator 59",
    "#column int reactionId",
    "#column double time",
    "#column int count",
]


def _write(path: Path, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")


@pytest.fixture
def make_dump():
    def _make_dump(
        parent: Path,
        name: str,
        *,
        molarity_M: float = 0.0,
        energy_deposit_eV: float = 1e7,
        runs: list[dict] | None = None,
        with_manifest: bool = True,
    ) -> Path:
        d = Path(parent) / name
        d.mkdir(parents=True, exist_ok=True)
        if with_manifest:
            if runs is None:
                runs = [
                    {
                        "run": 0,
                        "events": 100,
                        "particle": "e-",
                        "beamEnergy_keV": 100,
                        "energyDeposit_eV": energy_deposit_eV,
                        "seed": 12345,
                    }
                ]
            manifest = {
                "schemaVersion": 1,
                "chemistry": "BoscoloChem",
                "pH": 7,
                "scavengers": [{"species": "O2", "molarity_M": molarity_M}],
                "totalEvents": 100,
                "totalEnergyDeposit_eV": energy_deposit_eV,
                "runs": runs,
            }
            with open(d / "Manifest.json", "w", encoding="utf-8", newline="\r\n") as f:
                json.dump(manifest, f, indent=2)
        sp = [
            (0, 400000, "H3O^1"),
            (1, 480000, "°OH^0"),
            (3, 400000, "e_aq^-1"),
            (7, 3, "HO_2°^0"),
        ]
        sp_late = [270000, 250000, 260000, 50]
        rows = [f"{i},{n},100,{s},0.001,0.5,0.25" for i, n, s in sp]
        rows += [
            f"{i},{n},100,{s},999.999,0.1,0.01"
            for (i, _, s), n in zip(sp, sp_late)
        ]
        _write(d / "Species_nt_species.csv", _SPECIES_HEADER + rows)
        _write(
            d / "Reactions_nt_reactions.csv",
            _REACTIONS_HEADER + ["1,0.001,646", "25,0.001,4436", "25,999.999,9000"],
        )
        _write(
            d / "ReactionsMetadata.csv",
            [
                "reactionId,reaction",
                "1,H3O^1 + OH^-1 -> (no products)",
                "25,°OH^0 + °OH^0 -> H2O2^0",
            ],
        )
        return d

    return _make_dump
