from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from g4utils.DnaChem.dump import Dump


def _one_row(path, *, number=6, n_event=2, sum_g=14 / 3, sum_g2=148 / 9):
    file = path / "Species_nt_species.csv"
    header = file.read_text(encoding="utf-8").splitlines()[:11]
    file.write_text(
        "\n".join([*header, f"3,{number},{n_event},e_aq^-1,0.001,{sum_g},{sum_g2}"]) + "\n",
        encoding="utf-8",
    )


def _energy(path, value):
    file = path / "Manifest.json"
    manifest = json.loads(file.read_text(encoding="utf-8"))
    manifest["totalEnergyDeposit_eV"] = value
    file.write_text(json.dumps(manifest), encoding="utf-8")


def test_unequal_deposits_and_provenance(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a", energy_deposit_eV=400)
    _one_row(path)
    dump = Dump(path)
    cached = dump.species()
    original = cached.copy(deep=True)

    result = dump.particle_g_estimates()

    assert result.loc[0, "aggregate_energy_G"] == pytest.approx(1.5)
    assert result.loc[0, "mean_event_G"] == pytest.approx(7 / 3)
    assert pd.isna(result.loc[0, "mean_event_G_sem"])
    assert str(result["mean_event_G_sem"].dtype) == "Float64"
    for col in ("speciesID", "speciesName", "species", "time_s", "number", "nEvent",
                "sumG", "sumG2", "dump", "dump_key", "manifest_path", "prefix"):
        assert result.loc[0, col] == original.loc[0, col]
    assert result.loc[0, "G"] == pytest.approx(1.5)
    pd.testing.assert_frame_equal(cached, original)
    assert "aggregate_energy_G" not in cached


def test_equal_deposits_and_opted_in_sem(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a", energy_deposit_eV=200)
    _one_row(path, sum_g=6, sum_g2=20)
    result = Dump(path).particle_g_estimates(assume_valid_event_g=True)
    assert result.loc[0, "aggregate_energy_G"] == pytest.approx(3)
    assert result.loc[0, "mean_event_G"] == pytest.approx(3)
    assert result.loc[0, "mean_event_G_sem"] == pytest.approx(1)


def test_zero_event_value_is_included_when_asserted(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a", energy_deposit_eV=200)
    _one_row(path, number=2, sum_g=2, sum_g2=4)
    result = Dump(path).particle_g_estimates(assume_valid_event_g=True)
    assert result.loc[0, "mean_event_G"] == pytest.approx(1)
    assert result.loc[0, "mean_event_G_sem"] == pytest.approx(1)


@pytest.mark.parametrize("energy", [0, -1, math.nan, math.inf, None])
def test_invalid_total_energy_is_contextual(tmp_path, make_dump, energy):
    path = make_dump(tmp_path, "run_a")
    _energy(path, energy)
    with pytest.raises(ValueError, match=r"Manifest\.json.*totalEnergyDeposit_eV"):
        Dump(path).particle_g_estimates()


def test_zero_events_rejected_and_one_event_has_null_sem(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a", energy_deposit_eV=100)
    _one_row(path, n_event=0)
    with pytest.raises(ValueError, match=r"Species_nt_species\.csv.*nEvent"):
        Dump(path).particle_g_estimates()
    _one_row(path, number=4, n_event=1, sum_g=4, sum_g2=16)
    result = Dump(path).particle_g_estimates(assume_valid_event_g=True)
    assert result.loc[0, "mean_event_G"] == pytest.approx(4)
    assert pd.isna(result.loc[0, "mean_event_G_sem"])


@pytest.mark.parametrize(
    "sum_g,sum_g2,bad_field",
    [
        (math.nan, 4, "sumG"), (math.inf, 4, "sumG"),
        (2, math.nan, "sumG2"), (2, math.inf, "sumG2"),
        (2, -1, "sumG2"), (4, 1, "sumG2"),
    ],
)
def test_invalid_moments_are_contextual(tmp_path, make_dump, sum_g, sum_g2, bad_field):
    path = make_dump(tmp_path, "run_a")
    _one_row(path, sum_g=sum_g, sum_g2=sum_g2)
    with pytest.raises(ValueError, match=rf"Species_nt_species\.csv.*{bad_field}"):
        Dump(path).particle_g_estimates()


def test_rounding_tolerance_clamps_only_small_negative_variance(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a")
    _one_row(path, sum_g=8.14, sum_g2=8.14**2 / 2 - 1e-5)
    result = Dump(path).particle_g_estimates(assume_valid_event_g=True)
    assert result.loc[0, "mean_event_G_sem"] == pytest.approx(0)
    _one_row(path, sum_g=8.14, sum_g2=8.14**2 / 2 - 0.01)
    with pytest.raises(ValueError, match="inconsistent"):
        Dump(path).particle_g_estimates()


def test_one_event_requires_matching_moments(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a")
    _one_row(path, n_event=1, sum_g=1, sum_g2=100)
    with pytest.raises(ValueError, match="inconsistent"):
        Dump(path).particle_g_estimates()


def test_large_event_count_does_not_overflow_sem_denominator(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a")
    _one_row(path, n_event=5_000_000_000, sum_g=0, sum_g2=0)
    result = Dump(path).particle_g_estimates(assume_valid_event_g=True)
    assert result.loc[0, "mean_event_G_sem"] == 0


def test_missing_required_column_is_contextual(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a")
    file = path / "Species_nt_species.csv"
    lines = file.read_text(encoding="utf-8").splitlines()
    lines.remove("#column double sumG2")
    lines[10:] = [line.rsplit(",", 1)[0] for line in lines[10:]]
    file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Species_nt_species\.csv.*sumG2"):
        Dump(path).particle_g_estimates()

@pytest.mark.parametrize('n_event', [1.5, 2.5])
def test_fractional_event_count_is_not_truncated(tmp_path, make_dump, n_event):
    path = make_dump(tmp_path, 'run_a')
    _one_row(path, n_event=n_event, sum_g=4, sum_g2=16)
    with pytest.raises(ValueError, match=r'Species_nt_species\.csv.*nEvent'):
        Dump(path).particle_g_estimates(assume_valid_event_g=True)


def test_standardized_estimates_use_stored_statistics(tmp_path, make_standardized_dump):
    path = make_standardized_dump(tmp_path, "new", energy_deposit_eV=400)
    result = Dump(path).particle_g_estimates()
    assert result["mean_event_G"].tolist() == [0.5, 0.4, 0.3]
    assert result["mean_event_G_sem"].tolist() == [0.25, 0.25, 0.25]
    assert str(result["mean_event_G_sem"].dtype) == "Float64"
    assert result["aggregate_energy_G"].tolist() == pytest.approx([10 * 100 / 400, 8 * 100 / 400, 6 * 100 / 400])
    assert result["G"].tolist() == [0.5, 0.4, 0.3]
    assert "speciesName" in result and "time_s" in result


def test_standardized_estimates_require_energy(tmp_path, make_standardized_dump):
    path = make_standardized_dump(tmp_path, "new", energy_deposit_eV=0)
    with pytest.raises(ValueError, match="totalEnergyDeposit_eV"):
        Dump(path).particle_g_estimates()
