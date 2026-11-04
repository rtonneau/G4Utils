from __future__ import annotations

import pytest
from g4utils.DnaChem import Dump


@pytest.mark.parametrize('label', ['run', 'count', 'totalEvents', 'aggregate_energy_G'])
def test_folder_labels_cannot_replace_reader_data(tmp_path, make_dump, label):
    path = make_dump(tmp_path, 'run_7', manifest_extra={'totalEvents': 1})
    dump = Dump(path, rf'run_(?P<{label}>\d+)')
    if label == 'run':
        (path / 'TrackLengths.csv').write_text(
            'run,event,primaryLength_nm,primaryEkin0_keV,primaryEkinEnd_keV,primaryEnd,'
            'secondaryFirstGen_nm,nSecondaryFirstGen,secondaryAll_nm,nSecondaryAll\n'
            '0,0,1,10,0,stopped,2,1,3,1\n', encoding='utf-8')
        call = dump.track_lengths
    elif label in ('count', 'totalEvents'):
        (path / 'PhysicsInteractions.csv').write_text('label,count\nx,4\n', encoding='utf-8')
        call = dump.physical_interactions
    else:
        call = dump.particle_g_estimates
    with pytest.raises(ValueError, match='label.*conflict|label.*collision'):
        call()

@pytest.mark.parametrize('label', ['run', 'event'])
def test_prechemical_labels_preserve_event_identity(tmp_path, make_dump, label):
    path = make_dump(tmp_path, 'run_7')
    values = ('1', 'H2O', '0:4', '10', '1', '2', '3')
    widths = (11, 10, 14, 13, 22, 22, 22)
    (path / 'PreChemical_run0_event0.txt').write_text(
        ''.join(value.ljust(width) for value, width in zip(values, widths)) + '\n',
        encoding='utf-8')
    with pytest.raises(ValueError, match='labels conflict'):
        Dump(path, rf'run_(?P<{label}>\d+)').prechemical_records()


@pytest.mark.parametrize('label', ['count', 'handOverTime_ns'])
def test_meso_labels_preserve_counts_and_stage_metadata(tmp_path, make_dump, label):
    path = make_dump(tmp_path, 'run_7', manifest_extra={'mesoEnabled': True})
    (path / 'SpeciesMeso.csv').write_text('time_ns,species,count\n5,H2O2^0,97\n', encoding='utf-8')
    with pytest.raises(ValueError, match='labels conflict'):
        Dump(path, rf'run_(?P<{label}>\d+)').meso_species()
