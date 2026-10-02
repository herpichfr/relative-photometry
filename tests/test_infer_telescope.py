from pathlib import Path

from relphot.db.load_night import _infer_telescope


def test_telescope_from_the_staging_and_the_permanent_layout():
    assert _infer_telescope(Path("/ssdsto1/data/T80S_reduced/20251104/relphot")) == "T80S"
    assert _infer_telescope(Path("/mnt/sto01/T80S/reduced/20251104/relphot")) == "T80S"
    assert _infer_telescope(Path("/mnt/sto01/ROBO43/reduced/20250911/relphot")) == "ROBO43"


def test_no_telescope_component():
    assert _infer_telescope(Path("/reduced/20251104/relphot")) is None
    assert _infer_telescope(Path("/data/20251104/relphot")) is None
