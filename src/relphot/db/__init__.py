"""relphot.db: PostgreSQL results-database connection, schema, and loaders.

See :doc:`/docs/DB_PLAN` for the schema design. :mod:`relphot.db.connect`
resolves a DSN and opens a connection; :mod:`relphot.db.schema` applies the
versioned migrations packaged under :mod:`relphot.db.sql`;
:mod:`relphot.db.load_night` loads one night's relphot outputs;
:mod:`relphot.db.load_multinight` loads one multi-night tie run's tie rows
and detections; :mod:`relphot.db.refresh` recomputes derived
``relphot.object`` summaries; :mod:`relphot.db.analyze` computes
periodograms and PERIOD. Requires the ``db`` extra
(``pip install 'relphot[db]'``).
"""

from __future__ import annotations

from relphot.db.analyze import AnalyzeReport, analyze
from relphot.db.connect import connect, default_env_path, read_env_value, resolve_dsn
from relphot.db.load_multinight import MultiNightLoadReport, load_multinight
from relphot.db.load_night import LoadReport, load_night
from relphot.db.refresh import refresh_objects
from relphot.db.schema import current_version, init_schema, migration_files

__all__ = [
    "AnalyzeReport",
    "LoadReport",
    "MultiNightLoadReport",
    "analyze",
    "connect",
    "current_version",
    "default_env_path",
    "init_schema",
    "load_multinight",
    "load_night",
    "migration_files",
    "read_env_value",
    "refresh_objects",
    "resolve_dsn",
]
