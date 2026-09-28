"""FastAPI web API and static front end for the relphot results database.

See docs/DB_PLAN.md ("Web (requirements 10-15)") for the design. The
:class:`fastapi.FastAPI` application itself lives in :mod:`relphot.web.app`
and is deliberately not imported here, so importing :mod:`relphot.web` never
requires the ``web`` extra (fastapi/uvicorn) to be installed.
"""

from __future__ import annotations

__all__: list[str] = []
