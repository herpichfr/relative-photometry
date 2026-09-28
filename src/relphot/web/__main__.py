"""``python -m relphot.web`` / the ``relphot-web`` console script.

Runs the FastAPI app (:data:`relphot.web.app.app`) with uvicorn.
"""

from __future__ import annotations

import argparse

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="relphot-web")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="bind port (default: 8050)")
    parser.add_argument(
        "--reload", action="store_true", help="autoreload on code changes (development only)"
    )
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run("relphot.web.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
