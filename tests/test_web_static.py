"""String-level checks of the static web page."""

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "relphot" / "web" / "static"


def test_only_the_template_holds_rerun_entries():
    # app.js treats every .rerun-entry in the document as a rerun form (reads its .rr-night);
    # any other element with that class breaks the RERUN panel.
    html = (STATIC / "index.html").read_text()
    outside = re.sub(r"<template\b.*?</template>", "", html, flags=re.S)
    assert not re.search(r'class="[^"]*\brerun-entry\b', outside)
