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


def test_keep_this_control_is_wired_to_its_endpoint_and_marks_superseded_events():
    js = (STATIC / "app.js").read_text()
    assert "/api/detection/${ev.det_id}/keep" in js
    # the control is built only for an event that competes with another of the same light curve
    assert "competing_det_ids" in js and "keepEventControl" in js
    assert "if (competing.length === 0) return null;" in js
    # superseded events are labelled, and drawn grey and dashed on the light curve
    assert "superseded by det" in js and 'dash: "dash"' in js
