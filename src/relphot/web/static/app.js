"use strict";

// ---------------------------------------------------------------------
// relphot results database -- front end (vanilla JS + Plotly).
// No build step, no external network requests: Plotly is served locally
// from /static/plotly.min.js. See docs/DB_PLAN.md, "Web (requirements
// 10-15)" for what this page must do.
// ---------------------------------------------------------------------

const state = {
  sort: "obj_id",
  order: "asc",
  offset: 0,
  pageSize: 50,
  total: 0,
  rows: [],
  currentObject: null, // full /api/object/{id} response
  currentNightId: null,
  reprocessPending: false,
  phase: null, // last /api/object/{id}/phase response
  reprocessTimer: null, // pending poll of the reprocess queue
  rerunWatch: new Map(), // req_id -> request, for each request last seen queued or running
  rerunPollTimer: null, // next poll of the all-objects pending RERUN list
  rerunPolling: false, // a poll of that list is in flight
  rerunKey: null, // ids of the pending requests at the last poll (null before the first)
  lcView: null, // plotted light curve {kind, nightId, lc}, re-plotted when the y unit changes
};

function $(id) {
  return document.getElementById(id);
}

function qs(sel, root) {
  return (root || document).querySelector(sel);
}

function qsa(sel, root) {
  return Array.from((root || document).querySelectorAll(sel));
}

// ---------------------------------------------------------------------
// Sexagesimal formatting
// ---------------------------------------------------------------------

function raToHms(deg) {
  if (deg === null || deg === undefined || Number.isNaN(deg)) return "";
  let hours = deg / 15.0;
  const h = Math.floor(hours);
  let minutes = (hours - h) * 60.0;
  const m = Math.floor(minutes);
  const s = (minutes - m) * 60.0;
  return `${String(h).padStart(2, "0")}h${String(m).padStart(2, "0")}m${s.toFixed(2).padStart(5, "0")}s`;
}

function decToDms(deg) {
  if (deg === null || deg === undefined || Number.isNaN(deg)) return "";
  const sign = deg < 0 ? "-" : "+";
  const a = Math.abs(deg);
  const d = Math.floor(a);
  let minutes = (a - d) * 60.0;
  const m = Math.floor(minutes);
  const s = (minutes - m) * 60.0;
  return `${sign}${String(d).padStart(2, "0")}d${String(m).padStart(2, "0")}m${s.toFixed(2).padStart(5, "0")}s`;
}

// ---------------------------------------------------------------------
// Search form -> query params
// ---------------------------------------------------------------------

function collectFilters() {
  const params = new URLSearchParams();
  for (const cb of qsa('input[name="class"]:checked')) {
    params.append("class", cb.value);
  }
  for (const cb of qsa('input[name="detection_kind"]:checked')) {
    params.append("detection_kind", cb.value);
  }
  const textFields = [
    ["filter-source-db", "source_db"],
    ["filter-telescope", "telescope"],
    ["filter-name", "name"],
    ["filter-gaia-id", "gaia_id"],
  ];
  for (const [id, key] of textFields) {
    const v = $(id).value.trim();
    if (v) params.set(key, v);
  }
  const selectFields = [
    ["filter-is-exop", "is_exop"],
    ["filter-is-var", "is_var"],
    ["filter-needs-review", "needs_review"],
    ["filter-user-reviewed", "user_reviewed"],
    ["filter-rerun-pending", "rerun_pending"],
    ["filter-known", "known"],
    ["filter-status", "status"],
    ["filter-has-periodogram", "has_periodogram"],
    ["filter-detection-scope", "detection_scope"],
  ];
  for (const [id, key] of selectFields) {
    const v = $(id).value;
    if (v) params.set(key, v);
  }
  const numberFields = [
    ["filter-ra", "ra"], ["filter-dec", "dec"], ["filter-radius", "radius"],
    ["filter-mag-min", "mag_min"], ["filter-mag-max", "mag_max"],
    ["filter-period-min", "period_min"], ["filter-period-max", "period_max"],
    ["filter-snr-min", "snr_min"], ["filter-depth-min", "depth_min"],
    ["filter-tier-max", "tier_max"], ["filter-n-nights-min", "n_nights_min"],
    ["filter-min-p-match", "min_p_match"],
  ];
  for (const [id, key] of numberFields) {
    const v = $(id).value;
    if (v !== "") params.set(key, v);
  }
  const dateFields = [["filter-night-from", "night_from"], ["filter-night-to", "night_to"]];
  for (const [id, key] of dateFields) {
    const v = $(id).value;
    if (v) params.set(key, v);
  }
  return params;
}

function resetFilters() {
  qs("#search-form").reset();
  state.sort = "obj_id";
  state.order = "asc";
  state.offset = 0;
  doSearch();
}

// ---------------------------------------------------------------------
// Results table
// ---------------------------------------------------------------------

const RESULT_COLUMNS = [
  "obj_id", "name", "ra", "dec", "class", "is_exop", "is_var", "known", "source_db",
  "known_name", "known_type", "period", "period_err", "period_source", "known_period",
  "mean_mag", "mean_mag_app", "mag_zp_source", "n_nights", "best_snr", "depth", "duration_h",
  "duration_lower_limit",
  "amplitude", "status", "first_night", "last_night", "n_review_pending", "n_nights_reviewed",
  "n_transit_events", "max_p_match", "period_delta",
  "period_delta_err", "period_verify_status", "period_verify_note",
  "n_rerun_pending", "last_rerun_status", "last_rerun_finished_at",
];

function renderResultsHeader() {
  const tr = qs("#results-table thead tr");
  tr.innerHTML = "";
  const thLoad = document.createElement("th");
  thLoad.textContent = "";
  tr.appendChild(thLoad);
  for (const col of RESULT_COLUMNS) {
    const th = document.createElement("th");
    th.textContent = col + (state.sort === col ? (state.order === "asc" ? " ▲" : " ▼") : "");
    th.dataset.sort = col;
    th.style.cursor = "pointer";
    th.addEventListener("click", () => {
      if (state.sort === col) {
        state.order = state.order === "asc" ? "desc" : "asc";
      } else {
        state.sort = col;
        state.order = "asc";
      }
      state.offset = 0;
      doSearch();
    });
    tr.appendChild(th);
  }
}

// whether objId is the object loaded in the detail panel
function isCurrentObject(objId) {
  return !!state.currentObject && state.currentObject.object.obj_id === objId;
}

// highlight the loaded object's row (renderResultsBody does it again on every re-render)
function highlightCurrentRow() {
  for (const tr of qsa("#results-table tbody tr")) {
    tr.classList.toggle("current-object", isCurrentObject(parseInt(tr.dataset.objId, 10)));
  }
}

// RERUN marker of a results row: the pending count, else the outcome of its newest request
function rerunRowBadge(row) {
  let text = "";
  let title = "";
  let cls = "";
  if (row.n_rerun_pending > 0) {
    cls = "pending";
    text = `RERUN ${row.n_rerun_pending}`;
    title = `${row.n_rerun_pending} RERUN request(s) queued or running`;
  } else if (row.last_rerun_status === "done" || row.last_rerun_status === "failed") {
    cls = row.last_rerun_status;
    text = `rerun ${row.last_rerun_status}`;
    title = `newest RERUN ${row.last_rerun_status}`
      + (row.last_rerun_finished_at ? " " + String(row.last_rerun_finished_at).replace("T", " ").slice(0, 19) : "");
  }
  if (!text) return null;
  const span = document.createElement("span");
  span.className = `rerun-badge ${cls}`;
  span.textContent = text;
  span.title = title;
  return span;
}

function renderResultsBody(rows) {
  const tbody = qs("#results-table tbody");
  tbody.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.dataset.objId = String(row.obj_id);
    if (isCurrentObject(row.obj_id)) tr.classList.add("current-object");
    const tdLoad = document.createElement("td");
    const btn = document.createElement("button");
    btn.textContent = "Load";
    btn.addEventListener("click", () => loadObject(row.obj_id));
    tdLoad.appendChild(btn);
    const rerun = rerunRowBadge(row);
    if (rerun) tdLoad.appendChild(rerun);
    tr.appendChild(tdLoad);
    for (const col of RESULT_COLUMNS) {
      const td = document.createElement("td");
      // a duration is a lower limit ("≥ x.xx h") for an incomplete transit
      const v = col === "duration_h" ? row.duration_display : row[col];
      td.textContent = v === null || v === undefined ? "" : String(v);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function updatePageInfo() {
  const page = Math.floor(state.offset / state.pageSize) + 1;
  const pages = Math.max(1, Math.ceil(state.total / state.pageSize));
  $("page-info").textContent = `page ${page} / ${pages}`;
  $("results-info").textContent = `${state.total} objects`;
  $("btn-prev-page").disabled = state.offset <= 0;
  $("btn-next-page").disabled = state.offset + state.pageSize >= state.total;
}

async function doSearch() {
  const params = collectFilters();
  params.set("sort", state.sort);
  params.set("order", state.order);
  params.set("limit", String(state.pageSize));
  params.set("offset", String(state.offset));
  const resp = await fetch("/api/search?" + params.toString());
  if (!resp.ok) {
    $("results-info").textContent = `search failed: ${resp.status}`;
    return;
  }
  const data = await resp.json();
  state.total = data.total;
  state.rows = data.rows;
  renderResultsHeader();
  renderResultsBody(data.rows);
  updatePageInfo();
}

function exportCsv() {
  const params = collectFilters();
  params.set("sort", state.sort);
  params.set("order", state.order);
  params.set("limit", "100000");
  window.location.href = "/api/search.csv?" + params.toString();
}

// ---------------------------------------------------------------------
// Advanced SQL
// ---------------------------------------------------------------------

async function runSql() {
  const sql = $("sql-textarea").value.trim();
  $("sql-error").hidden = true;
  $("sql-error").textContent = "";
  if (!sql) return;
  const resp = await fetch("/api/sql", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sql }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    $("sql-error").hidden = false;
    $("sql-error").textContent = data.detail || "query failed";
    $("sql-result-container").innerHTML = "";
    return;
  }
  renderSqlResult(data);
}

function renderSqlResult(data) {
  const container = $("sql-result-container");
  container.innerHTML = "";
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.appendChild(document.createElement("th"));
  for (const col of data.columns) {
    const th = document.createElement("th");
    th.textContent = col;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  const objIdIdx = data.columns.indexOf("obj_id");
  for (const row of data.rows) {
    const tr = document.createElement("tr");
    const tdLoad = document.createElement("td");
    if (objIdIdx >= 0) {
      const btn = document.createElement("button");
      btn.textContent = "Load";
      btn.addEventListener("click", () => loadObject(row[objIdIdx]));
      tdLoad.appendChild(btn);
    }
    tr.appendChild(tdLoad);
    for (const v of row) {
      const td = document.createElement("td");
      td.textContent = v === null || v === undefined ? "" : String(v);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  container.appendChild(table);
  if (data.truncated) {
    const note = document.createElement("div");
    note.className = "note";
    note.textContent = `truncated to ${data.rows.length} rows`;
    container.appendChild(note);
  }
}

// ---------------------------------------------------------------------
// Object detail
// ---------------------------------------------------------------------

const META_FIELDS = [
  ["name", "Name"], ["ra_sexagesimal", "RA"], ["dec_sexagesimal", "Dec"],
  ["ra", "RA (deg)"], ["dec", "Dec (deg)"], ["class", "CLASS"],
  ["is_exop", "exoplanet host"], ["exop_source", "exoplanet flag source"],
  ["is_var", "variable"], ["var_source", "variable flag source"],
  ["period", "PERIOD (d)"], ["period_err", "PERIOD error (d)"],
  ["period_n_nights", "PERIOD from n nights"],
  ["duration_display", "transit duration"],
  ["period_source", "period source"], ["known", "KNOWN"], ["source_db", "SOURCE_DB"],
  ["known_name", "known name"], ["known_type", "known type"], ["known_period", "known period (d)"],
  ["status", "status"], ["gaia_id", "Gaia ID"], ["mean_mag_display", "mean mag (apparent)"],
  ["mean_mag", "mean mag (instrumental)"],
  ["n_nights", "n nights"], ["n_review_pending", "nights awaiting review"],
  ["n_nights_reviewed", "nights reviewed"], ["notes", "notes"],
];

// a zero point as written in a label: 27.85, 20 (no trailing zeros)
function zpNum(zp) {
  return String(Number(Number(zp).toFixed(2)));
}

// "16.32 (Gaia ZP)", "≈ 17.19 (ZP 27.85, measured)" or "≈ 17.10 (ZP=20 assumed)": the apparent
// mean magnitude (the instrumental one plus the night zero points) and which zero point it rests on
function appMagText(obj) {
  if (obj.mean_mag_app === null || obj.mean_mag_app === undefined) return "";
  const m = Number(obj.mean_mag_app).toFixed(2);
  const hasZp = obj.mag_zp !== null && obj.mag_zp !== undefined;
  if (obj.mag_zp_source === "gaia") return `${m} (Gaia ZP)`;
  if (obj.mag_zp_source === "mixed") return `≈ ${m} (mixed ZP)`;
  if (obj.mag_zp_source === "measured") {
    return `≈ ${m} (${hasZp ? "ZP " + zpNum(obj.mag_zp) + ", " : ""}measured)`;
  }
  return `≈ ${m} (ZP=${hasZp ? zpNum(obj.mag_zp) : "20"} assumed)`;
}

// the zero point of a light-curve axis
function zpText(source, zp) {
  const known = zp !== null && zp !== undefined;
  if (source === "gaia") return "Gaia ZP" + (known ? " " + Number(zp).toFixed(2) : "");
  if (source === "mixed") return "mixed ZP";
  if (source === "measured") return (known ? `ZP=${zpNum(zp)} ` : "") + "measured (Gaia-matched)";
  return `ZP=${known ? zpNum(zp) : "20"} assumed`;
}

function renderMeta(obj) {
  const table = $("detail-meta");
  table.innerHTML = "";
  const display = Object.assign({}, obj, {
    mean_mag_display: appMagText(obj),
    ra_sexagesimal: raToHms(obj.ra),
    dec_sexagesimal: decToDms(obj.dec),
  });
  for (const [key, label] of META_FIELDS) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = label;
    const td = document.createElement("td");
    const v = display[key];
    td.textContent = v === null || v === undefined ? "" : String(v);
    tr.appendChild(th);
    tr.appendChild(td);
    table.appendChild(tr);
  }
}

function renderCatalogMatches(rows) {
  const tbody = qs("#detail-catalog-matches tbody");
  tbody.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    for (const key of ["catalog", "name", "type", "period", "sep_arcsec"]) {
      const td = document.createElement("td");
      td.textContent = r[key] === null || r[key] === undefined ? "" : String(r[key]);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function renderDetections(rows) {
  const tbody = qs("#detail-detections tbody");
  tbody.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    const source = r.night_id !== null && r.night_id !== undefined
      ? `${r.night_label || ""} (${r.telescope || ""})`
      : (r.mn_run_stem || "");
    const values = [r.kind, source, r.snr, r.depth, r.duration_display, r.tier, r.period, r.fap];
    for (const v of values) {
      const td = document.createElement("td");
      td.textContent = v === null || v === undefined ? "" : String(v);
      tr.appendChild(td);
    }
    const tdStatus = document.createElement("td");
    tdStatus.appendChild(detectionStatusSelect(r));
    tr.appendChild(tdStatus);
    tbody.appendChild(tr);
  }
}

// A REJECTED event is no evidence; a CONFIRMED one is evidence that no longer awaits review.
// The select is the person's verdict; an event the cross-candidate check rejected automatically
// (too many similar events on its night) is shown as "REJECTED (auto)" with the reason, and only
// a person's CONFIRMED overrides that.
function detectionStatusSelect(det) {
  const wrap = document.createElement("div");
  const sel = document.createElement("select");
  for (const s of ["UNCONFIRMED", "CONFIRMED", "REJECTED"]) {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s;
    sel.appendChild(opt);
  }
  sel.value = det.status || "UNCONFIRMED";
  sel.addEventListener("change", () => patchDetection(det.det_id, { status: sel.value }, det));
  wrap.appendChild(sel);
  if (det.auto_status === "REJECTED") {
    const eff = document.createElement("div");
    eff.className = "auto-reject";
    eff.textContent = det.effective_status === "REJECTED (auto)"
      ? "REJECTED (auto)"
      : `auto-rejected; the verdict ${det.status} stands`;
    wrap.appendChild(eff);
    if (det.auto_reason) {
      const why = document.createElement("div");
      why.className = "auto-reject-reason";
      why.textContent = det.auto_reason;
      wrap.appendChild(why);
    }
  }
  return wrap;
}

// The events of other objects on the same night that made an event look like a systematic:
// links to their objects (the nearest in time; the list is capped, n_similar is the total).
function similarEventsDetails(ev) {
  if (!ev.similar_events || ev.similar_events.length === 0) return null;
  const details = document.createElement("details");
  details.className = "similar-events";
  const total = ev.n_similar === null || ev.n_similar === undefined
    ? ev.similar_events.length : ev.n_similar;
  const summary = document.createElement("summary");
  summary.textContent = `${total} similar event${total === 1 ? "" : "s"}`
    + (ev.similar_events.length < total ? ` (nearest ${ev.similar_events.length} listed)` : "");
  const expected = ev.n_expected === null || ev.n_expected === undefined
    ? "" : `expected ${fmtValue(ev.n_expected, 3)} by chance, p = ${fmtValue(ev.p_chance, 2)}`;
  summary.title = expected;
  details.appendChild(summary);
  const list = document.createElement("div");
  for (const s of ev.similar_events) {
    const a = document.createElement("a");
    a.href = "#";
    const dt = ev.tc !== null && ev.tc !== undefined && s.tc !== null && s.tc !== undefined
      ? ` dt ${((s.tc - ev.tc) * 1440).toFixed(1)} min` : "";
    a.textContent = s.obj_name || `obj ${s.obj_id}`;
    a.title = `depth ${fmtValue(s.depth, 3)}, T14 ${s.duration_display || ""}${dt}`;
    a.addEventListener("click", (e) => {
      e.preventDefault();
      loadObject(s.obj_id);
    });
    list.appendChild(a);
    list.appendChild(document.createTextNode(
      ` (depth ${fmtValue(s.depth, 3)}, T14 ${s.duration_display || ""}${dt})`));
    list.appendChild(document.createElement("br"));
  }
  details.appendChild(list);
  return details;
}

function fmtValue(v, digits) {
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  return typeof v === "number" && digits !== undefined ? v.toPrecision(digits) : String(v);
}

function fmtPm(v, err, digits) {
  if (v === null || v === undefined) return "";
  const base = fmtValue(v, digits);
  return err === null || err === undefined ? base : `${base} ± ${fmtValue(err, 2)}`;
}

function renderTransitEvents(events) {
  const tbody = qs("#detail-transit-events tbody");
  tbody.innerHTML = "";
  for (const ev of events || []) {
    const tr = document.createElement("tr");
    const tc = ev.tc !== null && ev.tc !== undefined ? ev.tc : ev.det_tc;
    const fit = ev.tc !== null && ev.tc !== undefined
      ? `${ev.converged ? "converged" : "not converged"} (${ev.input || ""}, chi2r ${fmtValue(ev.chi2_red, 3)})`
      : "no fit";
    const cells = [
      `${ev.night_label || ""} (${ev.telescope || ""})`,
      tc === null || tc === undefined ? "" : (tc - 2460000).toFixed(5),
      fmtPm(ev.depth !== null && ev.depth !== undefined ? ev.depth : ev.det_depth, ev.depth_err, 4),
      // an incomplete transit's duration is only a minimum: "≥ x.xx h", never a measurement
      ev.duration_lower_limit
        ? ev.duration_display
        : fmtPm(ev.t14_h !== null && ev.t14_h !== undefined ? ev.t14_h : ev.det_duration_h, ev.t14_err, 3) + " h",
      fmtPm(ev.ingress_frac, ev.ingress_err, 3),
      fmtValue(ev.tier),
      ev.flags || "",
      fit,
    ];
    for (const text of cells) {
      const td = document.createElement("td");
      td.textContent = text;
      if (ev.incomplete_reason) td.title = `incomplete: ${ev.incomplete_reason}`;
      tr.appendChild(td);
    }
    const tdStatus = document.createElement("td");
    tdStatus.appendChild(detectionStatusSelect(ev));
    const similar = similarEventsDetails(ev);
    if (similar) tdStatus.appendChild(similar);
    tr.appendChild(tdStatus);
    const tdView = document.createElement("td");
    const btn = document.createElement("button");
    btn.textContent = "Plot";
    btn.addEventListener("click", () => loadNightLc(ev.night_id));
    tdView.appendChild(btn);
    tr.appendChild(tdView);
    tbody.appendChild(tr);
  }
}

async function patchDetection(detId, body, ev) {
  const resp = await fetch(`/api/detection/${detId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json();
  if (!resp.ok) {
    $("edit-status-msg").textContent = data.detail || "detection update failed";
    return;
  }
  if (ev) {
    ev.status = data.status;
    ev.notes = data.notes;
  }
  // the event changed the night's automatic evidence: pull the re-derived flags and reviews
  await refreshObject();
  $("edit-status-msg").textContent = `event ${detId}: ${data.status}`;
}

// Re-fetch the current object and redraw everything a verdict can change (flags, counters,
// the per-night review table and both event tables).
async function refreshObject() {
  if (!state.currentObject) return;
  const objId = state.currentObject.object.obj_id;
  const resp = await fetch(`/api/object/${objId}`);
  if (!resp.ok || !state.currentObject || state.currentObject.object.obj_id !== objId) return;
  const data = await resp.json();
  state.currentObject = data;
  renderMeta(data.object);
  renderDetections(data.detections);
  renderTransitEvents(data.transit_events);
  renderNightReviews(data.nights, data.object);
  prefillEditBox(data.object);
}

function renderTransitMatches(matches) {
  const tbody = qs("#detail-transit-matches tbody");
  tbody.innerHTML = "";
  for (const m of matches || []) {
    const tr = document.createElement("tr");
    const periods = (m.commensurate_periods || []).slice(0, 5).map((p) => p.toFixed(4));
    const more = (m.commensurate_periods || []).length > 5 ? " ..." : "";
    const cells = [
      `${m.night_a} / ${m.night_b}${m.same_telescope ? "" : " (different telescopes)"}`,
      m.t14_a_display || "",
      m.t14_b_display || "",
      fmtValue(m.dt_days, 6),
      fmtValue(m.depth_z, 3),
      fmtValue(m.t14_z, 3),
      fmtValue(m.ingress_z, 3),
      m.p_match === null || m.p_match === undefined ? "" : m.p_match.toPrecision(3),
      periods.join(", ") + more,
    ];
    for (const text of cells) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function renderPeriodEstimates(rows) {
  const tbody = qs("#detail-period-estimates tbody");
  tbody.innerHTML = "";
  for (const r of rows || []) {
    const tr = document.createElement("tr");
    const lowCoverage = r.phase_coverage !== null && r.phase_coverage !== undefined
      && r.phase_coverage < 0.5;
    const cells = [
      r.computed_at ? String(r.computed_at).replace("T", " ").slice(0, 19) : "",
      r.method || "",
      fmtValue(r.n_nights),
      r.last_night || "",
      fmtValue(r.baseline_days, 4),
      fmtValue(r.guess, 7),
      fmtPm(r.period, r.period_err, 7),
      fmtValue(r.harmonic),
      fmtPm(r.lit_period, r.lit_period_err, 7),
      fmtPm(r.delta, r.delta_err, 3),
      fmtValue(r.delta_z, 3),
      r.phase_coverage === null || r.phase_coverage === undefined
        ? "" : `${(100 * r.phase_coverage).toFixed(0)}%${lowCoverage ? " (low)" : ""}`,
      fmtValue(r.n_cycles, 3),
      r.verify_status || "",
    ];
    cells.forEach((text, i) => {
      const td = document.createElement("td");
      td.textContent = text;
      if (r.verify_note) td.title = r.verify_note;
      if (lowCoverage && i === 11) td.className = "warn";
      tr.appendChild(td);
    });
    const tdNote = document.createElement("td");
    tdNote.textContent = r.verify_note || "";
    tr.appendChild(tdNote);
    const tdAdopt = document.createElement("td");
    if (r.method === "LS-guided" && r.period !== null && r.period !== undefined
        && r.verify_status !== "long_period_needs_tie") {
      tdAdopt.appendChild(adoptButton(r.est_id));
    }
    tr.appendChild(tdAdopt);
    tbody.appendChild(tr);
  }
  const withDelta = (rows || []).filter((r) => r.delta !== null && r.delta !== undefined);
  if (withDelta.length === 0) {
    Plotly.purge("plot-period-verification");
    return;
  }
  const trace = {
    x: withDelta.map((r) => r.n_nights),
    y: withDelta.map((r) => r.delta),
    error_y: {
      type: "data", visible: true,
      array: withDelta.map((r) => (r.delta_err === null || r.delta_err === undefined ? 0 : r.delta_err)),
    },
    text: withDelta.map((r) => `last night ${r.last_night}, harmonic ${r.harmonic}, ${r.verify_status || ""}`),
    type: "scatter", mode: "markers", marker: { size: 8 },
  };
  Plotly.newPlot("plot-period-verification", [trace], {
    xaxis: { title: "number of nights", dtick: 1 },
    yaxis: { title: "P_obs / harmonic - P_lit (d)", zeroline: true },
    margin: { t: 20 },
  }, { responsive: true });
}

function adoptButton(estId) {
  const btn = document.createElement("button");
  btn.textContent = "Adopt as period";
  btn.title = "set this as the object's manual PERIOD (with its error)";
  btn.addEventListener("click", () => adoptPeriod(estId));
  return btn;
}

async function adoptPeriod(estId) {
  const objId = state.currentObject.object.obj_id;
  const resp = await fetch(`/api/object/${objId}/adopt_period`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ est_id: estId }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    $("edit-status-msg").textContent = data.detail || "adopting the period failed";
    return;
  }
  state.currentObject.object = Object.assign({}, state.currentObject.object, data);
  renderMeta(state.currentObject.object);
  prefillEditBox(state.currentObject.object);
  $("edit-status-msg").textContent = `PERIOD set to ${data.period} d (manual)`;
  loadPhase(null);
}

function renderNightButtons(obj, nights) {
  const container = $("night-buttons");
  container.innerHTML = "";
  for (const night of nights) {
    const btn = document.createElement("button");
    btn.textContent = `${night.label} (${night.telescope || ""})`;
    btn.addEventListener("click", () => loadNightLc(night.night_id));
    container.appendChild(btn);
  }
  if (obj.n_nights >= 2) {
    const btn = document.createElement("button");
    btn.textContent = "All nights";
    btn.addEventListener("click", loadCombinedLc);
    container.appendChild(btn);
  }
}

// ---------------------------------------------------------------------
// Night reviews: a person's verdict per night (EXOP and VAR each blank = automatic)
// ---------------------------------------------------------------------

function yesNo(v) {
  return v ? "yes" : "no";
}

function autoEvidenceText(has, open) {
  if (!has) return "no";
  return open ? "yes (awaiting review)" : "yes";
}

function reviewSelect(value) {
  const sel = document.createElement("select");
  for (const [v, text] of [["", "(auto)"], ["CONFIRMED", "CONFIRMED"], ["REJECTED", "REJECTED"]]) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = text;
    sel.appendChild(opt);
  }
  sel.value = value || "";
  return sel;
}

function errorText(data, fallback) {
  if (data && typeof data.detail === "string") return data.detail;
  if (data && data.detail) return JSON.stringify(data.detail);
  return fallback;
}

function renderNightReviewSummary(nights, obj, message) {
  const pendingNights = nights.filter((n) => n.pending).length;
  const parts = [
    `Literature: known planet host ${yesNo(obj.lit_exop)}, known variable ${yesNo(obj.lit_var)}.`,
    `${pendingNights} of ${nights.length} night(s) awaiting review`
      + ` (${obj.n_review_pending || 0} pending item(s) in all),`
      + ` ${obj.n_nights_reviewed || 0} night(s) reviewed.`,
  ];
  if (message) parts.push(message);
  $("night-review-summary").textContent = parts.join(" ");
}

async function saveNightReview(index, exopSel, varSel, noteInput, msgSpan) {
  const nights = state.currentObject.nights;
  const night = nights[index];
  const objId = state.currentObject.object.obj_id;
  msgSpan.textContent = "";
  const resp = await fetch(`/api/object/${objId}/night/${night.night_id}/review`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      exop: exopSel.value || null,
      var: varSel.value || null,
      note: noteInput.value.trim() || null,
    }),
  });
  let data = null;
  try {
    data = await resp.json();
  } catch (err) {
    data = null;
  }
  if (!resp.ok) {
    msgSpan.textContent = errorText(data, `save failed (${resp.status})`);
    return;
  }
  const review = data.review;
  night.review_exop = review ? review.exop : null;
  night.review_var = review ? review.var : null;
  night.review_note = review ? review.note : null;
  night.review_updated_at = review ? review.updated_at : null;
  Object.assign(night, data.night);
  const obj = Object.assign({}, state.currentObject.object, data.object);
  state.currentObject.object = obj;
  renderMeta(obj);
  prefillEditBox(obj);
  renderNightReviews(nights, obj, `Saved night ${night.label}.`);
}

function renderNightReviews(nights, obj, message) {
  const tbody = qs("#detail-night-reviews tbody");
  tbody.innerHTML = "";
  nights.forEach((night, index) => {
    const tr = document.createElement("tr");
    if (night.pending) tr.classList.add("needs-review");
    const exopSel = reviewSelect(night.review_exop);
    const varSel = reviewSelect(night.review_var);
    const noteInput = document.createElement("input");
    noteInput.type = "text";
    noteInput.maxLength = 2000;
    noteInput.value = night.review_note || "";
    const saveBtn = document.createElement("button");
    saveBtn.textContent = "Save";
    const msgSpan = document.createElement("span");
    msgSpan.className = "warn";
    saveBtn.addEventListener("click", () => saveNightReview(index, exopSel, varSel, noteInput, msgSpan));

    const cells = [
      `${night.label} (${night.telescope || ""})`,
      autoEvidenceText(night.auto_exop, night.exop_open),
      autoEvidenceText(night.auto_var, night.var_open),
    ];
    for (const text of cells) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    }
    for (const el of [exopSel, varSel, noteInput]) {
      const td = document.createElement("td");
      td.appendChild(el);
      tr.appendChild(td);
    }
    const tdEffective = document.createElement("td");
    tdEffective.textContent = `EXOP ${yesNo(night.exop_effective)}, VAR ${yesNo(night.var_effective)}`;
    if (night.review_updated_at) tdEffective.title = `verdict saved ${night.review_updated_at}`;
    tr.appendChild(tdEffective);
    const tdSave = document.createElement("td");
    tdSave.appendChild(saveBtn);
    tdSave.appendChild(msgSpan);
    tr.appendChild(tdSave);
    tbody.appendChild(tr);
  });
  renderNightReviewSummary(nights, obj, message);
}

function populatePeriodogramSelectors(periodograms) {
  const scopeSel = $("periodogram-scope-select");
  const methodSel = $("periodogram-method-select");
  scopeSel.innerHTML = "";
  methodSel.innerHTML = "";
  const scopes = Array.from(new Set(periodograms.map((p) => p.scope)));
  const methods = Array.from(new Set(periodograms.map((p) => p.method)));
  for (const s of scopes) {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s;
    scopeSel.appendChild(opt);
  }
  for (const m of methods) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    methodSel.appendChild(opt);
  }
  if (scopes.length === 0) {
    $("periodogram-message").textContent = "no periodogram available for this object";
    Plotly.purge("plot-periodogram");
  } else {
    $("periodogram-message").textContent = "";
    loadPeriodogram();
  }
}

function prefillEditBox(obj) {
  $("edit-exop-check").checked = !!obj.is_exop;
  $("edit-var-check").checked = !!obj.is_var;
  $("edit-flag-source").textContent =
    `exoplanet flag: ${obj.exop_source || "auto"}; variable flag: ${obj.var_source || "auto"}`;
  $("edit-status-select").value = obj.status || "";
  $("edit-notes-textarea").value = obj.notes || "";
  $("edit-period-input").value = obj.period === null || obj.period === undefined ? "" : obj.period;
  $("edit-status-msg").textContent = "";
}

// obj_id (and name) in the "Object detail" header; the RERUN badge waits for the requests
function renderDetailHeader(obj) {
  $("detail-obj-id").textContent = `obj_id ${obj.obj_id}` + (obj.name ? ` (${obj.name})` : "");
  $("rerun-badge").hidden = true;
}

async function loadObject(objId) {
  const resp = await fetch(`/api/object/${objId}`);
  if (!resp.ok) {
    window.alert(`failed to load object ${objId}`);
    return;
  }
  const data = await resp.json();
  state.currentObject = data;
  state.currentNightId = null;

  $("detail-panel").hidden = false;
  renderDetailHeader(data.object);
  highlightCurrentRow();
  renderMeta(data.object);
  renderCatalogMatches(data.catalog_matches);
  renderDetections(data.detections);
  renderTransitEvents(data.transit_events);
  renderTransitMatches(data.transit_matches);
  renderPeriodEstimates(data.period_estimates);
  renderNightButtons(data.object, data.nights);
  renderNightReviews(data.nights, data.object);
  populatePeriodogramSelectors(data.periodograms);
  prefillEditBox(data.object);
  Plotly.purge("plot-lightcurve");
  Plotly.purge("plot-phase");
  state.lcView = null;
  state.phase = null;
  // the phase diagram is for a variable or any object that has a period (PERIOD, or a guided one)
  const hasGuided = (data.period_estimates || []).some(
    (e) => e.method === "LS-guided" && e.period !== null && e.period !== undefined
  );
  $("phase-section").hidden = !(data.object.is_var || data.object.period !== null || hasGuided);
  if (!$("phase-section").hidden) loadPhase(null);
  resetReprocessBox();
  loadReprocess();
  $("detail-panel").scrollIntoView({ behavior: "smooth" });
}

// ---------------------------------------------------------------------
// Light curve plots
// ---------------------------------------------------------------------

function bestTransitForNight(nightId) {
  if (!state.currentObject) return null;
  return state.currentObject.detections.find(
    (d) => d.kind === "transit" && d.night_id === nightId && d.tc_bjd_tdb !== null
  );
}

// Light-curve y unit: apparent magnitude (mean magnitude + zero point + delta mag, brighter
// up) or relative flux. The plotted payload is kept so a change of unit needs no request.
function lcUnit() {
  return $("lc-unit-select").value;
}

function replotLightcurve() {
  const v = state.lcView;
  if (!v) return;
  if (v.kind === "night") plotNightLc(v.nightId, v.lc);
  else plotCombinedLc(v.lc);
}

async function loadNightLc(nightId) {
  state.currentNightId = nightId;
  const objId = state.currentObject.object.obj_id;
  const resp = await fetch(`/api/object/${objId}/lc?night_id=${nightId}`);
  if (!resp.ok) {
    window.alert("failed to load light curve");
    return;
  }
  const lc = await resp.json();
  state.lcView = { kind: "night", nightId, lc };
  plotNightLc(nightId, lc);
}

function plotNightLc(nightId, lc) {
  const x = lc.bjd_tdb.map((t) => t - 2460000);
  const finiteFlux = lc.flux.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  const baseline = finiteFlux.length ? finiteFlux[Math.floor(finiteFlux.length / 2)] : 1;
  // magnitudes: the night's apparent mean magnitude + delta mag = -2.5 log10(flux / median)
  const useMag = lcUnit() === "mag" && lc.app_mag !== null && lc.app_mag !== undefined;
  const yOf = (f) => (useMag ? lc.app_mag - 2.5 * Math.log10(f / baseline) : f);
  const trace = {
    x, y: lc.flux.map(yOf), type: "scatter", mode: "markers",
    error_y: {
      type: "data", visible: true,
      array: useMag ? lc.flux.map((f, i) => 1.0857 * lc.flux_err[i] / f) : lc.flux_err,
    },
    text: lc.frame_index.map((fi, i) => `frame ${fi}<br>${lc.file_name[i] || ""}<br>airmass ${lc.airmass[i]}`
      + `<br>flux ${(lc.flux[i] / baseline).toFixed(4)} (${(-2.5 * Math.log10(lc.flux[i] / baseline)).toFixed(4)} mag)`),
    hoverinfo: "x+y+text",
    marker: { size: 5 },
    name: lc.night_label || `night ${nightId}`,
    meta: { night_id: nightId },
  };
  const shapes = [];
  const traces = [trace];
  const annotations = [];
  const nightEvents = (state.currentObject.transit_events || []).filter(
    (ev) => ev.night_id === nightId && ev.tc !== null && ev.tc !== undefined
  );
  // a full trapezoid needs the ingress fraction, which an event with an unobserved ingress or
  // egress does not have; its (lower-limit) duration is only marked at the fitted tc
  const fitted = nightEvents.filter(
    (ev) => ev.depth !== null && ev.t14_h !== null && ev.ingress_frac !== null
  );
  for (const ev of nightEvents.filter((e) => !fitted.includes(e))) {
    shapes.push({
      type: "line", x0: ev.tc - 2460000, x1: ev.tc - 2460000, y0: 0, y1: 1, yref: "paper",
      line: { color: "rgb(200,30,30)", dash: "dot" },
    });
    annotations.push({
      x: ev.tc - 2460000, y: 1, yref: "paper", showarrow: false, yanchor: "bottom",
      text: `T14 ${ev.duration_display || ""}${ev.incomplete_reason ? " (incomplete)" : ""}`,
    });
  }
  for (const ev of fitted) {
    const t14 = ev.t14_h / 24.0;
    const tau = Math.max(ev.ingress_frac * t14, 1e-3 * t14);
    const xs = [];
    const ys = [];
    for (let k = 0; k <= 240; k++) {
      const t = ev.tc + (k / 240 - 0.5) * 3.0 * t14;
      const s = Math.min(1, Math.max(0, (t14 / 2 - Math.abs(t - ev.tc)) / tau));
      xs.push(t - 2460000);
      ys.push(yOf(baseline * (1 - ev.depth * s)));
    }
    traces.push({
      x: xs, y: ys, type: "scatter", mode: "lines", line: { color: "rgb(200,30,30)" },
      name: `trapezoid fit (${ev.converged ? "converged" : "not converged"})`,
      text: xs.map(() => `T14 ${ev.duration_display || ""}${ev.incomplete_reason ? " (incomplete: " + ev.incomplete_reason + ")" : ""}`),
      hoverinfo: "x+y+text",
    });
  }
  const transit = nightEvents.length ? null : bestTransitForNight(nightId);
  if (transit) {
    const tc = transit.tc_bjd_tdb - 2460000;
    const halfDur = (transit.duration_h || 0) / 24.0 / 2.0;
    const depth = transit.depth || 0;
    const yTop = Math.max(...lc.flux.filter((v) => Number.isFinite(v)));
    shapes.push({
      type: "rect", x0: tc - halfDur, x1: tc + halfDur, y0: yOf(1 - depth), y1: yOf(yTop),
      line: { color: "rgba(200,30,30,0.6)" }, fillcolor: "rgba(200,30,30,0.08)",
    });
    annotations.push({
      x: tc, y: 1, yref: "paper", showarrow: false, yanchor: "bottom",
      text: `T14 ${transit.duration_display || ""}`,
    });
  }
  Plotly.newPlot("plot-lightcurve", traces, {
    xaxis: { title: { text: "BJD_TDB - 2460000" } },
    yaxis: useMag
      ? { title: { text: `apparent mag (${zpText(lc.zp_source, lc.zp)})` }, autorange: "reversed" }
      : { title: { text: "relative flux" } },
    shapes,
    annotations,
    margin: { t: 20 },
    dragmode: lightcurveDragmode(),
    selectdirection: "h",
  }, { responsive: true });
  attachLightcurveEvents();
}

async function loadCombinedLc() {
  const objId = state.currentObject.object.obj_id;
  const resp = await fetch(`/api/object/${objId}/lc/combined`);
  if (!resp.ok) {
    window.alert("failed to load combined light curve");
    return;
  }
  const lc = await resp.json();
  state.lcView = { kind: "combined", lc };
  plotCombinedLc(lc);
}

// The combined curve: tied, `value` is the tie-calibrated magnitude (+ the anchor night's zero
// point = apparent); untied, it is flux over the night's median, given the night's apparent
// mean magnitude (`app_mag`) as its level.
function plotCombinedLc(lc) {
  const tied = lc.mode === "tied-mag";
  const useMag = lcUnit() === "mag" && (!tied || (lc.zp !== null && lc.zp !== undefined));
  const sortedValues = lc.value.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  const mref = sortedValues.length ? sortedValues[Math.floor(sortedValues.length / 2)] : 0;
  let yOf;
  let errOf;
  if (tied && useMag) {
    yOf = (i) => lc.value[i] + lc.zp;
    errOf = (i) => lc.value_err[i];
  } else if (tied) {
    yOf = (i) => Math.pow(10, -0.4 * (lc.value[i] - mref));
    errOf = (i) => yOf(i) * lc.value_err[i] / 1.0857;
  } else if (useMag) {
    yOf = (i) => (lc.app_mag[i] === null ? NaN : lc.app_mag[i] - 2.5 * Math.log10(lc.value[i]));
    errOf = (i) => 1.0857 * lc.value_err[i] / lc.value[i];
  } else {
    yOf = (i) => lc.value[i];
    errOf = (i) => lc.value_err[i];
  }
  const byNight = new Map();
  for (let i = 0; i < lc.bjd_tdb.length; i++) {
    const key = lc.night_id[i];
    if (!byNight.has(key)) byNight.set(key, { x: [], y: [], err: [], text: [], label: lc.night_label[i], night_id: key });
    const bucket = byNight.get(key);
    bucket.x.push(lc.bjd_tdb[i] - 2460000);
    bucket.y.push(yOf(i));
    bucket.err.push(errOf(i));
    bucket.text.push(lc.file_name[i] || "");
  }
  const traces = Array.from(byNight.values()).map((bucket) => ({
    x: bucket.x, y: bucket.y, type: "scatter", mode: "markers",
    error_y: { type: "data", array: bucket.err, visible: true },
    text: bucket.text, hoverinfo: "x+y+text",
    marker: { size: 5 }, name: bucket.label, meta: { night_id: bucket.night_id },
  }));
  const layout = {
    xaxis: { title: { text: "BJD_TDB - 2460000" } },
    yaxis: {
      title: {
        text: !useMag
          ? (tied ? "relative flux (from tie-calibrated magnitudes)" : "relative flux (per-night normalised)")
          : `apparent mag (${tied ? zpText(lc.zp_source, lc.zp) + ", tied" : "night means, " + zpText(lc.zp_source, null)})`,
      },
      autorange: useMag ? "reversed" : true,
    },
    margin: { t: 40 },
    dragmode: lightcurveDragmode(),
    selectdirection: "h",
    title: {
      text: lc.mode === "tied-mag"
        ? "combined (tie-calibrated magnitudes)"
        : "combined (per-night normalised flux -- nights are NOT tied)",
    },
  };
  Plotly.newPlot("plot-lightcurve", traces, layout, { responsive: true });
  attachLightcurveEvents();
}

// The rerun entry a click or drag on the light curve fills: the one you last edited.
function rerunTargetEntry() {
  if (!$("rerun-check").checked) {
    $("rp-msg").textContent = "tick RERUN first";
    return null;
  }
  // Target .rerun-entry.active or the last entry
  let target = qs(".rerun-entry.active");
  if (!target) {
    const entries = qsa(".rerun-entry");
    target = entries[entries.length - 1];
  }
  return target || null;
}

// Find and select the night option of an entry if it offers it.
function selectRerunNight(target, nightId) {
  if (!nightId) return false;
  const nightSelect = qs(".rr-night", target);
  const options = qsa("option", nightSelect);
  const opt = options.find(o => o.value === String(nightId));
  if (!opt) return false;
  nightSelect.value = nightId;
  syncRerunEntry(target);
  return true;
}

// a click on the light curve fills in the transit centre of the rerun entry you last edited
function fillRerunFromClick(ev) {
  if (!ev.points || ev.points.length === 0) return;
  const target = rerunTargetEntry();
  if (!target) return;

  qs(".rr-exop", target).checked = true;
  syncRerunEntry(target);
  const tcValue = (ev.points[0].x + 2460000).toFixed(5);
  qs(".rr-tc", target).value = tcValue;

  // Get night from meta or currentNightId
  const nightFromData = ev.points[0].data.meta && ev.points[0].data.meta.night_id;
  const nightId = nightFromData !== undefined ? nightFromData : state.currentNightId;
  selectRerunNight(target, nightId);
}

// While RERUN is ticked a drag on the light curve selects an x span (the mode bar's zoom
// button gets zooming back); otherwise the plot keeps dragging to zoom.
function lightcurveDragmode() {
  return $("rerun-check").checked ? "select" : "zoom";
}

function setLightcurveDragmode() {
  const gd = $("plot-lightcurve");
  if (gd.data) Plotly.relayout(gd, { dragmode: lightcurveDragmode() });
}

// x span [x0, x1] of a plotly_selected event: the dragged box (also over a gap with no
// points), else, for a lasso, the extent of the selected points; null if there is none.
function selectedSpan(ev) {
  let xs = ev && ev.range && ev.range.x;
  if (!xs && ev && ev.points && ev.points.length > 1) {
    const px = ev.points.map((p) => p.x);
    xs = [Math.min(...px), Math.max(...px)];
  }
  return xs && xs[1] > xs[0] ? [xs[0], xs[1]] : null;
}

// The night with the most points in [x0, x1] (each trace carries its night in meta), else
// the night being plotted.
function nightInSpan(gd, x0, x1) {
  const counts = new Map();
  for (const tr of gd.data || []) {
    const nid = tr.meta && tr.meta.night_id;
    if (nid === undefined || nid === null) continue;
    let n = 0;
    for (const x of tr.x) {
      if (x >= x0 && x <= x1) n++;
    }
    if (n > 0) counts.set(nid, (counts.get(nid) || 0) + n);
  }
  let best = null;
  for (const [nid, n] of counts) {
    if (best === null || n > counts.get(best)) best = nid;
  }
  return best !== null ? best : state.currentNightId;
}

// A drag on the light curve fills the rerun entry you last edited. EXOP (also when both are
// ticked): the transit centre (midpoint of the span) and the width of the suspected eclipse
// (span, in hours). VAR only: the period guess (span, in days: drag from one peak to the
// next). The entry's night becomes the night with data in the span.
function fillRerunFromDrag(ev) {
  const gd = $("plot-lightcurve");
  const span = selectedSpan(ev);
  if (!span) return;
  // drop the selection box and the dimming of unselected points
  Plotly.relayout(gd, { selections: [] }).then(() => Plotly.restyle(gd, { selectedpoints: [null] }));
  const target = rerunTargetEntry();
  if (!target) return;

  const [x0, x1] = span;
  const forPeriod = qs(".rr-var", target).checked && !qs(".rr-exop", target).checked;
  const title = qs(".rr-title", target).textContent;
  let text;
  let outside = false;
  if (forPeriod) {
    const period = x1 - x0;
    qs(".rr-period", target).value = period.toFixed(6);
    text = `${title}: period ${period.toFixed(6)} d`;
  } else {
    qs(".rr-exop", target).checked = true;
    syncRerunEntry(target);
    const widthH = (x1 - x0) * 24.0;
    qs(".rr-tc", target).value = (0.5 * (x0 + x1) + 2460000).toFixed(5);
    qs(".rr-width", target).value = widthH.toFixed(3);
    outside = widthH < 0.1 || widthH > 12;
    text = `${title}: transit centre ${qs(".rr-tc", target).value}, width ${widthH.toFixed(3)} h`
      + (outside ? " (outside the allowed 0.1 to 12 h)" : "");
  }
  const nightSelect = qs(".rr-night", target);
  if (selectRerunNight(target, nightInSpan(gd, x0, x1)) && nightSelect.selectedOptions.length) {
    text += `, night ${nightSelect.selectedOptions[0].textContent}`;
  }
  $("rp-msg").className = outside ? "warn" : "";
  $("rp-msg").textContent = text;
}

function attachLightcurveEvents() {
  const gd = $("plot-lightcurve");
  if (!gd.on) return;
  if (gd.removeAllListeners) {
    gd.removeAllListeners("plotly_click");
    gd.removeAllListeners("plotly_selected");
  }
  gd.on("plotly_click", fillRerunFromClick);
  gd.on("plotly_selected", fillRerunFromDrag);
}

// ---------------------------------------------------------------------
// Phase diagram
// ---------------------------------------------------------------------

// Phase (0 <= phase < 1) of time t at epoch t0 and period P.
function phaseOf(t, t0, period) {
  const x = ((t - t0) / period) % 1.0;
  return x < 0 ? x + 1.0 : x;
}

// The 2-harmonic Fourier model of the /phase payload evaluated at time t.
function modelValue(model, t) {
  const arg = (2.0 * Math.PI * (t - model.t_ref)) / model.period;
  const [a1, b1, a2, b2] = model.coef;
  return model.mean + a1 * Math.cos(arg) + b1 * Math.sin(arg)
    + a2 * Math.cos(2 * arg) + b2 * Math.sin(2 * arg);
}

// Fraction of 20 phase bins holding a point, and the baseline in periods.
function phaseCoverageOf(ts, period) {
  const t0 = Math.min(...ts);
  const bins = new Set();
  for (const t of ts) bins.add(Math.min(19, Math.floor(phaseOf(t, t0, period) * 20)));
  return { coverage: bins.size / 20, cycles: (Math.max(...ts) - t0) / period };
}

// zero point that turns the phase diagram's tied magnitudes into apparent ones (0: untied flux)
function phaseZp(ph) {
  return ph.tied && ph.zp !== null && ph.zp !== undefined ? ph.zp : 0;
}

// One scatter trace per night (points coloured by night, with error bars); with `wide`, the
// points of the first and last quarter cycle are repeated one cycle away (-0.25 to 1.25).
// untied phase diagram in magnitudes too: each night's apparent mean magnitude - 2.5 log10(flux)
function phaseUsesMag(ph) {
  return !ph.tied && lcUnit() === "mag" && (ph.app_mag || []).some((v) => v !== null);
}

function phaseTraces(ph, t0, wide) {
  const useMag = phaseUsesMag(ph);
  const byNight = new Map();
  for (let i = 0; i < ph.bjd_tdb.length; i++) {
    const key = ph.night_label[i];
    if (!byNight.has(key)) byNight.set(key, { x: [], y: [], err: [], text: [] });
    const b = byNight.get(key);
    const x = phaseOf(ph.bjd_tdb[i], t0, ph.period);
    const shifts = wide ? [x, ...(x >= 0.75 ? [x - 1] : []), ...(x < 0.25 ? [x + 1] : [])] : [x];
    for (const xs of shifts) {
      b.x.push(xs);
      b.y.push(useMag
        ? (ph.app_mag[i] === null ? NaN : ph.app_mag[i] - 2.5 * Math.log10(ph.value[i]))
        : ph.value[i] + phaseZp(ph));
      b.err.push(useMag ? 1.0857 * ph.value_err[i] / ph.value[i] : ph.value_err[i]);
      b.text.push(`${key}<br>${ph.file_name[i] || ""}`);
    }
  }
  return Array.from(byNight.entries()).map(([label, b]) => ({
    x: b.x, y: b.y, type: "scatter", mode: "markers", name: label || "",
    error_y: { type: "data", array: b.err, visible: true },
    marker: { size: 5 }, text: b.text, hoverinfo: "x+y+text",
  }));
}

function plotPhase() {
  const ph = state.phase;
  if (!ph || !ph.period || ph.bjd_tdb.length === 0) {
    Plotly.purge("plot-phase");
    return;
  }
  const wide = $("phase-range-check").checked;
  const useFit = $("phase-epoch-select").value === "fit" && ph.model;
  const t0 = useFit ? ph.model.t_zero : ph.t_first;
  const traces = phaseTraces(ph, t0, wide);
  const useMag = phaseUsesMag(ph);
  const known = (ph.app_mag || []).filter((v) => v !== null);
  const meanApp = known.length ? known.reduce((a, b) => a + b, 0) / known.length : 0;
  if (ph.model && $("phase-model-check").checked) {
    const xs = [];
    const ys = [];
    const lo = wide ? -0.25 : 0.0;
    const hi = wide ? 1.25 : 1.0;
    for (let x = lo; x <= hi + 1e-9; x += 0.005) {
      xs.push(x);
      const m = modelValue(ph.model, t0 + x * ph.period);
      ys.push(useMag ? meanApp - 2.5 * Math.log10(m) : m + phaseZp(ph));
    }
    traces.push({
      x: xs, y: ys, type: "scatter", mode: "lines", name: "Fourier model (2 harmonics)",
      line: { color: "black", width: 2 },
    });
  }
  Plotly.newPlot("plot-phase", traces, {
    xaxis: { title: { text: `phase (P = ${ph.period} d)` }, range: wide ? [-0.25, 1.25] : [0, 1] },
    yaxis: {
      title: {
        text: ph.tied
          ? (phaseZp(ph) ? `apparent mag (${zpText(ph.zp_source, ph.zp)}, tied)` : "tied magnitude")
          : (useMag ? `apparent mag (night means, ${zpText(ph.zp_source, null)}, untied)`
            : "relative flux (per-night normalised, untied)"),
      },
      autorange: ph.tied || useMag ? "reversed" : true,
    },
    title: { text: ph.label },
    margin: { t: 40 },
  }, { responsive: true });
}

function renderPhaseControls() {
  const ph = state.phase;
  const sel = $("phase-candidate-select");
  sel.innerHTML = "";
  for (const c of ph.period_candidates) {
    const opt = document.createElement("option");
    opt.value = String(c.period);
    opt.textContent = `${c.label}: ${Number(c.period).toPrecision(8)} d`;
    sel.appendChild(opt);
  }
  const custom = document.createElement("option");
  custom.value = "";
  custom.textContent = "typed value";
  sel.appendChild(custom);
  const match = ph.period_candidates.find((c) => c.period === ph.period);
  sel.value = match ? String(match.period) : "";
  $("phase-period-input").value = ph.period === null || ph.period === undefined ? "" : ph.period;

  const note = $("phase-mode-note");
  note.textContent = ph.tied
    ? "Tie-calibrated magnitudes: the nights are on one scale."
    : "UNTIED: each night is normalised to its own median, so night-to-night changes are "
      + "removed and a period longer than a night cannot be seen; ties need a multi-night run.";
  note.className = ph.tied ? "note" : "note warn";

  const cov = $("phase-coverage-note");
  if (ph.phase_coverage === null || ph.phase_coverage === undefined) {
    cov.textContent = "";
    cov.className = "note";
  } else {
    const low = ph.phase_coverage < 0.5;
    cov.textContent = `Phase coverage ${(100 * ph.phase_coverage).toFixed(0)}% of 20 bins, `
      + `baseline ${ph.n_cycles.toFixed(2)} cycles`
      + (low ? " -- WARNING: coverage below 50%, this period is poorly sampled" : "")
      + (ph.period > ph.long_period_days && !ph.tied
        ? " -- WARNING: a period longer than a night is not measurable on untied data" : "");
    cov.className = low || (ph.period > ph.long_period_days && !ph.tied) ? "note warn" : "note";
  }

  const box = $("phase-aliases");
  box.innerHTML = "";
  if (ph.aliases.length) {
    const label = document.createElement("span");
    label.className = "note";
    label.textContent = "alias / next-peak candidates:";
    box.appendChild(label);
  }
  for (const a of ph.aliases) {
    const btn = document.createElement("button");
    btn.textContent = `${Number(a.period).toPrecision(6)} d`
      + (a.power === null || a.power === undefined ? "" : ` (power ${Number(a.power).toFixed(2)})`);
    btn.addEventListener("click", () => loadPhase(a.period));
    box.appendChild(btn);
  }
}

// Fetch the phase payload (at `period`, or the object's first candidate) and draw it.
async function loadPhase(period) {
  if (!state.currentObject) return;
  const objId = state.currentObject.object.obj_id;
  const url = `/api/object/${objId}/phase` + (period ? `?period=${encodeURIComponent(period)}` : "");
  const resp = await fetch(url);
  if (!resp.ok) {
    $("phase-mode-note").textContent = "no light curve for the phase diagram";
    Plotly.purge("plot-phase");
    return;
  }
  state.phase = await resp.json();
  renderPhaseControls();
  plotPhase();
}

function phasePeriodFromInput() {
  const p = parseFloat($("phase-period-input").value);
  return Number.isFinite(p) && p > 0 ? p : null;
}

// ---------------------------------------------------------------------
// Periodogram
// ---------------------------------------------------------------------

async function loadPeriodogram() {
  const objId = state.currentObject.object.obj_id;
  const scope = $("periodogram-scope-select").value;
  const method = $("periodogram-method-select").value;
  if (!scope || !method) return;
  const resp = await fetch(`/api/object/${objId}/periodogram?scope=${encodeURIComponent(scope)}&method=${encodeURIComponent(method)}`);
  if (!resp.ok) {
    $("periodogram-message").textContent = "no periodogram for this scope/method";
    Plotly.purge("plot-periodogram");
    return;
  }
  const pg = await resp.json();
  $("periodogram-message").textContent = "";
  const freqs = [];
  for (let k = 0; k < pg.n; k++) freqs.push(pg.fmin + k * pg.df);
  const periods = freqs.map((f) => (f > 0 ? 1.0 / f : null));
  const trace = { x: periods, y: pg.power, type: "scatter", mode: "lines" };
  const shapes = [];
  if (pg.peak_period) {
    shapes.push({
      type: "line", x0: pg.peak_period, x1: pg.peak_period, y0: 0, y1: 1, yref: "paper",
      line: { color: "red", dash: "dot" },
    });
  }
  const objPeriod = state.currentObject.object.period;
  if (objPeriod) {
    shapes.push({
      type: "line", x0: objPeriod, x1: objPeriod, y0: 0, y1: 1, yref: "paper",
      line: { color: "green", dash: "dash" },
    });
  }
  Plotly.newPlot("plot-periodogram", [trace], {
    xaxis: { title: "period (days)", type: "log" },
    yaxis: { title: "power" },
    shapes,
    margin: { t: 20 },
  }, { responsive: true });
}

// ---------------------------------------------------------------------
// Manual edit
// ---------------------------------------------------------------------

async function saveEdit() {
  const objId = state.currentObject.object.obj_id;
  const body = {};
  // a flag is only sent when the person changed its checkbox; the server then writes a
  // CONFIRMED/REJECTED verdict on every night the object has data for now (never on later nights)
  const obj = state.currentObject.object;
  const exopChecked = $("edit-exop-check").checked;
  const varChecked = $("edit-var-check").checked;
  if (exopChecked !== !!obj.is_exop) body.is_exop = exopChecked;
  if (varChecked !== !!obj.is_var) body.is_var = varChecked;
  const statusVal = $("edit-status-select").value;
  if (statusVal) body.status = statusVal;
  body.notes = $("edit-notes-textarea").value;
  const periodVal = $("edit-period-input").value;
  if (periodVal !== "") body.period = parseFloat(periodVal);
  await patchObject(objId, body);
}

async function resetExopAuto() {
  await patchObject(state.currentObject.object.obj_id, { exop_source: "auto" });
}

async function resetVarAuto() {
  await patchObject(state.currentObject.object.obj_id, { var_source: "auto" });
}

async function resetPeriodAuto() {
  await patchObject(state.currentObject.object.obj_id, { period_source: "auto" });
}

async function patchObject(objId, body) {
  const resp = await fetch(`/api/object/${objId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json();
  if (!resp.ok) {
    $("edit-status-msg").textContent = data.detail || "save failed";
    return;
  }
  state.currentObject.object = Object.assign({}, state.currentObject.object, data);
  renderMeta(state.currentObject.object);
  prefillEditBox(state.currentObject.object);
  // the shorthand wrote (or cleared) per-night verdicts: redraw the review table
  await refreshObject();
  $("edit-status-msg").textContent = "saved";
}

// ---------------------------------------------------------------------
// User-guided reprocessing (RERUN on demand)
// ---------------------------------------------------------------------

function rerunNights() {
  if (!state.currentObject) return [];
  return (state.currentObject.nights || []).filter(n => n.has_lc);
}

function syncRerunEntry(entry) {
  const exopBox = qs(".rr-exop-box", entry);
  const varBox = qs(".rr-var-box", entry);
  const exopChecked = qs(".rr-exop", entry).checked;
  const varChecked = qs(".rr-var", entry).checked;
  const allNightsChecked = qs(".rr-all-nights", entry).checked;

  exopBox.hidden = !exopChecked;
  varBox.hidden = !varChecked;

  // Hide night label when VAR+All nights ticked and EXOP not
  const nightLabel = qs(".rr-night-label", entry);
  nightLabel.hidden = (varChecked && allNightsChecked && !exopChecked);

  // Disable .rr-add when no selectable night is left unused and entry has no successor
  const allEntries = qsa(".rerun-entry");
  const entryIndex = Array.from(allEntries).indexOf(entry);
  const addCheckbox = qs(".rr-add", entry);
  const nextEntry = entryIndex + 1 < allEntries.length ? allEntries[entryIndex + 1] : null;

  const nights = rerunNights();
  const usedNights = new Set();
  for (let i = 0; i < allEntries.length; i++) {
    if (i !== entryIndex) {
      const nightVal = qs(".rr-night", allEntries[i]).value;
      if (nightVal) usedNights.add(parseInt(nightVal, 10));
    }
  }
  const availableNights = nights.filter(n => !usedNights.has(n.night_id));
  const shouldDisable = availableNights.length === 0 && !nextEntry;
  addCheckbox.disabled = shouldDisable;
  if (shouldDisable) addCheckbox.title = "no unused night available";
}

function resetReprocessBox() {
  if (state.reprocessTimer) {
    clearTimeout(state.reprocessTimer);
    state.reprocessTimer = null;
  }
  $("rerun-check").checked = false;
  $("rerun-panel").hidden = true;
  $("rerun-entries").innerHTML = "";
  $("rp-note").value = "";
  $("rp-msg").textContent = "";
}

function onRerunToggle() {
  const checked = $("rerun-check").checked;
  $("rerun-panel").hidden = !checked;
  if (checked && $("rerun-entries").children.length === 0) {
    addRerunEntry();
  }
  setLightcurveDragmode();
}

function addRerunEntry(after) {
  const template = $("rerun-entry-template");
  const entry = template.content.cloneNode(true);
  const fieldset = qs("fieldset", entry);

  // Fill night select
  const nights = rerunNights();
  const usedNights = new Set();
  for (const e of qsa(".rerun-entry")) {
    const nightVal = qs(".rr-night", e).value;
    if (nightVal) usedNights.add(parseInt(nightVal, 10));
  }

  const nightSelect = qs(".rr-night", entry);
  for (const night of nights) {
    if (!usedNights.has(night.night_id)) {
      const opt = document.createElement("option");
      opt.value = night.night_id;
      opt.textContent = `${night.label} (${night.telescope})`;
      nightSelect.appendChild(opt);
    }
  }
  if (nightSelect.options.length > 0) {
    nightSelect.value = nightSelect.options[0].value;
  }

  const title = qs(".rr-title", entry);
  title.textContent = `Rerun ${qsa(".rerun-entry").length + 1}`;

  // Event listeners
  for (const cls of [".rr-exop", ".rr-var", ".rr-all-nights", ".rr-night"]) {
    const el = qs(cls, entry);
    if (el) el.addEventListener("change", () => syncRerunEntry(fieldset));
  }

  qs(".rr-add", entry).addEventListener("change", (ev) => {
    if (ev.target.checked) {
      addRerunEntry();
    } else {
      removeEntriesAfter(fieldset);
    }
  });

  fieldset.addEventListener("focusin", () => {
    for (const e of qsa(".rerun-entry.active")) e.classList.remove("active");
    fieldset.classList.add("active");
  });

  const container = $("rerun-entries");
  if (after) {
    after.parentNode.insertBefore(fieldset, after.nextSibling);
  } else {
    container.appendChild(fieldset);
  }

  fieldset.classList.add("active");
  syncRerunEntry(fieldset);
}

function removeEntriesAfter(entry) {
  const allEntries = qsa(".rerun-entry");
  const entryIndex = Array.from(allEntries).indexOf(entry);
  const toRemove = Array.from(allEntries).slice(entryIndex + 1);
  for (const e of toRemove) {
    e.remove();
  }
}

function readRerunEntry(entry) {
  const exopChecked = qs(".rr-exop", entry).checked;
  const varChecked = qs(".rr-var", entry).checked;
  const nightVal = qs(".rr-night", entry).value;
  const night_id = nightVal ? parseInt(nightVal, 10) : null;

  const exop = exopChecked ? {
    tc_guess: qs(".rr-tc", entry).value ? parseFloat(qs(".rr-tc", entry).value) : null,
    width_guess_h: qs(".rr-width", entry).value ? parseFloat(qs(".rr-width", entry).value) : null,
  } : null;

  const var_ = varChecked ? {
    period_guess: qs(".rr-period", entry).value ? parseFloat(qs(".rr-period", entry).value) : null,
    all_nights: qs(".rr-all-nights", entry).checked,
  } : null;

  return { night_id, exop, var: var_ };
}

function validateRerunEntries(entries) {
  const errors = [];
  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    if (!entry.exop && !entry.var) {
      errors.push(`entry ${i + 1}: tick EXOP and/or VAR`);
    }
    // Server will validate the rest; we can add client-side checks here if desired
  }
  return errors;
}

function reprocessGuess(r) {
  let msg = "";
  if (r.kind === "variable") {
    msg = `P ${fmtValue(r.period_guess, 7)} d`;
  } else {
    msg = `tc ${fmtValue(r.tc_guess, 8)}, width ${fmtValue(r.width_guess_h, 3)} h`;
  }
  msg += r.night_label ? ", night " + r.night_label : (r.kind === "variable" ? ", all nights" : "");
  return msg;
}

// A finished request's result as text, plus the action it offers (if any).
function reprocessResult(r, tdResult, tdAction) {
  if (r.status === "failed") {
    tdResult.textContent = r.error || "failed";
    tdResult.className = "warn";
    return;
  }
  if (r.status === "queued") {
    tdResult.textContent = `queued (${r.requests_ahead} ahead)`;
    return;
  }
  if (r.status === "running") {
    tdResult.textContent = "running...";
    return;
  }
  const res = r.result || {};
  if (r.kind === "variable") {
    if (!res.found) {
      tdResult.textContent = res.verify_note || "no peak near the guess";
      return;
    }
    const usable = res.verify_status !== "long_period_needs_tie";
    tdResult.textContent = `P = ${fmtPm(res.period, res.period_err, 8)} d (${res.input}, `
      + `${res.n_nights} nights, coverage ${res.phase_coverage === null ? "?" : (100 * res.phase_coverage).toFixed(0)}%, `
      + `${fmtValue(res.n_cycles, 3)} cycles); ${res.verify_status || ""}`
      + (res.delta !== null && res.delta !== undefined
        ? `; delta ${fmtPm(res.delta, res.delta_err, 3)} d (harmonic ${res.harmonic})` : "")
      + (res.verify_note ? `; ${res.verify_note}` : "");
    if (usable) tdAction.appendChild(adoptButton(res.est_id));
    const plot = document.createElement("button");
    plot.textContent = "Phase plot";
    plot.addEventListener("click", () => loadPhase(res.period));
    tdAction.appendChild(plot);
  } else {
    tdResult.textContent = `event ${res.det_id}: tc ${fmtPm(res.tc, res.tc_err, 9)}, depth `
      + `${fmtPm(res.depth, res.depth_err, 4)}, T14 `
      + `${res.t14_lower_limit ? ">= " + fmtValue(res.t14_h, 3) : fmtPm(res.t14_h, res.t14_err, 3)} h`
      + `${res.incomplete_reason ? " (incomplete: " + res.incomplete_reason + ")" : ""}, chi2r `
      + `${fmtValue(res.chi2_red, 3)}, ${res.n_matches} matching-transit pair(s)`
      + (res.search_det_id ? `; search event ${res.search_det_id} kept` : "");
    const btn = document.createElement("button");
    btn.textContent = "Plot night";
    btn.addEventListener("click", () => refreshAndPlotNight(res.night_id));
    tdAction.appendChild(btn);
  }
}

// a new user transit event: reload the object (events, matches) and draw its night
async function refreshAndPlotNight(nightId) {
  const objId = state.currentObject.object.obj_id;
  const resp = await fetch(`/api/object/${objId}`);
  if (resp.ok) {
    const data = await resp.json();
    state.currentObject = data;
    renderTransitEvents(data.transit_events);
    renderTransitMatches(data.transit_matches);
    renderDetections(data.detections);
    renderPeriodEstimates(data.period_estimates);
  }
  loadNightLc(nightId);
}

function renderReprocess(data) {
  const tbody = qs("#detail-reprocess tbody");
  tbody.innerHTML = "";
  for (const r of data.requests) {
    const tr = document.createElement("tr");
    const tdResult = document.createElement("td");
    const tdAction = document.createElement("td");
    reprocessResult(r, tdResult, tdAction);
    const cells = [
      String(r.req_id), r.kind, reprocessGuess(r),
      r.requested_at ? String(r.requested_at).replace("T", " ").slice(0, 19) : "", r.status,
    ];
    for (const text of cells) {
      const td = document.createElement("td");
      td.textContent = text;
      if (r.note) td.title = r.note;
      tr.appendChild(td);
    }
    tr.appendChild(tdResult);
    tr.appendChild(tdAction);
    tbody.appendChild(tr);
  }
  $("rp-queue").textContent =
    `queue: ${data.queue.queued} queued, ${data.queue.running} running (all objects)`;
  renderRerunBadge(data.requests);
}

// Status badge in the detail header: RERUN pending, else how the newest request ended.
function renderRerunBadge(requests) {
  const badge = $("rerun-badge");
  const running = requests.filter((r) => r.status === "running").length;
  const queued = requests.filter((r) => r.status === "queued").length;
  const newest = requests[0];
  let text = "";
  let cls = "";
  let title = "";
  if (running + queued > 0) {
    cls = "pending";
    text = `RERUN ${running ? "running..." : "queued"} (${running + queued} pending)`;
    title = "see the request history below";
  } else if (newest && newest.status === "failed") {
    cls = "failed";
    const error = newest.error || "failed";
    text = `RERUN failed: ${error.length > 80 ? error.slice(0, 80) + "..." : error}`;
    title = error;
  } else if (newest && newest.status === "done") {
    cls = "done";
    text = "RERUN done — see results below";
    title = "finished " + String(newest.finished_at).replace("T", " ").slice(0, 19);
  }
  badge.hidden = !text;
  badge.className = `rerun-badge ${cls}`;
  badge.textContent = text;
  badge.title = title;
}

// Load this object's requests; keep polling every 3 s while one is queued or running.
async function loadReprocess() {
  if (state.reprocessTimer) {
    clearTimeout(state.reprocessTimer);
    state.reprocessTimer = null;
  }
  if (!state.currentObject) return;
  const objId = state.currentObject.object.obj_id;
  const resp = await fetch(`/api/object/${objId}/reprocess`);
  if (!resp.ok || !state.currentObject || state.currentObject.object.obj_id !== objId) return;
  const data = await resp.json();
  const wasPending = state.reprocessPending;
  state.reprocessPending = data.requests.some((r) => r.status === "queued" || r.status === "running");
  renderReprocess(data);
  if (state.reprocessPending) {
    state.reprocessTimer = setTimeout(loadReprocess, 3000);
  } else if (wasPending) {
    // a request just finished: pick up its new estimate / event
    const detail = await fetch(`/api/object/${objId}`);
    if (detail.ok) {
      const d = await detail.json();
      state.currentObject = d;
      renderTransitEvents(d.transit_events);
      renderTransitMatches(d.transit_matches);
      renderDetections(d.detections);
      renderPeriodEstimates(d.period_estimates);
      renderMeta(d.object);
      loadPhase(null);
    }
  }
}

async function submitReprocess() {
  const objId = state.currentObject.object.obj_id;
  const entries = [];
  for (const el of qsa(".rerun-entry")) {
    entries.push(readRerunEntry(el));
  }

  // Validate client-side
  const errors = validateRerunEntries(entries);
  if (errors.length > 0) {
    for (let i = 0; i < qsa(".rerun-entry").length; i++) {
      const entry = qsa(".rerun-entry")[i];
      const msg = qs(".rr-msg", entry);
      const entryErrors = errors.filter(e => e.startsWith(`entry ${i + 1}:`));
      msg.textContent = entryErrors.length > 0 ? entryErrors[0].replace(/^entry \d+: /, "") : "";
    }
    $("rp-msg").textContent = errors.join("; ");
    $("rp-msg").className = "warn";
    return;
  }

  const body = {
    entries,
    note: $("rp-note").value || null,
  };

  const resp = await fetch(`/api/object/${objId}/reprocess`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json();
  if (!resp.ok) {
    $("rp-msg").textContent = typeof data.detail === "string" ? data.detail : "request rejected";
    $("rp-msg").className = "warn";
    return;
  }
  $("rp-msg").className = "";
  const ids = data.requests.map(r => r.req_id).join(", ");
  $("rp-msg").textContent = `queued ${data.requests.length} request(s): ${ids}`;

  // Clear entries and add one fresh entry
  $("rerun-entries").innerHTML = "";
  addRerunEntry();

  state.reprocessPending = true;
  loadReprocess();
  pollPendingReruns();
}

// ---------------------------------------------------------------------
// Pending RERUNs of all objects (header indicator, finish notices)
// ---------------------------------------------------------------------

const RERUN_POLL_MS = 10000;

// A non-blocking notice that a request finished, with a link to open the object.
function showRerunNotice(r) {
  const box = $("rerun-notices");
  const div = document.createElement("div");
  div.className = `rerun-notice ${r.status}`;
  const msg = document.createElement("span");
  msg.textContent = `RERUN for obj ${r.obj_id}${r.obj_name ? " (" + r.obj_name + ")" : ""} finished: ${r.status}`
    + (r.status === "failed" && r.error ? ` (${r.error.slice(0, 120)})` : "");
  const open = document.createElement("button");
  open.textContent = "Open";
  open.addEventListener("click", () => {
    div.remove();
    loadObject(r.obj_id);
  });
  const close = document.createElement("button");
  close.textContent = "×";
  close.title = "dismiss";
  close.addEventListener("click", () => div.remove());
  div.append(msg, open, close);
  box.appendChild(div);
  while (box.children.length > 5) box.firstChild.remove();
}

// "N RERUNs pending" button in the header; its list names the objects (click to load one)
function renderPendingReruns(requests, queue) {
  const total = queue.queued + queue.running;
  $("rerun-global").hidden = total === 0;
  $("btn-rerun-list").textContent = `${total} RERUN${total === 1 ? "" : "s"} pending`
    + (queue.running ? ` (${queue.running} running)` : "");
  const list = $("rerun-list");
  list.innerHTML = "";
  if (total === 0) list.hidden = true;
  const byObj = new Map();
  for (const r of requests) {
    if (!byObj.has(r.obj_id)) byObj.set(r.obj_id, { name: r.obj_name, n: 0, running: 0 });
    const o = byObj.get(r.obj_id);
    o.n += 1;
    if (r.status === "running") o.running += 1;
  }
  for (const [objId, o] of byObj) {
    const a = document.createElement("a");
    a.href = "#";
    a.textContent = `obj ${objId}${o.name ? " (" + o.name + ")" : ""}: ${o.n} pending`
      + (o.running ? ", running" : "");
    a.addEventListener("click", (ev) => {
      ev.preventDefault();
      list.hidden = true;
      loadObject(objId);
    });
    list.appendChild(a);
  }
}

// Ask which requests (all objects) are queued or running, and how the ones seen pending
// before ended. Polls every 10 s while any is pending and stops at 0; a search or a
// submitted RERUN starts it again. Read-only: nothing is ever queued from here.
async function pollPendingReruns() {
  if (state.rerunPolling) return;
  state.rerunPolling = true;
  if (state.rerunPollTimer) {
    clearTimeout(state.rerunPollTimer);
    state.rerunPollTimer = null;
  }
  let data = null;
  try {
    const params = new URLSearchParams({ status: "queued,running" });
    for (const id of state.rerunWatch.keys()) params.append("watch", String(id));
    const resp = await fetch("/api/reprocess?" + params.toString());
    if (resp.ok) data = await resp.json();
  } catch (err) {
    data = null; // server briefly unreachable: try again below
  }
  state.rerunPolling = false;
  if (data) {
    const pendingIds = new Set(data.requests.map((r) => r.req_id));
    let finished = false;
    for (const r of data.watched) {
      if (r.status !== "done" && r.status !== "failed") continue;
      finished = true;
      showRerunNotice(r);
      if (isCurrentObject(r.obj_id)) loadReprocess();
    }
    // watch what is pending now: whatever leaves the list is reported by the next poll
    state.rerunWatch = new Map(data.requests.map((r) => [r.req_id, r]));
    renderPendingReruns(data.requests, data.queue);
    // the results table's RERUN markers follow any change of the pending set
    const key = Array.from(pendingIds).sort((a, b) => a - b).join(",");
    if (finished || (state.rerunKey !== null && key !== state.rerunKey)) doSearch();
    state.rerunKey = key;
    if (data.queue.queued + data.queue.running > 0) {
      state.rerunPollTimer = setTimeout(pollPendingReruns, RERUN_POLL_MS);
    }
  } else if (state.rerunWatch.size > 0) {
    state.rerunPollTimer = setTimeout(pollPendingReruns, RERUN_POLL_MS);
  }
}

// ---------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------

function init() {
  qs("#search-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    state.offset = 0;
    doSearch();
    pollPendingReruns();
  });
  $("btn-reset").addEventListener("click", resetFilters);
  $("btn-export-csv").addEventListener("click", exportCsv);
  $("btn-run-sql").addEventListener("click", runSql);
  $("btn-prev-page").addEventListener("click", () => {
    state.offset = Math.max(0, state.offset - state.pageSize);
    doSearch();
  });
  $("btn-next-page").addEventListener("click", () => {
    state.offset += state.pageSize;
    doSearch();
  });
  $("page-size-select").addEventListener("change", (ev) => {
    state.pageSize = parseInt(ev.target.value, 10);
    state.offset = 0;
    doSearch();
  });
  $("periodogram-scope-select").addEventListener("change", loadPeriodogram);
  $("periodogram-method-select").addEventListener("change", loadPeriodogram);
  $("btn-phase-x2").addEventListener("click", () => {
    const current = phasePeriodFromInput();
    if (current !== null) loadPhase(current * 2);
  });
  $("btn-phase-div2").addEventListener("click", () => {
    const current = phasePeriodFromInput();
    if (current !== null) loadPhase(current / 2);
  });
  $("phase-period-input").addEventListener("change", () => {
    const current = phasePeriodFromInput();
    if (current !== null) loadPhase(current);
  });
  $("phase-candidate-select").addEventListener("change", (ev) => {
    if (ev.target.value !== "") loadPhase(parseFloat(ev.target.value));
  });
  for (const id of ["phase-epoch-select", "phase-range-check", "phase-model-check"]) {
    $(id).addEventListener("change", plotPhase);
  }
  $("lc-unit-select").addEventListener("change", () => {
    replotLightcurve();
    plotPhase();
  });
  $("rerun-check").addEventListener("change", onRerunToggle);
  $("btn-rerun-list").addEventListener("click", () => {
    $("rerun-list").hidden = !$("rerun-list").hidden;
  });
  $("btn-rp-submit").addEventListener("click", submitReprocess);
  $("btn-save-edit").addEventListener("click", saveEdit);
  $("btn-reset-exop-auto").addEventListener("click", resetExopAuto);
  $("btn-reset-var-auto").addEventListener("click", resetVarAuto);
  $("btn-reset-period-auto").addEventListener("click", resetPeriodAuto);

  renderResultsHeader();
  doSearch();
  pollPendingReruns();
}

document.addEventListener("DOMContentLoaded", init);
