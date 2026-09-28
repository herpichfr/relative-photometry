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
  "obj_id", "name", "ra", "dec", "class", "known", "source_db", "known_name",
  "known_type", "period", "period_source", "known_period", "mean_mag", "n_nights",
  "best_snr", "depth", "duration_h", "amplitude", "status", "first_night", "last_night",
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

function renderResultsBody(rows) {
  const tbody = qs("#results-table tbody");
  tbody.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    const tdLoad = document.createElement("td");
    const btn = document.createElement("button");
    btn.textContent = "Load";
    btn.addEventListener("click", () => loadObject(row.obj_id));
    tdLoad.appendChild(btn);
    tr.appendChild(tdLoad);
    for (const col of RESULT_COLUMNS) {
      const td = document.createElement("td");
      const v = row[col];
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
  ["ra", "RA (deg)"], ["dec", "Dec (deg)"], ["class", "CLASS"], ["period", "PERIOD (d)"],
  ["period_source", "period source"], ["known", "KNOWN"], ["source_db", "SOURCE_DB"],
  ["known_name", "known name"], ["known_type", "known type"], ["known_period", "known period (d)"],
  ["status", "status"], ["gaia_id", "Gaia ID"], ["mean_mag", "mean mag"],
  ["n_nights", "n nights"], ["notes", "notes"],
];

function renderMeta(obj) {
  const table = $("detail-meta");
  table.innerHTML = "";
  const display = Object.assign({}, obj, {
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
    const values = [r.kind, source, r.snr, r.depth, r.duration_h, r.tier, r.period, r.fap];
    for (const v of values) {
      const td = document.createElement("td");
      td.textContent = v === null || v === undefined ? "" : String(v);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
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

function findBestPeriodogram(periodograms, objClass) {
  if (!periodograms || periodograms.length === 0) return null;

  // Preference 1: scope='combined' AND method='BLS' if class is 'EXOP'
  if (objClass === 'EXOP') {
    const entry = periodograms.find((p) => p.scope === 'combined' && p.method === 'BLS');
    if (entry) return entry;
  }

  // Preference 2: scope='combined' AND method='LS'
  const combined = periodograms.find((p) => p.scope === 'combined' && p.method === 'LS');
  if (combined) return combined;

  // Preference 3: first 'night:*' LS entry with smallest fap
  const nightEntries = periodograms.filter((p) => p.scope && p.scope.startsWith('night:') && p.method === 'LS');
  if (nightEntries.length > 0) {
    return nightEntries.reduce((best, current) => {
      const bestFap = best.fap !== null && best.fap !== undefined ? best.fap : Infinity;
      const currentFap = current.fap !== null && current.fap !== undefined ? current.fap : Infinity;
      return currentFap < bestFap ? current : best;
    });
  }

  return null;
}

function updatePhasePeriodNote(noteText) {
  const noteElem = $("phase-period-note");
  if (noteElem) {
    noteElem.textContent = noteText;
  }
}

function prefillEditBox(obj) {
  $("edit-class-select").value = obj.class || "";
  $("edit-status-select").value = obj.status || "";
  $("edit-notes-textarea").value = obj.notes || "";
  $("edit-period-input").value = obj.period === null || obj.period === undefined ? "" : obj.period;

  // Phase period prefill with fallback
  if (obj.period !== null && obj.period !== undefined) {
    $("phase-period-input").value = obj.period;
    updatePhasePeriodNote(`PERIOD (${obj.period_source || ""})`);
  } else {
    // No PERIOD set; try to find a fallback from periodograms
    const best = findBestPeriodogram(state.currentObject.periodograms, obj.class);
    if (best && best.peak_period) {
      $("phase-period-input").value = best.peak_period;
      updatePhasePeriodNote(`PERIOD not set — using ${best.method} ${best.scope} periodogram peak`);
    } else {
      $("phase-period-input").value = "";
      updatePhasePeriodNote("No period available — enter one");
    }
  }

  $("edit-status-msg").textContent = "";

  // Re-render phase plot if data is available (will be no-op if no light curve loaded yet)
  plotPhase();
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
  state.currentCombinedLc = null;

  $("detail-panel").hidden = false;
  renderMeta(data.object);
  renderCatalogMatches(data.catalog_matches);
  renderDetections(data.detections);
  renderNightButtons(data.object, data.nights);
  populatePeriodogramSelectors(data.periodograms);
  prefillEditBox(data.object);
  Plotly.purge("plot-lightcurve");
  Plotly.purge("plot-phase");
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

async function loadNightLc(nightId) {
  state.currentNightId = nightId;
  state.currentCombinedLc = null;
  const objId = state.currentObject.object.obj_id;
  const resp = await fetch(`/api/object/${objId}/lc?night_id=${nightId}`);
  if (!resp.ok) {
    window.alert("failed to load light curve");
    return;
  }
  const lc = await resp.json();
  const x = lc.bjd_tdb.map((t) => t - 2460000);
  const trace = {
    x, y: lc.flux, type: "scatter", mode: "markers",
    error_y: { type: "data", array: lc.flux_err, visible: true },
    text: lc.frame_index.map((fi, i) => `frame ${fi}<br>${lc.file_name[i] || ""}<br>airmass ${lc.airmass[i]}`),
    hoverinfo: "x+y+text",
    marker: { size: 5 },
    name: lc.night_label || `night ${nightId}`,
  };
  const shapes = [];
  const transit = bestTransitForNight(nightId);
  if (transit) {
    const tc = transit.tc_bjd_tdb - 2460000;
    const halfDur = (transit.duration_h || 0) / 24.0 / 2.0;
    const depth = transit.depth || 0;
    const yTop = Math.max(...lc.flux.filter((v) => Number.isFinite(v)));
    shapes.push({
      type: "rect", x0: tc - halfDur, x1: tc + halfDur, y0: 1 - depth, y1: yTop,
      line: { color: "rgba(200,30,30,0.6)" }, fillcolor: "rgba(200,30,30,0.08)",
    });
  }
  Plotly.newPlot("plot-lightcurve", [trace], {
    xaxis: { title: "BJD_TDB - 2460000" },
    yaxis: { title: "relative flux" },
    shapes,
    margin: { t: 20 },
  }, { responsive: true });
}

async function loadCombinedLc() {
  const objId = state.currentObject.object.obj_id;
  const resp = await fetch(`/api/object/${objId}/lc/combined`);
  if (!resp.ok) {
    window.alert("failed to load combined light curve");
    return;
  }
  const lc = await resp.json();
  const byNight = new Map();
  for (let i = 0; i < lc.bjd_tdb.length; i++) {
    const key = lc.night_id[i];
    if (!byNight.has(key)) byNight.set(key, { x: [], y: [], err: [], text: [], label: lc.night_label[i] });
    const bucket = byNight.get(key);
    bucket.x.push(lc.bjd_tdb[i] - 2460000);
    bucket.y.push(lc.value[i]);
    bucket.err.push(lc.value_err[i]);
    bucket.text.push(lc.file_name[i] || "");
  }
  const traces = Array.from(byNight.values()).map((bucket) => ({
    x: bucket.x, y: bucket.y, type: "scatter", mode: "markers",
    error_y: { type: "data", array: bucket.err, visible: true },
    text: bucket.text, hoverinfo: "x+y+text",
    marker: { size: 5 }, name: bucket.label,
  }));
  const layout = {
    xaxis: { title: "BJD_TDB - 2460000" },
    yaxis: {
      title: lc.mode === "tied-mag" ? "tied magnitude" : "relative flux (per-night normalised)",
      autorange: lc.mode === "tied-mag" ? "reversed" : true,
    },
    margin: { t: 20 },
    title: lc.mode === "tied-mag"
      ? "combined (tie-calibrated magnitudes)"
      : "combined (per-night normalised flux -- nights are NOT tied)",
  };
  Plotly.newPlot("plot-lightcurve", traces, layout, { responsive: true });
  state.currentCombinedLc = lc;
}

// ---------------------------------------------------------------------
// Phase diagram
// ---------------------------------------------------------------------

function currentPhaseSourceData() {
  if (state.currentCombinedLc) {
    const lc = state.currentCombinedLc;
    return { bjd: lc.bjd_tdb, value: lc.value, night: lc.night_label };
  }
  return null;
}

async function plotPhase() {
  const period = parseFloat($("phase-period-input").value);
  if (!period || period <= 0) return;

  const source = currentPhaseSourceData();
  let bjd, value, night;
  if (source) {
    ({ bjd, value, night } = source);
  } else if (state.currentNightId !== null) {
    const objId = state.currentObject.object.obj_id;
    const resp = await fetch(`/api/object/${objId}/lc?night_id=${state.currentNightId}`);
    if (!resp.ok) return;
    const lc = await resp.json();
    bjd = lc.bjd_tdb;
    value = lc.flux;
    night = lc.bjd_tdb.map(() => lc.night_label);
  } else {
    return;
  }

  const transit = state.currentObject.detections.find((d) => d.kind === "transit" && d.tc_bjd_tdb !== null);
  const t0 = transit ? transit.tc_bjd_tdb : bjd[0];

  const byNight = new Map();
  for (let i = 0; i < bjd.length; i++) {
    let phase = ((bjd[i] - t0) / period) % 1.0;
    if (phase < -0.5) phase += 1.0;
    if (phase > 1.0) phase -= 1.0;
    const key = night[i];
    if (!byNight.has(key)) byNight.set(key, { x: [], y: [] });
    const bucket = byNight.get(key);
    bucket.x.push(phase);
    bucket.y.push(value[i]);
  }
  const traces = Array.from(byNight.entries()).map(([label, bucket]) => ({
    x: bucket.x, y: bucket.y, type: "scatter", mode: "markers",
    marker: { size: 5 }, name: label || "",
  }));
  Plotly.newPlot("plot-phase", traces, {
    xaxis: { title: "phase", range: [-0.5, 1.0] },
    yaxis: { title: "value" },
    margin: { t: 20 },
  }, { responsive: true });
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
  const classVal = $("edit-class-select").value;
  if (classVal) body.class = classVal;
  const statusVal = $("edit-status-select").value;
  if (statusVal) body.status = statusVal;
  body.notes = $("edit-notes-textarea").value;
  const periodVal = $("edit-period-input").value;
  if (periodVal !== "") body.period = parseFloat(periodVal);
  await patchObject(objId, body);
}

async function resetClassAuto() {
  await patchObject(state.currentObject.object.obj_id, { class_source: "auto" });
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
  $("edit-status-msg").textContent = "saved";
  state.currentObject.object = data;
  renderMeta(data);
  prefillEditBox(data);
}

// ---------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------

function init() {
  qs("#search-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    state.offset = 0;
    doSearch();
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
    const current = parseFloat($("phase-period-input").value);
    if (!Number.isFinite(current) || current <= 0) return;
    $("phase-period-input").value = current * 2;
    plotPhase();
  });
  $("btn-phase-div2").addEventListener("click", () => {
    const current = parseFloat($("phase-period-input").value);
    if (!Number.isFinite(current) || current <= 0) return;
    $("phase-period-input").value = current / 2;
    plotPhase();
  });
  $("phase-period-input").addEventListener("change", plotPhase);
  $("btn-save-edit").addEventListener("click", saveEdit);
  $("btn-reset-class-auto").addEventListener("click", resetClassAuto);
  $("btn-reset-period-auto").addEventListener("click", resetPeriodAuto);

  renderResultsHeader();
  doSearch();
}

document.addEventListener("DOMContentLoaded", init);
