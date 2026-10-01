# ruff: noqa  -- one-off diagnostic script kept with its figures, not library code
"""obj 3419 / night 20251106 / T80S: target/median-comp and median-of-target/comp_i light curves (read-only)."""
import json, subprocess, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from relphot.comparison import _median_ensemble, MEDIAN_SE_FACTOR
from relphot.numeric import nanmedian_quiet, mad_sigma
from relphot.lightcurve import point_to_point_sigma
from relphot.db.analyze import _fit_transit_shape, _NightData, _TransitDet, _trapezoid_shape

import os
OUT = os.path.dirname(os.path.abspath(__file__)) + os.sep
D = "/ssdsto1/data/T80S_reduced/20251106/relphot/"
OBJ, NAME, NIGHT, STAR, TILE, AP = 3419, "RP J051758.97-702101.6", "20251106", 2936, 40, 0
DET_TC, DET_DEPTH, DET_DUR_H = 2460986.763111168, 0.0477127, 0.55

night = np.load(D + "night.npz", allow_pickle=True)
ref = np.load(D + "ref.npz", allow_pickle=True)
lcz = np.load(D + "lc/night_lc.npz", allow_pickle=True)
mem = np.load(D + "lc/night_lc_members.npz", allow_pickle=True)
fm = json.loads(str(night["frame_meta_json"]))
t_all = np.array([f["bjd_tdb"] for f in fm])
nF = len(t_all)

# --- star identification (DB star_id == array index; verified by RA/Dec)
assert int(ref["tile_core_tile"][STAR]) == TILE and int(lcz["star_best_aper"][STAR]) == AP
print("star", STAR, "RA/Dec npz", night["ra"][STAR], night["dec"][STAR])

# --- frame cuts as the pipeline: frame_kept & flags ok (epoch_ok)
epoch_ok = lcz["epoch_ok"][STAR].astype(bool)
frame_kept = ref["frame_kept"].astype(bool)
print("frames total", nF, "kept", frame_kept.sum(), "epoch_ok(target)", epoch_ok.sum())

# --- target decorrelated relative flux T = lc * ens_weighted = rel_flux * corr  (corr = lc/lc_raw)
lc_st = lcz["lc"][STAR, :, AP].astype(float); lc_raw_st = lcz["lc_raw"][STAR, :, AP].astype(float)
ens_w = lcz["comparison_ensemble"][TILE, :, AP]; sig_ens_w = lcz["comparison_sigma_ensemble"][TILE, :, AP]
rel = ref["relative_flux"][STAR, :, AP]
with np.errstate(invalid="ignore", divide="ignore"):
    corr = lc_st / lc_raw_st
T = lc_st * ens_w
assert np.nanmax(np.abs(T / (rel * corr) - 1)) < 1e-5, "T consistency"
R = ref["R"][TILE, :, AP]
sig_T = night["fluxerr"][STAR, :, AP].astype(float) / R * corr         # target photon error (rel-flux units, decorrelated scale)
# consistency with stored lc_err_raw
chk = np.sqrt((sig_T / ens_w) ** 2 + (lc_raw_st * corr * sig_ens_w / ens_w) ** 2)
print("stored lc_err_raw vs reconstructed max rel diff:", np.nanmax(np.abs(chk / lcz["lc_err_raw"][STAR, :, AP] - 1)))

# --- comparison set (tile 40, aperture 0), target excluded
k = int(np.where((mem["used_pairs"][:, 0] == TILE) & (mem["used_pairs"][:, 1] == AP))[0][0])
a, b = int(mem["comp_offsets"][k]), int(mem["comp_offsets"][k + 1])
comp_star = mem["comp_star"][a:b]
c = mem["comp_norm_flux"][a:b].astype(float)               # each comp / its own baseline (median) -> median 1
keep_c = comp_star != STAR
c = c[keep_c]; comp_star = comp_star[keep_c]
c = np.where(c > 0, c, np.nan)
N = c.shape[0]
print("N comps", N, "target in set:", STAR in mem["comp_star"][a:b], " median of per-comp medians", np.nanmedian(np.nanmedian(c, axis=1)))
ok_frame = epoch_ok & frame_kept & np.isfinite(T) & (sig_T > 0)

# ===================== LC 1: target / median comparison
ens_m, sig_ens_m = _median_ensemble(c)                      # per-frame median over comps, 1.2533*MAD_sigma/sqrt(n)
with np.errstate(invalid="ignore", divide="ignore"):
    lc1 = T / ens_m
    s_phot1 = sig_T / ens_m
    s_ens1 = lc1 * sig_ens_m / ens_m
    s1 = np.sqrt(s_phot1 ** 2 + s_ens1 ** 2)
m1 = np.nanmedian(lc1[ok_frame])
lc1, s1 = lc1 / m1, s1 / m1
lc1 = np.where(ok_frame, lc1, np.nan); s1 = np.where(ok_frame, s1, np.nan)

# ===================== LC 2: median over i of target / comp_i (each normalised to own median)
with np.errstate(invalid="ignore", divide="ignore"):
    r = T[None, :] / c
r = np.where(ok_frame[None, :], r, np.nan)
r = r / nanmedian_quiet(r, axis=1)[:, None]
lc2 = nanmedian_quiet(r, axis=0)
n_used = np.count_nonzero(np.isfinite(r), axis=0)
sig_med2 = np.where(n_used > 0, MEDIAN_SE_FACTOR * mad_sigma(r, axis=0) / np.sqrt(np.maximum(n_used, 1)), np.nan)
Tn = T / np.nanmedian(T[ok_frame])
with np.errstate(invalid="ignore", divide="ignore"):
    s_phot2 = (sig_T / np.nanmedian(T[ok_frame])) * lc2 / Tn
s2_raw = np.sqrt(s_phot2 ** 2 + sig_med2 ** 2)
m2 = np.nanmedian(lc2[ok_frame]); lc2 = lc2 / m2; s2_raw = s2_raw / m2
lc2 = np.where(ok_frame, lc2, np.nan); s2_raw = np.where(ok_frame, s2_raw, np.nan)

# pipeline error inflation ("excess", blended star): factor = max(1, p2p / median(err))
def inflate(lc, err):
    p2p = point_to_point_sigma(np.where(ok_frame, lc, np.nan)[None, :, None], ok_frame[None, :], 10)[0, 0]
    return max(1.0, p2p / np.nanmedian(err)), p2p
f1, p1 = inflate(lc1, s1); f2, p2 = inflate(lc2, s2_raw)
print("p2p sigma LC1 %.5f LC2 %.5f ; median err LC1 %.5f LC2 %.5f ; inflation factors %.3f %.3f ; stored err_scale %s blended %s"
      % (p1, p2, np.nanmedian(s1), np.nanmedian(s2_raw), f1, f2, lcz["err_scale"][STAR, AP], lcz["blended"][STAR]))
APPLY_INFL = False   # figure errors follow the user's stated formulas; factor reported only
s1p, s2p = (s1 * f1, s2_raw * f2) if APPLY_INFL else (s1, s2_raw)

# ===================== stored DB light curve (weighted ensemble)
q = ("select bjd_tdb, flux, flux_err from relphot.lightcurve where obj_id=3419 and night_id=3")
import psycopg  # noqa
env = dict(l.strip().split("=", 1) for l in open("/home/herpich/.config/relphot/relphotdb.env") if "=" in l and not l.startswith("#"))
dsn = env["RELPHOT_DB_DSN"] if False else None
raw = subprocess.run(["podman", "exec", "relphotdb-db", "psql", "-U", "postgres", "-d", "relphot", "-At", "-F", "|", "-c",
                      "select array_to_string(bjd_tdb,','), array_to_string(flux,','), array_to_string(flux_err,',') from relphot.lightcurve where obj_id=3419 and night_id=3"],
                     capture_output=True, text=True, check=True).stdout.strip().split("|")
db_t, db_f, db_e = (np.array([float(x) for x in s.split(",")]) for s in raw)
assert np.allclose(db_t, t_all[epoch_ok])
db_norm = np.median(db_f)
dbf, dbe = db_f / db_norm, db_e / db_norm
print("DB LC == npz lc (decorrelated) max rel:", np.max(np.abs(db_f / lc_st[epoch_ok] - 1)))

# ===================== LC 3: stored weighted-ensemble LC (stored lc, stored lc_err = lc_err_raw * err_scale)
lc3_n = np.median(lc_st[ok_frame])
lc3 = np.where(ok_frame, lc_st / lc3_n, np.nan)
s3 = np.where(ok_frame, lcz["lc_err"][STAR, :, AP].astype(float) / lc3_n, np.nan)
assert np.allclose(s3[epoch_ok] * lc3_n, db_e, rtol=1e-5)
print("LC3 err_scale", lcz["err_scale"][STAR, AP])

# ===================== trapezoid fit with the DB's analyze fitter
def fit(t, y, e, tag):
    nd = _NightData(night_id=3, night_date=None, bjd=t, flux=y, flux_err=e, telescope="T80S")
    det = _TransitDet(det_id=5342, night_id=3, tc=DET_TC, depth=DET_DEPTH, duration_h=DET_DUR_H, flags="OK")
    out = _fit_transit_shape(nd, det, None, None)
    d0 = DET_DUR_H / 24
    tn = t; med = np.median(y[np.isfinite(y)]); yy = y / med; ee = e / med
    tc, dep, t14, ing = out["tc"], out["depth"], out["t14_h"] / 24.0, out["ingress_frac"]
    sel = np.abs(t - DET_TC) <= max(1.5 * d0, d0 + 1.0 / 24.0)
    mshape = lambda tt: 1.0 - dep * _trapezoid_shape(tt, tc, t14, ing)
    w = 1.0 / ee[sel] ** 2; msel = mshape(t[sel])
    base = np.sum(w * yy[sel] * msel) / np.sum(w * msel ** 2)      # converged weighted-LS baseline (fit's 5th parameter)
    print(tag, {k_: (round(v, 5) if isinstance(v, float) else v) for k_, v in out.items()}, "baseline", round(base, 5))
    return out, base, mshape, med

res = {}
for tag, lc, err in (("LC1", lc1, s1p), ("LC2", lc2, s2p), ("LC3", lc3, s3)):
    o = ok_frame
    out, base, mshape, med = fit(t_all[o], lc[o], err[o], tag)
    res[tag] = (out, base, mshape)
o_db, b_db, _, _ = fit(db_t, db_f, db_e, "DB-LC (refit by me)")

# ===================== plotting
def p16_84(x): return np.nanpercentile(x, 16), np.nanpercentile(x, 84)

def figure(tag, lc, err, title_main, fname, errtxt, overlay=True):
    o = ok_frame
    out, base, mshape, = res[tag][0], res[tag][1], res[tag][2]
    tt = t_all
    mod = lambda x: base * mshape(x)
    tfine = np.linspace(t_all.min(), t_all.max(), 3000)
    resid = lc - mod(tt)
    rr = resid[o]
    mean_, (p16, p84) = np.nanmean(lc[o]), p16_84(lc[o])
    rmean, (r16, r84) = np.nanmean(rr), p16_84(rr)
    rms = np.sqrt(np.nanmean(rr ** 2))
    diff = lc[o] - dbf[np.isin(np.where(epoch_ok)[0], np.where(o)[0])]  # 0 for the stored LC itself
    rms_db = np.sqrt(np.mean(diff ** 2)); mean_db_diff = np.mean(diff)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 7), dpi=130, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    x0 = 2460000.0
    if overlay: ax.plot(db_t - x0, dbf, ".", color="0.7", ms=4, zorder=1, label=f"stored DB LC (weighted ens.), RMS diff {rms_db*1e3:.1f} mmag-ish (x1e-3 flux)".replace(" mmag-ish (x1e-3 flux)", "e-3"))
    ax.errorbar(tt[o] - x0, lc[o], err[o], fmt="o", ms=3.5, color="C0", ecolor="C0", elinewidth=0.8, capsize=0, zorder=3, label="new LC")
    ax.plot(tfine - x0, mod(tfine), "-", color="C3", lw=1.6, zorder=4, label="trapezoid fit")
    for v, lab, ls in ((mean_, f"mean {mean_:.4f}", "--"), (p16, f"P16 {p16:.4f}", "--"), (p84, f"P84 {p84:.4f}", "--")):
        ax.axhline(v, color="k", ls=ls, lw=0.8, alpha=0.7)
        ax.text(1.002, v, lab, transform=ax.get_yaxis_transform(), va="center", fontsize=7)
    ax.set_ylabel("normalised flux"); ax.legend(fontsize=7.5, loc="lower left", ncol=3)
    ax.set_title(f"{NAME}  (obj {OBJ}, EXOP)  night {NIGHT} T80S  ap {AP}  N_comp={N}\n{title_main}  |  depth={out['depth']:.4f}, T14={out['t14_h']*60:.1f} min, tc={out['tc']-x0:.5f}, resid RMS={rms*1e3:.2f}e-3",
                 fontsize=9)
    fig.text(0.08, 0.008, errtxt, ha="left", va="bottom", fontsize=7, color="0.25")
    axr.errorbar(tt[o] - x0, rr, err[o], fmt="o", ms=3, color="C0", elinewidth=0.7)
    axr.axhline(0, color="C3", lw=1.0)
    for v, lab in ((rmean, f"mean {rmean:+.4f}"), (r16, f"P16 {r16:+.4f}"), (r84, f"P84 {r84:+.4f}")):
        axr.axhline(v, color="k", ls="--", lw=0.8, alpha=0.7)
        axr.text(1.002, v, lab, transform=axr.get_yaxis_transform(), va="center", fontsize=7)
    axr.set_ylabel("obs - model"); axr.set_xlabel("BJD_TDB - 2460000")
    axr.set_xlim(t_all.min() - x0 - 0.003, t_all.max() - x0 + 0.003)
    fig.subplots_adjust(left=0.08, right=0.89, top=0.9, bottom=0.16)
    fig.savefig(OUT + fname); plt.close(fig)
    print(f"{tag}: mean {mean_:.5f} P16 {p16:.5f} P84 {p84:.5f} | resid mean {rmean:+.5f} P16 {r16:+.5f} P84 {r84:+.5f} RMS {rms:.5f} | "
          f"RMS(diff vs DB LC) {rms_db:.5f} mean diff {mean_db_diff:+.5f} | median err {np.nanmedian(err[o]):.5f} RMS in-window "
          f"{np.sqrt(np.mean(rr[np.abs(tt[o]-DET_TC)<=0.0646]**2)):.5f} | RMS LC off-transit-ish robust MAD {1.4826*np.median(np.abs(rr-np.median(rr))):.5f}")
    print("   fit:", {k_: out[k_] for k_ in ("tc","tc_err","depth","depth_err","t14_h","t14_err","ingress_frac","ingress_err","chi2_red","n_points","converged","t14_lower_limit","incomplete_reason")}, "baseline", round(base,5))

e1 = ("LC = T / M,  T = target flux (decorrelated, ref-normalised);  M(t) = median_i c_i(t),  c_i = comp_i / its own median;\n"
      "LC normalised to median 1.   err = sqrt[ (sigma_T/M)^2 + (LC*sigma_M/M)^2 ] / norm,  sigma_M = sqrt(pi/2)*1.4826*MAD_i(c_i)/sqrt(N)")
e2 = ("LC = median_i r_i(t),  r_i = T/c_i normalised to its own median, then LC to median 1.\n"
      "err = sqrt[ sigma_phot^2 + (sqrt(pi/2)*1.4826*MAD_i(r_i)/sqrt(N))^2 ],  sigma_phot = LC*sigma_T/T")
figure("LC1", lc1, s1p, "1. Target / median comparison", "obj3419_20251106_target_over_median_comp.png", e1)
figure("LC2", lc2, s2p, "2. Median of target / comp_i", "obj3419_20251106_median_target_over_comp.png", e2)
e3 = ("LC = stored weighted-clipped-mean-ensemble LC (DB lightcurve.flux = night_lc.npz lc), normalised to median 1.\n"
      "err = stored lc_err = lc_err_raw * err_scale (err_scale = 1.0 for this star/ap); lc_err_raw = sqrt[(sigma_T/E)^2 + (lc_raw*sigma_E/E)^2] * decorr. factor, E = weighted clipped mean")
figure("LC3", lc3, s3, "3. Stored weighted ensemble", "obj3419_20251106_stored_weighted_ensemble.png", e3, overlay=False)
print("diff LC1-LC2 rms", np.sqrt(np.nanmean((lc1 - lc2) ** 2)), " n frames with n_used<N:", int((n_used < N).sum()), " min n_used", n_used[ok_frame].min())
np.savez(OUT + "lcs.npz", t=t_all, lc1=lc1, s1=s1, lc2=lc2, s2=s2_raw)
