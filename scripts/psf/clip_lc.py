# ruff: noqa
"""Sigma-clip outlier epochs of the PSF light curves, in place, between `relphot lightcurves` and `relphot search`.

usage: python clip_lc.py <relphot_dir> <lc_stem> [k=4.0] [window_minutes=15]

Per star: residual = lc - running median (window of `w` epochs in time order, centre epoch excluded, only epochs not yet
clipped), sigma = 1.4826*MAD of the residuals (floor 0.5*median lc_err), clip |res| > k*sigma, iterate (<=5) until stable.
Transit-safe: a run of >= 3 consecutive (in the time-ordered valid epochs) outliers of the same sign is never clipped.
Clipped epochs: epoch_ok=False and lc/lc_err/lc_raw/lc_err_raw/decorr_lc_corrected = NaN in lc/<stem>.npz; their rows are
removed from lc/<stem>_lightcurves.parquet; rms/chi2_reduced/expected_noise/n_epochs recomputed (relphot.stats.compute_star_stats)
in the npz and in lc/<stem>_starstats.parquet.  Originals are kept in lc/preclip/.
"""
import sys, os, json, shutil
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from relphot.io import load_lightcurves_npz
from relphot.stats import compute_star_stats

rdir, stem = sys.argv[1], sys.argv[2]
K = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
WMIN = float(sys.argv[4]) if len(sys.argv) > 4 else 15.0
lcdir = f'{rdir}/lc'
npz_p = f'{lcdir}/{stem}.npz'; lcp_p = f'{lcdir}/{stem}_lightcurves.parquet'; ss_p = f'{lcdir}/{stem}_starstats.parquet'
os.makedirs(f'{lcdir}/preclip', exist_ok=True)
for p in (npz_p, lcp_p, ss_p):
    b = f'{lcdir}/preclip/{os.path.basename(p)}'
    if not os.path.exists(b): shutil.copy2(p, b)
src = {os.path.basename(p): f'{lcdir}/preclip/{os.path.basename(p)}' for p in (npz_p, lcp_p, ss_p)}   # always clip from the pre-clip originals

night = np.load(f'{rdir}/night.npz', allow_pickle=False)
bjd = np.array([m['bjd_tdb'] for m in json.loads(str(night['frame_meta_json']))])
D = dict(np.load(src[os.path.basename(npz_p)], allow_pickle=False))
lc_result, star_stats, *_ = load_lightcurves_npz(src[os.path.basename(npz_p)])
nS, nF, nA = D['lc'].shape
order = np.argsort(bjd); ts = bjd[order]
cad = float(np.median(np.diff(ts))) * 86400.0
w = int(round(WMIN * 60.0 / cad)); w = max(5, min(31, w)); w += (w % 2 == 0)
print(f'{rdir}: {nS} stars, {nF} frames, cadence {cad:.1f} s -> running-median window {w} epochs ({w*cad/60:.1f} min), k={K}')
ba = D['star_best_aper'].astype(int)
ok0 = D['epoch_ok'][:, order].copy()
Y = np.where(ok0, D['lc'][np.arange(nS), :, 0][:, order] if nA == 1 else
             np.take_along_axis(D['lc'], np.clip(ba, 0, nA - 1)[:, None, None], axis=2)[:, :, 0][:, order], np.nan).astype(np.float64)
Eb = np.where(ok0, D['lc_err'][np.arange(nS), :, 0][:, order] if nA == 1 else
              np.take_along_axis(D['lc_err'], np.clip(ba, 0, nA - 1)[:, None, None], axis=2)[:, :, 0][:, order], np.nan).astype(np.float64)
Y[~np.isfinite(Y)] = np.nan
valid0 = np.isfinite(Y)
errmed = np.nanmedian(np.where(valid0, Eb, np.nan), axis=1)
keep = valid0.copy()
nvalid = valid0.sum(1)
use_star = (nvalid >= 20) & (ba >= 0)
protected_total = 0; prot_first = 0; out_first = 0
for it in range(5):
    Yk = np.where(keep, Y, np.nan)
    pad = w // 2
    Yp = np.pad(Yk, ((0, 0), (pad, pad)), constant_values=np.nan)
    win = sliding_window_view(Yp, w, axis=1).copy()          # (nS, nF, w)
    win[:, :, pad] = np.nan                                  # exclude the epoch itself
    cnt = np.isfinite(win).sum(2)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rm = np.nanmedian(win, axis=2)
        gm = np.nanmedian(Yk, axis=1)
    rm = np.where(cnt >= 3, rm, gm[:, None])
    r = Y - rm
    rk = np.where(keep, r, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sig = 1.4826 * np.nanmedian(np.abs(rk - np.nanmedian(rk, axis=1)[:, None]), axis=1)
    sig = np.maximum(sig, 0.5 * np.nan_to_num(errmed, nan=0.0))
    out = valid0 & use_star[:, None] & (np.abs(r) > K * sig[:, None])
    # transit protection: runs >= 3 consecutive same-sign outliers among the valid epochs are kept
    prot = np.zeros_like(out)
    for s in np.nonzero(out.any(1))[0]:
        idx = np.nonzero(valid0[s])[0]
        o = out[s, idx]; sg = np.sign(r[s, idx])
        j = 0
        while j < len(idx):
            if o[j]:
                e = j
                while e + 1 < len(idx) and o[e + 1] and sg[e + 1] == sg[j]: e += 1
                if e - j + 1 >= 3: prot[s, idx[j:e + 1]] = True
                j = e + 1
            else: j += 1
    newkeep = valid0 & ~(out & ~prot)
    protected_total = int(prot.sum())
    if it == 0: prot_first = int(prot.sum()); out_first = int(out.sum())
    if (newkeep == keep).all(): break
    keep = newkeep
clip_sorted = valid0 & ~keep
clipped = np.zeros((nS, nF), bool); clipped[:, order] = clip_sorted
n_valid = int(valid0.sum()); n_clip = int(clip_sorted.sum())
print(f'iterations {it+1}; valid epochs {n_valid}; clipped {n_clip} = {n_clip/n_valid*100:.3f}%; stars with >=1 clipped {int(clipped.any(1).sum())}/{int(use_star.sum())}; first pass: {out_first} candidate outlier epochs ({out_first/n_valid*100:.3f}%), of which {prot_first} sit in same-sign runs of >=3 consecutive epochs and are never clipped; last pass protected {protected_total}')
for kk in (3.0, 3.5, 5.0):
    pass
# apply
ok_new = D['epoch_ok'] & ~clipped
D['epoch_ok'] = ok_new
for key in ('lc', 'lc_err', 'lc_raw', 'lc_err_raw', 'decorr_lc_corrected'):
    if key in D:
        D[key] = np.where(clipped[:, :, None], np.nan, D[key]).astype(D[key].dtype)
lc_result.lc = D['lc']; lc_result.lc_err = D['lc_err']; lc_result.lc_raw = D['lc_raw']; lc_result.epoch_ok = ok_new
if 'lc_err_raw' in D: lc_result.lc_err_raw = D['lc_err_raw']
new = compute_star_stats(lc_result)
chk = ~clipped.any(1) & (D['n_epochs'][:, 0] > 0) if nA == 1 else None
if chk is not None:
    dif = np.nanmax(np.abs(new.rms[chk, 0] - star_stats.rms[chk, 0]))
    print(f'sanity: recomputed rms of unclipped stars differs from stored by max {dif:.2e}; n_epochs equal: {bool((new.n_epochs[chk,0]==star_stats.n_epochs[chk,0]).all())}')
for key, val in (('rms', new.rms), ('chi2_reduced', new.chi2_reduced), ('expected_noise', new.expected_noise), ('n_epochs', new.n_epochs)):
    D[key] = val
np.savez(npz_p, **D)
lcp = pd.read_parquet(src[os.path.basename(lcp_p)])
bad = clipped[lcp['star_id'].values, lcp['frame'].values]
lcp = lcp[~bad].reset_index(drop=True)
lcp.to_parquet(lcp_p)
ss = pd.read_parquet(src[os.path.basename(ss_p)])
sid = ss['star_id'].values; a = np.clip(ss['best_aperture'].values, 0, nA - 1)
upd = ss['best_aperture'].values >= 0      # stars without a usable aperture keep their original starstats row
for col, arr in (('rms', new.rms), ('chi2_reduced', new.chi2_reduced), ('expected_noise', new.expected_noise), ('n_epochs', new.n_epochs)):
    v = ss[col].values.copy(); v[upd] = arr[sid[upd], a[upd]].astype(v.dtype); ss[col] = v
ss.to_parquet(ss_p)
cl = pd.DataFrame({'star_id': np.nonzero(clipped)[0], 'frame': np.nonzero(clipped)[1]})
cl['bjd_tdb'] = bjd[cl['frame'].values]
cl.to_csv(f'{lcdir}/{stem}_clipped_epochs.csv', index=False)
json.dump(dict(k=K, window_epochs=w, cadence_s=cad, n_valid=n_valid, n_clipped=n_clip, frac=n_clip / n_valid, stars_clipped=int(clipped.any(1).sum()),
               stars_used=int(use_star.sum()), protected_epochs_last=protected_total, protected_epochs_first=prot_first, outlier_candidates_first=out_first), open(f'{lcdir}/{stem}_clip_summary.json', 'w'))
print(f'wrote {npz_p}, {lcp_p} ({len(lcp)} rows, was {len(lcp)+int(bad.sum())}), {ss_p}, {stem}_clipped_epochs.csv')
