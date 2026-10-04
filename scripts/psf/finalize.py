# ruff: noqa
"""usage: python finalize.py <TEL> <night> [rel_thr]  -- assign bit 64 (poor fit), write forced-format PSF catalogues + proc.fits symlinks, flag statistics.

Environment: PSF_OUT (required) output dir (also holds psf_work/ from run_night.py); PSF_PROD (default /mnt/sto01/<TEL>/reduced/<night>);
PSF_FORCED (default $PSF_PROD/forced) forced catalogues; PSF_FAINTZP=0 switches the faint-star zero-point correction off.
Faint-star zero point: the PSF flux of faint / blended sources depends on the frame's PSF width (up to 25-40 % at 1500-4000 counts).  A per-frame,
common-mode correction a_j * h(cell_i) (cell = night-median SNR bin x night-median CHI2_CORE class, h estimated night-wide, rank-1) is applied to
FLUX_APER_1 / FLUXERR_APER_1; raw flux in FLUX_PSF_RAW, applied correction in FZP_MAG [mag], frame amplitude in FZP_A, per-night table psf_faint_zp.csv.
  The <stem>_proc.fits links are relative (../<stem>_proc.fits): the PSF catalogues
live in <night>/psf/, the images in <night>/.  Exit 0 and a final FINALIZE_OK line on success."""
import sys, os, glob, pickle, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import psfphot as P
FZP_ON = os.environ.get('PSF_FAINTZP', '1') != '0'
FZP_MINFR = 0.5        # a star enters the fit if it is good (FLAGS & 252 == 0, flux > 0) in >= this fraction of the frames (and >= 5)
FZP_SNRMIN = 2.0       # ... and has night-median SNR >= this
FZP_UQ = (.01, .03, .06, .12, .25, .45, .70)   # candidate SNR-bin edges as star-count quantiles of log10(SNR)
FZP_NV = 3             # chi2 classes (quantiles of the star's night-median CHI2_CORE) inside each SNR bin
FZP_NBMIN = 30         # min fit stars per cell (an SNR bin needs 3x this to exist; the number of chi2 classes is reduced to fit)
FZP_SFLOOR = 0.005     # mag, systematic floor added to the cell-median errors (keeps bright cells from dominating the rank-1 fit)
FZP_NITER = 20
FZP_NMIN = 40          # min stars (in cells with |h| >= 0.25) in a frame, else that frame is left uncorrected
FZP_AMAX = 1.0         # mag, |a_j| above this = failed fit -> frame left uncorrected
FZP_HMIN = 0.03        # cells with |h| < this (fraction of the largest cell response) are left uncorrected: ordinary bright stars stay untouched
FZP_NSIG = 3.0         # whole night left uncorrected when std(a_j) < NSIG * median(sigma_a_j) (no significant frame-to-frame faint-star bias)


def faint_zp(fx, fe, fg, ch):
    """fx, fe, fg, ch: (nstar, nframe) PSF flux, flux error, FLAGS (bit 64 included), CHI2_CORE (NaN where missing).
    Model: dm_ij = zp_j + a_j h(cell_i),  dm_ij = -2.5 log10(f_ij / median_j f_ij);  cell = (night-median SNR bin) x (night-median CHI2_CORE class).
    h: night-wide rank-1 factor model on the per-frame cell medians (reference cell = brightest SNR bin, lowest chi2 class: h = 0, defines zp_j);
    a_j: per-frame amplitude, centred on the median frame.  Returns (a, sa, used, cell, h, info): the correction of source i in frame j is
    a[j] * h[cell[i]] mag (cell -1: none), corrected flux = f * 10**(0.4 a_j h).  info['status'] != 'ok': nothing is corrected."""
    n, m = fx.shape
    a = np.zeros(m); sa = np.full(m, np.nan); used = np.zeros(m, int); cell = np.full(n, -1); h = np.zeros(1)
    good = np.isfinite(fx) & (fx > 0) & np.isfinite(fe) & (fe > 0) & ((fg & 252) == 0)
    info = dict(status='ok', n_fit=0, n_cell=0, n_bad=0, bad=np.zeros(m, bool))
    if (good.sum(0) > 0).sum() < 5: info['status'] = 'few_frames'; return a, sa, used, cell, h, info
    F = np.where(good, fx, np.nan); ng = good.sum(1)
    ref = np.nanmedian(F, axis=1)
    snr = ref / np.nanmedian(np.where(good, fe, np.nan), axis=1)
    cm = np.nanmedian(np.where(good, ch, np.nan), axis=1)
    ok = (ng >= max(5, FZP_MINFR * m)) & np.isfinite(snr) & (snr > 0) & np.isfinite(cm)
    S = np.where(ok & (snr >= FZP_SNRMIN))[0]
    if len(S) < 200: info['status'] = 'few_stars'; return a, sa, used, cell, h, info
    u = np.log10(np.where(ok, snr, np.nan)); us = u[S]
    ue = []
    for q in np.quantile(us, FZP_UQ):
        lo = ue[-1] if ue else -np.inf
        if ((us > lo) & (us <= q)).sum() >= 3 * FZP_NBMIN and (us > q).sum() >= 3 * FZP_NBMIN: ue.append(q)
    ub = np.searchsorted(np.array(ue), u, side='left')
    nc = 0; first = []
    for b in range(len(ue) + 1):
        sb = S[ub[S] == b]; k = int(max(1, min(FZP_NV, len(sb) // FZP_NBMIN)))
        qe = np.quantile(cm[sb], np.arange(1, k) / k) if k > 1 else np.zeros(0)
        sel = ok & (ub == b); cell[sel] = nc + np.searchsorted(qe, cm[sel], side='right'); first.append(nc); nc += k
    info.update(n_fit=int(len(S)), n_cell=int(nc))
    if nc < 4: info['status'] = 'few_cells'; cell[:] = -1; return a, sa, used, cell, h, info
    cs = cell[S]; Y = -2.5 * np.log10(F[S] / ref[S, None]); V = np.isfinite(Y)
    M = np.full((m, nc), np.nan); E = np.full((m, nc), np.nan)
    for c in range(nc):
        Yb = Y[cs == c]; med = np.nanmedian(Yb, axis=0); M[:, c] = med
        E[:, c] = 1.2533 * 1.4826 * np.nanmedian(np.abs(Yb - med[None, :]), axis=0) / np.sqrt(np.maximum(np.isfinite(Yb).sum(0), 1))
    cref = first[-1]
    zp = np.where(np.isfinite(M[:, cref]), M[:, cref], np.nanmedian(M[:, cref]))
    cols = [c for c in range(nc) if c != cref]
    R = M[:, cols] - zp[:, None]; Ec = E[:, cols]
    Wm = np.where(np.isfinite(R) & np.isfinite(Ec) & (Ec > 0), 1.0 / (Ec ** 2 + FZP_SFLOOR ** 2), 0.0); R = np.where(Wm > 0, R, 0.0)
    af = R[:, np.argmax((Wm * R * R).sum(0))].copy()
    for _ in range(FZP_NITER):
        hc = (Wm * af[:, None] * R).sum(0) / np.maximum((Wm * af[:, None] ** 2).sum(0), 1e-30)
        af = (Wm * hc[None, :] * R).sum(1) / np.maximum((Wm * hc[None, :] ** 2).sum(1), 1e-30)
        sc = np.abs(hc).max()
        if not sc > 0: info['status'] = 'no_signal'; cell[:] = -1; return a, sa, used, cell, h, info
        hc /= sc; af *= sc
    fr = (Wm > 0).any(1); af = af - np.median(af[fr])
    res = R - af[:, None] * hc[None, :]
    sa = np.sqrt(np.maximum((Wm * res ** 2).sum(1) / np.maximum((Wm > 0).sum(1) - 1, 1), 1.0)) / np.sqrt(np.maximum((Wm * hc[None, :] ** 2).sum(1), 1e-30))
    if np.std(af[fr]) < FZP_NSIG * np.median(sa[fr]): info['status'] = 'insignificant'; cell[:] = -1; return a, sa, used, cell, h, info
    h = np.zeros(nc); h[cols] = np.where(np.abs(hc) < FZP_HMIN, 0.0, hc)
    used = (V & (np.abs(h) >= 0.25)[cs][:, None]).sum(0)
    bad = (used < FZP_NMIN) | ~np.isfinite(af) | (np.abs(af) > FZP_AMAX) | ~np.isfinite(sa)
    a = np.where(bad, 0.0, af); info.update(n_bad=int(bad.sum()), bad=bad)
    return a, sa, used, cell, h, info

tel, night = sys.argv[1], sys.argv[2]
REL_THR = sys.argv[3] if len(sys.argv) > 3 else 'auto'   # 'auto': 1 + 6 robust sigma of the relative core chi2 (night-wide); or a number
PROD = os.environ.get('PSF_PROD', f'/mnt/sto01/{tel}/reduced/{night}')
N = os.environ.get('PSF_OUT')
if not N:
    print('finalize.py: environment variable PSF_OUT (output directory) is required', file=sys.stderr)
    sys.exit(2)
FORCED = os.environ.get('PSF_FORCED', f'{PROD}/forced')
W = f'{N}/psf_work'
stems = sorted(os.path.basename(p)[:-len('.pkl')] for p in glob.glob(f'{W}/raw/*.pkl'))
nforced = len(glob.glob(f'{FORCED}/*_proc_forced_catalog.csv'))
if not stems or len(stems) != nforced:
    print(f'FRAME_COUNT_MISMATCH raw={len(stems)} forced={nforced}', flush=True)
    sys.exit(1)
raw = {s: pickle.load(open(f'{W}/raw/{s}.pkl', 'rb')) for s in stems}
nrow = {s: len(raw[s][0]) for s in stems}
# union of master rows: NUMBER-indexed matrix of core chi2
numbers = np.unique(np.concatenate([raw[s][0]['NUMBER'].values for s in stems]))
nidx = {int(v): i for i, v in enumerate(numbers)}
chi = np.full((len(numbers), len(stems)), np.nan)
for j, s in enumerate(stems):
    o = raw[s][0]
    ok = (o['flags'].values & (P.F_FAIL | P.F_NONPOS)) == 0
    chi[[nidx[int(v)] for v in o['NUMBER'].values[ok]], j] = o['chi2c'].values[ok]
msw = np.nanmedian(chi, axis=1)
rel0 = chi / msw[:, None]
fframe = np.nanmedian(rel0, axis=0)              # frame-level chi2 factor
rel = rel0 / fframe[None, :]
fin = np.isfinite(rel)
med = np.nanmedian(rel[fin]); sig = 1.4826 * np.nanmedian(np.abs(rel[fin] - med))
REL_THR = (1.0 + 6.0 * 0) if False else REL_THR
print(f'chi2_core relative: median {med:.3f}  MAD-sigma {sig:.3f}  quantiles 50/90/99/99.9 {np.nanquantile(rel[fin],[.5,.9,.99,.999]).round(3)}  frame factor range {np.nanmin(fframe):.2f}-{np.nanmax(fframe):.2f}')
REL_THR = max(1.5, 1.0 + 6.0 * sig) if REL_THR == 'auto' else float(REL_THR)
print(f'threshold {REL_THR:.3f}: fraction of finite epochs above = {(rel[fin] > REL_THR).mean():.4f}; ')
# per-frame flags (bit 64 included), then the faint-star zero-point correction (needs the final flags: relphot masks FLAGS & 252)
flg = {}; rix = {}
for j, s in enumerate(stems):
    o = raw[s][0]
    rix[s] = np.array([nidx[int(v)] for v in o['NUMBER'].values])
    fg = o['flags'].values.astype(np.int64).copy()
    r = rel[rix[s], j]
    fg[np.isfinite(r) & (r > REL_THR)] |= P.F_POORFIT
    flg[s] = fg
t_fz = time.time()
if FZP_ON:
    FX = np.full((len(numbers), len(stems)), np.nan); FE = np.full_like(FX, np.nan); CH = np.full_like(FX, np.nan); FG = np.full(FX.shape, 255, np.int64)
    for j, s in enumerate(stems):
        o = raw[s][0]; ix = rix[s]
        FX[ix, j] = o['flux'].values; FE[ix, j] = o['ferr'].values; CH[ix, j] = o['chi2c'].values; FG[ix, j] = flg[s]
    zA, zS, zU, zcell, zh, zinfo = faint_zp(FX, FE, FG, CH)
    del FX, FE, CH, FG
else:
    zA = np.zeros(len(stems)); zS = np.full(len(stems), np.nan); zU = np.zeros(len(stems), int); zcell = np.full(len(numbers), -1); zh = np.zeros(1)
    zinfo = dict(status='disabled', n_fit=0, n_cell=0, n_bad=0, bad=np.zeros(len(stems), bool))
zhs = np.where(zcell >= 0, zh[np.clip(zcell, 0, len(zh) - 1)], 0.0)      # response of every master row, mag per unit a_j
zfb = np.ones(len(stems), int) if zinfo['status'] != 'ok' else zinfo['bad'].astype(int)
pfw = np.array([raw[s][1].get('psf_fwhm', np.nan) for s in stems], float)
pd.DataFrame({'stem': stems, 'psf_fwhm': pfw, 'faintzp_a': zA, 'faintzp_sigma': zS, 'n_stars_used': zU, 'fallback': zfb, 'status': zinfo['status']}).to_csv(f'{N}/psf_faint_zp.csv', index=False, float_format='%.6g')
zv = np.isfinite(pfw) & (zfb == 0)
print(f"faint_zp: status {zinfo['status']}  cells {zinfo['n_cell']}  fit_stars {zinfo['n_fit']}  uncorrected_frames {int(zfb.sum())}/{len(stems)}  a_j min/max/std {zA.min():+.3f}/{zA.max():+.3f}/{zA.std():.3f} mag  "
      f"corr(a,psf_fwhm) {(np.corrcoef(zA[zv], pfw[zv])[0, 1] if zv.sum() > 3 and zA[zv].std() > 0 else np.nan):+.3f}  sigma_a median {np.nanmedian(zS) * 1e3:.1f} mmag  {time.time() - t_fz:.1f} s", flush=True)
cnt = {b: 0 for b in (1, 2, 4, 8, 16, 32, 64, 128)}; ntot = 0; nfail_frames = 0; fail_stems = []
rows_all = []
for j, s in enumerate(stems):
    o, info = raw[s]
    F = pd.read_csv(f'{FORCED}/{s}_proc_forced_catalog.csv')
    assert (F['NUMBER'].values == o['NUMBER'].values).all()
    flags = flg[s]
    flux = o['flux'].values; ferr = o['ferr'].values
    zc = zA[j] * zhs[rix[s]]; zg = 10.0 ** (0.4 * zc)                      # faint-star zero point: correction in mag, flux factor
    bad = (flags & (P.F_FAIL | P.F_NONPOS)) != 0
    snr = np.where(np.isfinite(flux) & np.isfinite(ferr) & (ferr > 0), flux / ferr, np.nan)
    outt = pd.DataFrame({
        'NUMBER': F['NUMBER'], 'X_IMAGE': F['X_IMAGE'], 'Y_IMAGE': F['Y_IMAGE'], 'ALPHA_J2000': F['ALPHA_J2000'],
        'DELTA_J2000': F['DELTA_J2000'], 'RA': F['RA'], 'DEC': F['DEC'], 'FLAGS': flags.astype(np.int32),
        'SNR': snr, 'FWHM': F['FWHM'], 'BACKGROUND': F['BACKGROUND'],
        'FLUX_APER_1': flux * zg, 'FLUXERR_APER_1': ferr * zg,
        'CHI2_CORE': o['chi2c'].values, 'CHI2_STAMP': o['chi2s'].values, 'NSRC_FIT': o['nsrc'].values,
        'SKY_FIT': o['sky'].values, 'NPEEL': o['npeel'].values, 'DX_REFINE': o['dx'].values, 'DY_REFINE': o['dy'].values, 'FLAGS_APER': F['FLAGS'].values, 'FLUX_APER_PROD_8ARCSEC': F['FLUX_APER_5'].values,
        'FLUX_PSF_RAW': flux, 'FZP_MAG': zc, 'FZP_A': zA[j],
    })
    outt.to_csv(f'{N}/{s}_proc_forced_catalog.csv', index=False, float_format='%.8g')
    ln = f'{N}/{s}_proc.fits'; tgt = f'../{s}_proc.fits'
    if os.path.lexists(ln):
        if not os.path.islink(ln):
            print(f'ERROR: {ln} exists and is not a symlink', flush=True); sys.exit(1)
        if os.readlink(ln) != tgt:
            os.remove(ln); os.symlink(tgt, ln)
    else:
        os.symlink(tgt, ln)
    ntot += len(flags)
    if info.get('frame_fail'): nfail_frames += 1; fail_stems.append(s)
    for b in cnt: cnt[b] += int(((flags & b) != 0).sum())
nbad = {}
print(f'{len(stems)} frames, {nfail_frames} frame-level failures, {ntot} source-epochs')
for s in fail_stems: print(f'frame_fail: {s}')
print('per-bit counts (source-epochs, fraction):')
for b in cnt: print(f'  {b:3d} {P.FLAG_DOC[b][:60]:60s} {cnt[b]:9d} {cnt[b]/ntot:.4f}')
json.dump(dict(frames=len(stems), frame_fail=nfail_frames, n_epochs=ntot, bits=cnt, rel_thr=REL_THR, chi_rel_sigma=sig, faint_zp=dict(status=zinfo['status'], n_cell=zinfo['n_cell'], n_fit=zinfo['n_fit'], uncorrected_frames=int(zfb.sum()), a_std=float(zA.std()))), open(f'{N}/psf_flag_counts.json', 'w'))
with open(f'{N}/psf_flag_mapping.txt', 'w') as fh:
    fh.write('PSF photometry FLAGS bit mapping (relphot masks FLAGS & 252)\n')
    for b in cnt: fh.write(f'{b:3d}: {P.FLAG_DOC[b]}   [{cnt[b]} / {ntot} source-epochs]\n')
print(f'FINALIZE_OK frames={len(stems)} frame_fail={nfail_frames}', flush=True)
