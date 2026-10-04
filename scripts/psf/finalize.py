# ruff: noqa
"""usage: python finalize.py <TEL> <night> [rel_thr]  -- assign bit 64 (poor fit), write forced-format PSF catalogues + proc.fits symlinks, flag statistics.

Environment: PSF_OUT (required) output dir (also holds psf_work/ from run_night.py); PSF_PROD (default /mnt/sto01/<TEL>/reduced/<night>);
PSF_FORCED (default $PSF_PROD/forced) forced catalogues.  The <stem>_proc.fits links are relative (../<stem>_proc.fits): the PSF catalogues
live in <night>/psf/, the images in <night>/.  Exit 0 and a final FINALIZE_OK line on success."""
import sys, os, glob, pickle, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import psfphot as P
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
cnt = {b: 0 for b in (1, 2, 4, 8, 16, 32, 64, 128)}; ntot = 0; nfail_frames = 0; fail_stems = []
rows_all = []
for j, s in enumerate(stems):
    o, info = raw[s]
    F = pd.read_csv(f'{FORCED}/{s}_proc_forced_catalog.csv')
    assert (F['NUMBER'].values == o['NUMBER'].values).all()
    flags = o['flags'].values.astype(np.int64).copy()
    r = np.array([rel[nidx[int(v)], j] for v in o['NUMBER'].values])
    flags[np.isfinite(r) & (r > REL_THR)] |= P.F_POORFIT
    flux = o['flux'].values; ferr = o['ferr'].values
    bad = (flags & (P.F_FAIL | P.F_NONPOS)) != 0
    snr = np.where(np.isfinite(flux) & np.isfinite(ferr) & (ferr > 0), flux / ferr, np.nan)
    outt = pd.DataFrame({
        'NUMBER': F['NUMBER'], 'X_IMAGE': F['X_IMAGE'], 'Y_IMAGE': F['Y_IMAGE'], 'ALPHA_J2000': F['ALPHA_J2000'],
        'DELTA_J2000': F['DELTA_J2000'], 'RA': F['RA'], 'DEC': F['DEC'], 'FLAGS': flags.astype(np.int32),
        'SNR': snr, 'FWHM': F['FWHM'], 'BACKGROUND': F['BACKGROUND'],
        'FLUX_APER_1': flux, 'FLUXERR_APER_1': ferr,
        'CHI2_CORE': o['chi2c'].values, 'CHI2_STAMP': o['chi2s'].values, 'NSRC_FIT': o['nsrc'].values,
        'SKY_FIT': o['sky'].values, 'NPEEL': o['npeel'].values, 'DX_REFINE': o['dx'].values, 'DY_REFINE': o['dy'].values, 'FLAGS_APER': F['FLAGS'].values, 'FLUX_APER_PROD_8ARCSEC': F['FLUX_APER_5'].values,
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
json.dump(dict(frames=len(stems), frame_fail=nfail_frames, n_epochs=ntot, bits=cnt, rel_thr=REL_THR, chi_rel_sigma=sig), open(f'{N}/psf_flag_counts.json', 'w'))
with open(f'{N}/psf_flag_mapping.txt', 'w') as fh:
    fh.write('PSF photometry FLAGS bit mapping (relphot masks FLAGS & 252)\n')
    for b in cnt: fh.write(f'{b:3d}: {P.FLAG_DOC[b]}   [{cnt[b]} / {ntot} source-epochs]\n')
print(f'FINALIZE_OK frames={len(stems)} frame_fail={nfail_frames}', flush=True)
