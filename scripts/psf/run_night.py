# ruff: noqa
"""usage: python run_night.py <TEL> <night> <nproc> [pass1_step=7] [max_frames]

Per frame: SExtractor + PSFEx (psf_work/psfex/<stem>.psf), then the forced PSF fit.
Pass 1 (every pass1_step-th frame): fit with companion search (psfphot.PEEL) and per-frame detection nuisance sources; the sources found
(companions added from the residual image + unmatched per-frame detections) are converted to sky coordinates, clustered across the
pass-1 frames and kept when present in >= 50% of them and not within 1.4" of a master source -> psf_work/static_extras.csv.
Pass 2 (all frames): fit with that static extra-source list (same sources in every frame, positions via the frame WCS), no per-frame
companion search -> psf_work/raw/<stem>.pkl.  (A per-frame companion search made the source model differ from frame to frame and
produced spurious dips in faint light curves.)

Environment: PSF_OUT (required) output dir; PSF_PROD (default /mnt/sto01/<TEL>/reduced/<night>) night dir holding <stem>_proc.fits;
PSF_FORCED (default $PSF_PROD/forced) forced catalogues + forced_reference.csv; PSF_CATDIR (default $PSF_PROD) <stem>_proc_catalog.csv.
Exit 0 and a final RUN_NIGHT_OK line only if every frame was fitted; otherwise FRAME_ERRORS / FRAME_COUNT_MISMATCH and exit 1."""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')    # one BLAS thread per fork worker (results identical to multi-threaded)
os.environ.setdefault('TMPDIR', '/ssdsto1/data/mnt')
import sys, time, subprocess, json, glob, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
import numpy as np, pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import psfphot
CODE = os.path.dirname(os.path.abspath(__file__))
CFG = {'ROBO43': dict(scale=0.5282, satur=40000), 'T80S': dict(scale=0.5545, satur=52000)}
tel, night, nproc = sys.argv[1], sys.argv[2], int(sys.argv[3])
step1 = int(sys.argv[4]) if len(sys.argv) > 4 else 7
maxf = int(sys.argv[5]) if len(sys.argv) > 5 else None
PROD = os.environ.get('PSF_PROD', f'/mnt/sto01/{tel}/reduced/{night}')
N = os.environ.get('PSF_OUT')
if not N:
    print('run_night.py: environment variable PSF_OUT (output directory) is required', file=sys.stderr)
    sys.exit(2)
FORCED = os.environ.get('PSF_FORCED', f'{PROD}/forced')
CATDIR = os.environ.get('PSF_CATDIR', PROD)
W = f'{N}/psf_work'
for d in ('psfex', 'raw', 'log', 'pass1'):
    os.makedirs(f'{W}/{d}', exist_ok=True)
STATIC = None

def dump_atomic(obj, path):
    with open(path + '.tmp', 'wb') as fh:
        pickle.dump(obj, fh)
    os.replace(path + '.tmp', path)

def make_psf(stem):
    cfg = CFG[tel]; psfp = f'{W}/psfex/{stem}.psf'
    if os.path.exists(psfp): return psfp, 0.0, 0.0, None
    f = f'{PROD}/{stem}_proc.fits'
    t0 = time.time()
    ldac = f'{W}/psfex/{stem}.ldac'
    subprocess.run(['source-extractor', f + '[0]', '-c', f'{CODE}/psf.sex', '-PARAMETERS_NAME', f'{CODE}/psf.param',
                    '-FILTER_NAME', f'{CODE}/default.conv', '-CATALOG_NAME', ldac, '-PIXEL_SCALE', str(cfg['scale']),
                    '-SATUR_LEVEL', str(cfg['satur'])], capture_output=True, cwd=f'{W}/psfex')
    t1 = time.time(); used = None
    for sn in (250, 400, 800):
        try:
            subprocess.run(['psfex', ldac, '-c', f'{CODE}/psfex.conf', '-PSF_DIR', f'{W}/psfex', '-SAMPLE_MINSN', str(sn)],
                           capture_output=True, timeout=120, cwd=f'{W}/psfex')
        except subprocess.TimeoutExpired:
            pass
        if os.path.exists(psfp) and os.path.getsize(psfp) > 0:
            used = sn; break
    if os.path.exists(ldac): os.remove(ldac)
    return (psfp if used else None), t1 - t0, time.time() - t1, used

def work(args):
    stem, passno = args
    t0 = time.time()
    try:
        psfp, tse, tpx, sn = make_psf(stem)
        if psfp is not None and not os.path.exists(psfp): psfp = None
        kw = dict()
        if passno == 1:
            psfphot.ISO_REF = 1.5
            kw = dict(peel=True, use_det=True)
        else:
            psfphot.ISO_REF = 0.8
            if STATIC is not None and len(STATIC):
                w = WCS(fits.getheader(f'{PROD}/{stem}_proc.fits', 0), relax=True)
                x, y = w.all_world2pix(STATIC['ra'].values, STATIC['dec'].values, 1)
                xy = np.c_[x - 1.0, y - 1.0]
            else:
                xy = np.zeros((0, 2))
            kw = dict(peel=False, use_det=False, extra_xy=xy)
        out, info = psfphot.fit_frame(f'{PROD}/{stem}_proc.fits', f'{FORCED}/{stem}_proc_forced_catalog.csv',
                                      f'{CATDIR}/{stem}_proc_catalog.csv', psfp, CFG[tel]['scale'], **kw)
        info.update(stem=stem, t_sex=tse, t_psfex=tpx, minsn=sn, ttotal=time.time() - t0, passno=passno)
        if passno == 1:
            dump_atomic((out, info), f'{W}/pass1/{stem}.pkl')
            info = {k: v for k, v in info.items() if k not in ('peel_xy', 'nuis_det_xy')}
        else:
            info = dict(info); info.pop('peel_xy', None); info.pop('nuis_det_xy', None)
            dump_atomic((out, info), f'{W}/raw/{stem}.pkl')
        return stem, info
    except Exception:
        import traceback
        return stem, dict(error=traceback.format_exc(), stem=stem, ttotal=time.time() - t0)

def build_static(p1stems):
    """cluster the pass-1 extra sources (peeled companions + unmatched per-frame detections) in sky coordinates"""
    allra, alldec, allf, allk = [], [], [], []
    usable, skipped = [], []
    for s in p1stems:
        pk = f'{W}/pass1/{s}.pkl'
        if not os.path.exists(pk):
            skipped.append((s, 'no pass-1 pickle')); continue
        out, info = pickle.load(open(pk, 'rb'))
        if 'peel_xy' not in info:
            skipped.append((s, 'no peel_xy (frame_fail/error)')); continue
        usable.append((s, info))
    for s, why in skipped:
        print(f'build_static: skipping pass-1 frame {s}: {why}', flush=True)
    if not usable:
        raise RuntimeError(f'build_static: no usable pass-1 frame ({len(skipped)} skipped)')
    p1stems = [s for s, _ in usable]
    for j, (s, info) in enumerate(usable):
        w = WCS(fits.getheader(f'{PROD}/{s}_proc.fits', 0), relax=True)
        for kind, arr in ((0, info['peel_xy'][:, :2]), (1, info['nuis_det_xy'])):
            if len(arr):
                ra, dec = w.all_pix2world(arr[:, 0] + 1.0, arr[:, 1] + 1.0, 1)
                allra.append(ra); alldec.append(dec); allf.append(np.full(len(ra), j)); allk.append(np.full(len(ra), kind))
    ra = np.concatenate(allra); dec = np.concatenate(alldec); fr = np.concatenate(allf); kd = np.concatenate(allk)
    ra0, dec0 = np.median(ra), np.median(dec)
    P = np.c_[(ra - ra0) * np.cos(np.radians(dec0)) * 3600.0, (dec - dec0) * 3600.0]
    t = cKDTree(P); pairs = t.query_pairs(0.5, output_type='ndarray')
    n = len(P)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    nc, lab = connected_components(g, directed=False)
    nfr = np.zeros(nc, int); 
    df = pd.DataFrame({'lab': lab, 'fr': fr, 'ra': ra, 'dec': dec, 'kd': kd})
    agg = df.groupby('lab').agg(nfr=('fr', 'nunique'), ra=('ra', 'median'), dec=('dec', 'median'), npeel=('kd', lambda v: int((v == 0).sum())), n=('kd', 'size'))
    keep = agg[agg.nfr >= int(np.ceil(0.5 * len(p1stems)))].copy()
    # drop sources coinciding with a master source (<= 1.4")
    M = pd.read_csv(f'{FORCED}/forced_reference.csv')
    tm = cKDTree(np.c_[(M.RA - ra0) * np.cos(np.radians(dec0)) * 3600.0, (M.DEC - dec0) * 3600.0])
    d, _ = tm.query(np.c_[(keep.ra - ra0) * np.cos(np.radians(dec0)) * 3600.0, (keep.dec - dec0) * 3600.0])
    keep['sep_master'] = d
    keep = keep[keep.sep_master > 1.4].reset_index(drop=True)
    keep.to_csv(f'{W}/static_extras.csv', index=False)
    print(f'static extra sources: {len(agg)} clusters from pass-1 sources, {int((agg.nfr >= np.ceil(0.5*len(p1stems))).sum())} in >=50% of {len(p1stems)} frames, {len(keep)} kept after removing master duplicates (<=1.4")', flush=True)
    return keep

if __name__ == '__main__':
    stems = sorted(os.path.basename(p)[:-len('_proc_forced_catalog.csv')] for p in glob.glob(f'{FORCED}/*_proc_forced_catalog.csv'))
    if maxf: stems = stems[:maxf]
    p1 = stems[::step1]
    t0 = time.time(); allinfo = []; err2 = []
    ctx = mp.get_context('fork')
    if not os.path.exists(f'{W}/static_extras.csv'):
        todo = [s for s in p1 if not os.path.exists(f'{W}/pass1/{s}.pkl')]
        print(tel, night, 'pass 1:', len(todo), 'of', len(p1), 'frames', flush=True)
        with ctx.Pool(min(nproc, max(len(todo), 1))) as pool:
            for stem, info in pool.imap_unordered(work, [(s, 1) for s in todo]):
                allinfo.append(info)
                if 'error' in info: print('ERROR pass 1', stem, info['error'], flush=True)
                print('P1', stem, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in info.items() if k != 'stem'}, flush=True)
        err1 = sorted(i.get('stem', '?') for i in allinfo if 'error' in i)
        if err1:
            print(f'FRAME_ERRORS pass 1 {len(err1)}: {" ".join(err1)}', flush=True)
            sys.exit(1)
        STATIC = build_static(p1)
    else:
        STATIC = pd.read_csv(f'{W}/static_extras.csv'); print('using existing static_extras.csv', len(STATIC))
    t1 = time.time()
    todo = [s for s in stems if not os.path.exists(f'{W}/raw/{s}.pkl')]
    print(tel, night, 'pass 2:', len(todo), 'frames to do', flush=True)
    with ctx.Pool(nproc) as pool:
        for stem, info in pool.imap_unordered(work, [(s, 2) for s in todo]):
            allinfo.append(info)
            if 'error' in info:
                err2.append(stem); print('ERROR pass 2', stem, info['error'], flush=True)
            print(stem, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in info.items() if k != 'stem'}, flush=True)
    print('total wall', time.time() - t0, 'pass1 wall', t1 - t0, 'pass2 wall', time.time() - t1, flush=True)
    json.dump(allinfo, open(f'{W}/log/timing_{int(t0)}.json', 'w'), default=str)
    nraw = len(glob.glob(f'{W}/raw/*.pkl'))
    nforced = len(glob.glob(f'{FORCED}/*_proc_forced_catalog.csv')) if not maxf else len(stems)
    bad = False
    if err2:
        print(f'FRAME_ERRORS {len(err2)}: {" ".join(sorted(err2))}', flush=True); bad = True
    if nraw != nforced:
        print(f'FRAME_COUNT_MISMATCH raw={nraw} forced={nforced}', flush=True); bad = True
    if bad:
        sys.exit(1)
    print(f'RUN_NIGHT_OK frames={nraw}', flush=True)
