# ruff: noqa
"""Forced PSF photometry at fixed (production forced-catalogue) positions, PSFEx model, joint stamp fits.

Per source: a (2h+1)^2 stamp is fitted by weighted linear least squares with the PSFEx model of that position
(cubic B-spline interpolation) rendered at the catalogue position of the primary and of every neighbouring source (master rows +
per-frame detections not in the master) + a local background plane (3 parameters).  Only the flux of the primary is kept.
Position handling: a smooth per-frame shift field between the PSFEx model frame and the catalogue positions is calibrated on bright
isolated stars (spread over the detector, best-chi2 half, polynomial of degree 1-3 chosen by 5-fold CV); stars with flux/err > SNR_REF and no neighbour within ISO_REF*FWHM get a linearised
centroid refinement (|shift| <= MAXSHIFT_REF px, else the catalogue position is kept); stamps with a poor core fit get up to
NPEEL extra sources from the smoothed residual image (companions merged in the master catalogue), fitted jointly with the
primary.  Pixel weights: ERR plane first, then model-based Poisson variance.  FLAGS bit meaning: see FLAG_DOC.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import cKDTree
from scipy import ndimage as ndi

warnings.simplefilter('ignore')

# DQ bits (robo43.products.dq.DQ)
DQ_HOT, DQ_DEAD, DQ_UNSTABLE, DQ_SAT, DQ_CR, DQ_NODATA, DQ_NONLIN, DQ_BADCOL = 1, 2, 4, 8, 16, 32, 64, 128
DQ_SATNL = DQ_SAT | DQ_NONLIN
DQ_HARDBAD = DQ_HOT | DQ_DEAD | DQ_UNSTABLE | DQ_NODATA | DQ_BADCOL      # static / no-data defects
DQ_OTHERBAD = DQ_HARDBAD | DQ_CR                                         # cosmic-ray (repaired) pixels are masked in the fit but not flagged
DQ_FITMASK = DQ_SATNL | DQ_OTHERBAD

# FLAGS bits of the PSF catalogues
F_NEIGHBOUR, F_GROUP, F_SATNL, F_EDGE, F_BADPIX, F_FAIL, F_POORFIT, F_NONPOS = 1, 2, 4, 8, 16, 32, 64, 128
FLAG_DOC = {
    1: 'another source (master row or unmatched per-frame detection) within 4.0 arcsec (production flag_radius)',
    2: 'fitted as a blend: another source within 1.4 arcsec (master row or per-frame detection), or a companion added from the residual image of the stamp',
    4: 'saturated or non-linear DQ pixel within 4.0 arcsec of the source (pixels also excluded from the fit)',
    8: 'source centre within 20 px of the detector edge (production edge_margin_px; stamp truncated)',
    16: 'hot/dead/unstable/no-data/bad-column pixel within 1 PSF FWHM of the source, or >20% of the stamp masked in the fit (incl. repaired cosmic-ray pixels)',
    32: 'fit failed (frame without PSF model, <50% usable stamp pixels, singular normal matrix, non-finite result)',
    64: 'poor fit: core chi2 above threshold relative to the star\'s own median and the frame (assigned in finalize)',
    128: 'flux non-finite or <= 0 (or non-finite/<=0 error)',
}


class PSF:
    def __init__(self, path, order=0):
        h = fits.open(path)
        hd = h[1].header
        self.mask = np.array(h[1].data['PSF_MASK'][0], dtype=np.float64)
        self.samp = float(hd['PSF_SAMP'])
        assert abs(self.samp - 1.0) < 1e-6, 'only PSF_SAMP=1 supported'
        self.z = (float(hd['POLZERO1']), float(hd['POLZERO2']))
        self.s = (float(hd['POLSCAL1']), float(hd['POLSCAL2']))
        self.deg = int(hd['POLDEG1'])
        self.fwhm = float(hd['PSF_FWHM'])
        self.chi2 = float(hd['CHI2']); self.nacc = int(hd['ACCEPTED']); self.nload = int(hd['LOADED'])
        self.nb, self.ny, self.nx = self.mask.shape
        if order == 0:   # PSFEx monomial order for 2 variables: 1, x, x^2, y, xy, y^2
            self.exps = [(dx, dy) for dy in range(self.deg + 1) for dx in range(self.deg + 1 - dy)]
        else:            # by total degree: 1, x, y, x^2, xy, y^2
            self.exps = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
        assert len(self.exps) == self.nb
        self.c = (self.nx - 1) / 2.0

    def image(self, x1, y1):
        """PSF image (ny,nx), unit sum, at FITS-1-based pixel position (x1,y1)."""
        xn = (x1 - self.z[0]) / self.s[0]; yn = (y1 - self.z[1]) / self.s[1]
        b = np.array([xn ** i * yn ** j for (i, j) in self.exps])
        P = np.tensordot(b, self.mask, axes=(0, 0))
        return P / P.sum()


def cubic_w(u, n):
    """Cubic-convolution (Keys a=-0.5) weights, u: (...,S) sample coordinates -> (...,S,n); partition of unity."""
    t = np.abs(u[..., None] - np.arange(n))
    t2 = t * t; t3 = t2 * t
    w = np.where(t < 1, 1.5 * t3 - 2.5 * t2 + 1.0, np.where(t < 2, -0.5 * t3 + 2.5 * t2 - 4.0 * t + 2.0, 0.0))
    return w


REWEIGHT = True
REFINE = True      # per-star linearised centroid refinement
SNR_REF = 20.0     # ... for stars with flux/err above this
KREF = 6           # ... and at most this many fitted sources in the stamp
NIT_REF = 4        # Gauss-Newton iterations
MAXSHIFT_REF = 1.2 # px; larger shifts are rejected (position reverts to the catalogue position)
ISO_REF = 1.5      # refine the primary only if no other fitted source within ISO_REF * PSF FWHM
PEEL = True        # add companions found in the residual image
NPEEL = 3
PEEL_CHI = 3.0     # only when the core chi2 of the stamp exceeds this
PEEL_SIG = 6.0     # smoothed residual peak significance
PEEL_DMIN = 0.8    # companion at least PEEL_DMIN*FWHM from the primary
PEEL_DCHI2 = 40.0  # required chi2 improvement
INTERP = 'bspline' # 'bspline' (exact cubic B-spline interpolation of the sampled PSF) or 'keys' (cubic convolution)

def bspline_w(u, n):
    """Cubic B-spline weights on the coefficient grid (prefiltered PSF), partition of unity; (...,S) -> (...,S,n)."""
    i0 = np.floor(u)
    t = u - i0
    k = np.arange(n)
    d = k - (i0[..., None] - 1)          # 0..3 taps for k = i0-1..i0+2
    tt = t[..., None]
    w0 = (1 - tt) ** 3 / 6.0
    w1 = (3 * tt ** 3 - 6 * tt ** 2 + 4) / 6.0
    w2 = (-3 * tt ** 3 + 3 * tt ** 2 + 3 * tt + 1) / 6.0
    w3 = tt ** 3 / 6.0
    return np.where(d == 0, w0, np.where(d == 1, w1, np.where(d == 2, w2, np.where(d == 3, w3, 0.0))))




class StampFit:
    """Linear stamp model: K PSF sources (fluxes free) + background plane; positions are changed through set_pos/add."""
    def __init__(self, d, e, good, xp, yp, P, psf, wf, hs, xs, ys):
        self.d = d; self.Sy, self.Sx = d.shape
        self.gm = good.ravel()
        self.dflat = d.ravel()[self.gm]; self.eflat = e.ravel()[self.gm]
        gx, gy = np.meshgrid(xp, yp)
        self.base = np.concatenate([np.ones((1, d.size)), ((gx - xs) / hs).reshape(1, -1), ((gy - ys) / hs).reshape(1, -1)], 0)
        self.bgm = self.base[:, self.gm]
        self.xp = xp; self.yp = yp; self.P = P; self.psf = psf; self.wf = wf
        self.wgt = 1.0 / self.eflat

    def render(self, pxs, pys):
        Wx = self.wf(self.xp[None, :] - pxs[:, None] + self.psf.c, self.psf.nx); Wy = self.wf(self.yp[None, :] - pys[:, None] + self.psf.c, self.psf.ny)
        return np.matmul(np.matmul(Wy, self.P), Wx.transpose(0, 2, 1))   # (k,Sy,Sx)

    def set_sources(self, px, py):
        self.px = np.array(px, dtype=float); self.py = np.array(py, dtype=float)
        self.M = self.render(self.px, self.py); self.Mg = self.M.reshape(len(self.px), -1)[:, self.gm]

    def add(self, x, y):
        m = self.render(np.array([x]), np.array([y]))
        self.px = np.append(self.px, x); self.py = np.append(self.py, y)
        self.M = np.concatenate([self.M, m], 0); self.Mg = np.concatenate([self.Mg, m.reshape(1, -1)[:, self.gm]], 0)

    def drop_last(self):
        self.px = self.px[:-1]; self.py = self.py[:-1]; self.M = self.M[:-1]; self.Mg = self.Mg[:-1]

    def set_pos(self, k, x, y):
        self.px[k] = x; self.py[k] = y
        m = self.render(np.array([x]), np.array([y]))[0]
        self.M[k] = m; self.Mg[k] = m.ravel()[self.gm]

    def solve(self):
        self.K = len(self.px)
        self.Aw = np.concatenate([self.Mg, self.bgm], 0).T * self.wgt[:, None]
        ata = self.Aw.T @ self.Aw
        ata[np.diag_indices_from(ata)] *= (1.0 + 1e-9)
        self.cov = np.linalg.inv(ata)
        self.sol = self.cov @ (self.Aw.T @ (self.dflat * self.wgt))

    def resid_w(self):
        return self.dflat * self.wgt - self.Aw @ self.sol

    def chi2(self):
        return float((self.resid_w() ** 2).sum())

    def model_full(self):
        return (self.sol[:self.K] @ self.M.reshape(self.K, -1) + self.sol[self.K:] @ self.base).reshape(self.Sy, self.Sx)

    def reweight(self):
        model_src = self.sol[:self.K] @ self.Mg; sky = self.sol[self.K:] @ self.bgm
        var_n = np.maximum(self.eflat ** 2 - np.maximum(self.dflat - sky, 0.0), 1e-6)
        self.wgt = 1.0 / np.sqrt(var_n + np.maximum(model_src, 0.0))
        self.solve()

    def refine(self, mobile, nit, maxshift, dl=0.05):
        """Gauss-Newton position refinement of the sources `mobile`; a source that ends up > maxshift from its start reverts."""
        mobile = list(mobile)
        x0 = self.px[mobile].copy(); y0 = self.py[mobile].copy(); tot = np.zeros((len(mobile), 2))
        for _ in range(nit):
            cols = []
            for k in mobile:
                mx = self.render(np.array([self.px[k] + dl, self.px[k] - dl]), np.array([self.py[k], self.py[k]]))
                my = self.render(np.array([self.px[k], self.px[k]]), np.array([self.py[k] + dl, self.py[k] - dl]))
                f = self.sol[k]
                cols.append(f * ((mx[0] - mx[1]) / (2 * dl)).ravel()[self.gm]); cols.append(f * ((my[0] - my[1]) / (2 * dl)).ravel()[self.gm])
            Aug = np.concatenate([self.Mg, self.bgm, np.array(cols)], 0).T * self.wgt[:, None]
            ata = Aug.T @ Aug; ata[np.diag_indices_from(ata)] *= (1.0 + 1e-9)
            sa = np.linalg.solve(ata, Aug.T @ (self.dflat * self.wgt))
            st = sa[-2 * len(mobile):].reshape(-1, 2)
            if not np.isfinite(st).all():
                tot[:] = 1e9; break
            nrm = np.hypot(st[:, 0], st[:, 1]); st = st * np.where(nrm > 0.5, 0.5 / np.maximum(nrm, 1e-12), 1.0)[:, None]
            tot += st
            for j, k in enumerate(mobile):
                self.set_pos(k, x0[j] + tot[j, 0], y0[j] + tot[j, 1])
            self.solve()
            if nrm.max() < 0.02: break
        bad = np.hypot(tot[:, 0], tot[:, 1]) > maxshift
        for j, k in enumerate(mobile):
            if bad[j]:
                self.set_pos(k, x0[j], y0[j]); tot[j] = 0.0
        if bad.any(): self.solve()
        return tot

CAL_GRID = 10      # shift calibration: stars stratified over a CAL_GRID x CAL_GRID grid of the detector, brightest first in each cell ...
CAL_NMAX = 400     # ... at most this many candidates in all (cell rank first, so the truncation keeps the sampling even)
CAL_MAXS = 12.0    # px; fitted free shifts beyond this are dropped
CAL_NSEED = 40     # the CAL_NSEED candidates nearest the detector centre are searched on a coarse grid over +-6 px around 0, then Nelder-Mead ...
CAL_KNN = 6        # ... every further one (outwards) on a +-4.5 px grid around the median shift of its CAL_KNN nearest fitted stars (the field grows to the corners)
CAL_KEEP = 0.5     # fraction of the candidates (lowest chi2 at the best shift) kept: blends / extended objects bias the centroid
CAL_DEGS = (1, 2, 3)   # polynomial degrees of the shift field tried, chosen by 5-fold CV of the median residual
CAL_NMIN = (8, 24, 45) # fewest kept stars for each of CAL_DEGS (the higher degrees also need >= 7 of the 3x3 detector cells occupied)


def _pbasis(u, v, deg):
    return np.stack([u ** (t - j) * v ** j for t in range(deg + 1) for j in range(t + 1)], 1)


def _shift_eval(sm, xy):
    """Shift field (n,2) at pixel positions xy (n,2); arguments are clamped to the box of the calibration stars (no extrapolation)."""
    xy = np.atleast_2d(xy); b = sm['box']; hx, hy = sm['nx'] / 2.0, sm['ny'] / 2.0
    x = np.clip(xy[:, 0], b[0], b[1]); y = np.clip(xy[:, 1], b[2], b[3])
    return _pbasis((x - hx) / hx, (y - hy) / hy, sm['deg']) @ sm['coef'].T


def _clipfit(B, yv, nit=5, k=3.0):
    ok = np.ones(len(yv), bool)
    for _ in range(nit):
        c = np.linalg.lstsq(B[ok], yv[ok], rcond=None)[0]; r = yv - B @ c
        sd = 1.4826 * np.median(np.abs(r[ok] - np.median(r[ok]))) + 0.05
        ok2 = np.abs(r) < k * sd
        if ok2.sum() < B.shape[1] + 2: break
        ok = ok2
    return c, ok, sd


def _calibrate_shift(psf, sci, err, dq, X, Y, F, flags, iso, hs, ok, interp, nstar=CAL_NMAX, affine=True):
    """Smooth per-frame shift field (dx,dy)(x,y) between the PSFEx model frame and the catalogue positions, from free-shift fits of
    bright isolated unsaturated stars (stratified over the detector, searched from the centre outwards, shifts up to CAL_MAXS px): the star sits at catalogue position +
    (dx,dy) w.r.t. the model.  The stars with the best fit chi2 (CAL_KEEP) are fitted with a 3-sigma-clipped polynomial in
    ((x-nx/2)/(nx/2), (y-ny/2)/(ny/2)) of degree 1 (affine) to 3, the degree being chosen by 5-fold CV (a higher degree must lower the
    median CV residual by > 2%).  Returns (model dict for _shift_eval, n stars fitted, mad, diagnostics dict)."""
    from scipy.optimize import minimize
    ny, nx = sci.shape
    sm0 = dict(deg=0, coef=np.zeros((2, 1)), box=(-1e9, 1e9, -1e9, 1e9), nx=nx, ny=ny)
    snr = F['SNR'].values; fa = np.nan_to_num(F['FLUX_APER_5'].values, nan=0.0)
    cand = np.nonzero(ok & (snr > 100) & iso & ((flags & (F_SATNL | F_EDGE)) == 0) & ((F['FLAGS'].values & 0xFC) == 0))[0]
    dg = dict(shift_deg=0, shift_nuse=0, shift_max=0.0)
    if len(cand) < 8:
        return sm0, 0, np.nan, dg
    cand = cand[np.argsort(-fa[cand], kind='stable')][3:]
    cell = np.clip((X[cand] / nx * CAL_GRID).astype(int), 0, CAL_GRID - 1) * CAL_GRID + np.clip((Y[cand] / ny * CAL_GRID).astype(int), 0, CAL_GRID - 1)
    rank = np.zeros(len(cand), int)
    for c_ in np.unique(cell):
        m_ = np.nonzero(cell == c_)[0]; rank[m_] = np.arange(len(m_))
    cand = cand[np.lexsort((np.arange(len(cand)), rank))][:nstar]       # cell rank first (even sampling), flux order within a rank
    use_b = (interp or INTERP) == 'bspline'
    wf = bspline_w if use_b else cubic_w
    cand = cand[np.argsort(np.hypot(X[cand] - nx / 2.0, Y[cand] - ny / 2.0), kind='stable')]
    offs0, offs1 = np.arange(-6.0, 6.1, 1.5), np.arange(-4.5, 4.6, 1.5)
    res = []
    for i in cand:
        xs, ys = X[i], Y[i]; xi, yi = int(round(xs)), int(round(ys))
        if xi - hs < 0 or yi - hs < 0 or xi + hs + 1 > nx or yi + hs + 1 > ny: continue
        d = sci[yi - hs:yi + hs + 1, xi - hs:xi + hs + 1].astype(np.float64); e = err[yi - hs:yi + hs + 1, xi - hs:xi + hs + 1].astype(np.float64)
        q = dq[yi - hs:yi + hs + 1, xi - hs:xi + hs + 1]
        if ((q & DQ_FITMASK) != 0).any() or not (e > 0).all(): continue
        Pm = psf.image(xs + 1.0, ys + 1.0)
        if use_b: Pm = ndi.spline_filter(Pm, order=3, mode='mirror')
        xp = np.arange(xi - hs, xi + hs + 1, dtype=np.float64); yp = np.arange(yi - hs, yi + hs + 1, dtype=np.float64)
        gx, gy = np.meshgrid(xp, yp)
        w = (1.0 / e).ravel(); yw = d.ravel() * w
        base = np.stack([np.ones(d.size), ((gx - xs) / hs).ravel(), ((gy - ys) / hs).ravel()], 1)
        def chi(p):
            Wx = wf((xp - xs - p[0] + psf.c)[None], psf.nx)[0]; Wy = wf((yp - ys - p[1] + psf.c)[None], psf.ny)[0]
            M = Wy @ Pm @ Wx.T
            A = np.concatenate([M.reshape(-1, 1), base], 1) * w[:, None]
            sol = np.linalg.lstsq(A, yw, rcond=None)[0]
            return float(((yw - A @ sol) ** 2).sum())
        pc, offs = np.zeros(2), offs0
        if len(res) >= CAL_NSEED:
            R_ = np.array(res); pc = np.median(R_[np.argsort(np.hypot(R_[:, 0] - xs, R_[:, 1] - ys), kind='stable')[:CAL_KNN], 2:4], 0); offs = offs1
        g = np.array([[chi(pc + [a, b]) for b in offs] for a in offs]); ia, ib = np.unravel_index(np.argmin(g), g.shape)
        p0 = pc + [offs[ia], offs[ib]]
        o = minimize(chi, p0, method='Nelder-Mead', options=dict(xatol=2e-3, fatol=1e-3, maxiter=120, initial_simplex=np.array([p0, p0 + [0.5, 0.0], p0 + [0.0, 0.5]])))
        if np.all(np.abs(o.x) < CAL_MAXS): res.append((xs, ys, o.x[0], o.x[1], o.fun))
    if len(res) < 8:
        return sm0, len(res), np.nan, dg
    res = np.array(res); nres = len(res)
    sel = res[:, 4] <= np.quantile(res[:, 4], CAL_KEEP)
    if sel.sum() < 8: sel[:] = True
    R = res[sel]; n = len(R)
    hx, hy = nx / 2.0, ny / 2.0
    u, v = (R[:, 0] - hx) / hx, (R[:, 1] - hy) / hy
    occ = len(np.unique(np.clip((R[:, 0] / nx * 3).astype(int), 0, 2) * 3 + np.clip((R[:, 1] / ny * 3).astype(int), 0, 2)))
    degs = [dg_ for dg_, nm in zip(CAL_DEGS, CAL_NMIN) if n >= nm and (dg_ == 1 or occ >= 7)]
    fold = np.random.RandomState(0).permutation(n) % 5
    best = None
    for dgr in degs if len(degs) > 1 else []:
        B = _pbasis(u, v, dgr); e_ = np.zeros(n)
        for k_ in range(5):
            tr = fold != k_
            cx_, cy_ = [_clipfit(B[tr], R[tr, 2 + a])[0] for a in range(2)]
            e_[~tr] = np.hypot(R[~tr, 2] - B[~tr] @ cx_, R[~tr, 3] - B[~tr] @ cy_)
        sc = float(np.median(e_))
        if best is None or sc < 0.98 * best[0]: best = (sc, dgr)
    deg = best[1] if best else 1
    box = (R[:, 0].min(), R[:, 0].max(), R[:, 1].min(), R[:, 1].max())
    for deg in ((deg, 1) if deg > 1 else (1,)):
        B = _pbasis(u, v, deg); coef = np.zeros((2, B.shape[1])); mads = []; oks = np.ones(n, bool)
        for k in range(2):
            c, ok_, sd = _clipfit(B, R[:, 2 + k]); coef[k] = c; mads.append(sd); oks &= ok_
        sm = dict(deg=deg, coef=coef, box=box, nx=nx, ny=ny)
        gxy = np.stack(np.meshgrid(np.linspace(box[0], box[1], 17), np.linspace(box[2], box[3], 17)), -1).reshape(-1, 2)
        fmax = float(np.abs(_shift_eval(sm, gxy)).max())
        if fmax < CAL_MAXS + 1.0: break         # a wild field (sparse corners) falls back to the affine one
    dg = dict(shift_deg=deg, shift_nuse=int(oks.sum()), shift_max=fmax)
    return sm, nres, float(np.hypot(*mads)), dg


def fit_frame(sci_path, fcat_path, dcat_path, psf_path, scale_arcsec, h_override=None, order=0, kmax=30, verbose=False, interp=None,
              recal=True, affine_shift=True, rows=None, debug=None, extra_xy=None, use_det=True, peel=None):
    """Returns (DataFrame of per-source results aligned with the rows of the forced catalogue, info dict)."""
    t0 = time.time()
    F = pd.read_csv(fcat_path)
    n = len(F)
    out = pd.DataFrame({'NUMBER': F['NUMBER'].values})
    for c in ('flux', 'ferr', 'chi2c', 'chi2s', 'sky'):
        out[c] = np.nan
    out['nsrc'] = 0; out['ngood'] = 0; out['npeel'] = 0
    out['dx'] = 0.0; out['dy'] = 0.0
    out['flags'] = 0
    info = {}
    X = F['X_IMAGE'].values.astype(float) - 1.0; Y = F['Y_IMAGE'].values.astype(float) - 1.0
    if psf_path is None or not os.path.exists(psf_path):
        out['flags'] = F_FAIL | F_NONPOS
        info.update(frame_fail=True, twall=time.time() - t0)
        return out, info
    psf = PSF(psf_path, order)
    h = fits.open(sci_path, memmap=True)
    sci = np.asarray(h[0].data, dtype=np.float32); err = np.asarray(h[1].data, dtype=np.float32)
    dq = np.asarray(h[2].data, dtype=np.int32)
    ny, nx = sci.shape
    fw = psf.fwhm
    hs = int(h_override or min(17, int(np.ceil(2.2 * fw))))
    r_flag = 4.0 / scale_arcsec
    r_core = fw          # px
    r_chi = max(3.0, 1.2 * fw)
    margin = 20
    # sources: master rows + extra sources (both WCS-derived: they get the shift) + unmatched per-frame detections (true pixel positions)
    ok = np.isfinite(X) & np.isfinite(Y)
    exy = np.zeros((0, 2)) if extra_xy is None else np.asarray(extra_xy, dtype=float).reshape(-1, 2)
    wxy = np.vstack([np.c_[X, Y], exy])
    wtree = cKDTree(np.where(np.isfinite(wxy).all(1)[:, None], wxy, -1e9))
    xyq = np.where(ok[:, None], wxy[:n], -1e9)
    iso_cal = wtree.query_ball_point(xyq, hs + 8.0, return_length=True) == 1     # calibration stars: isolated among master rows + extras
    sigfloor = float(np.nanmedian(err[::50, ::50]))
    sm = dict(deg=0, coef=np.zeros((2, 1)), box=(-1e9, 1e9, -1e9, 1e9), nx=nx, ny=ny); info['shift_n'] = 0
    if recal:
        sm, info['shift_n'], info['shift_mad'], dgn = _calibrate_shift(psf, sci, err, dq, X, Y, F, np.zeros(n, dtype=np.int32), iso_cal, hs, ok, interp)
        info.update(dgn)
        if not affine_shift:
            sm = dict(sm, deg=0, coef=sm['coef'][:, :1])
    cA = sm['coef']; gg = np.zeros((2, 2))
    if sm['deg'] >= 1: gg = cA[:, 1:3] * np.array([1000.0 / (nx / 2.0), 1000.0 / (ny / 2.0)])   # linear terms at the detector centre, px per 1000 px
    info['shift_x'], info['shift_y'] = float(cA[0, 0]), float(cA[1, 0])
    info['shift_gx'], info['shift_gy'] = float(np.hypot(*gg[0])), float(np.hypot(*gg[1]))
    if debug is not None: debug['shift_model'] = sm
    def shift_at(xy):
        return _shift_eval(sm, xy)
    wxy = wxy + shift_at(wxy)
    nuis = np.zeros((0, 2))
    if use_det and dcat_path is not None and os.path.exists(dcat_path):
        D = pd.read_csv(dcat_path, usecols=['X_IMAGE', 'Y_IMAGE'])
        dxy = np.c_[D['X_IMAGE'].values - 1.0, D['Y_IMAGE'].values - 1.0]
        dxy = dxy[np.isfinite(dxy).all(1)]   # robo43 detection catalogues can carry all-NaN rows
        dd, _ = cKDTree(wxy[:n][ok]).query(dxy)    # unmatched = no (shifted) master row within 2.5 px
        nuis = dxy[dd > 2.5]
    info['nuis_det_xy'] = nuis - shift_at(nuis) if use_det else np.zeros((0, 2))   # back in WCS pixel space (build_static converts with the frame WCS)
    allxy = np.vstack([wxy[:n], nuis, wxy[n:]])
    fap = np.zeros(len(allxy)); fap[:n] = np.nan_to_num(F['FLUX_APER_5'].values, nan=0.0)
    allok = np.isfinite(allxy).all(1)
    tree = cKDTree(np.where(allok[:, None], allxy, -1e9))
    # neighbour / group flags: any other source (master row or unmatched per-frame detection) within 4" / 1.4"
    xyq = np.where(np.isfinite(allxy[:n]).all(1)[:, None], allxy[:n], -1e9)
    cnt4 = tree.query_ball_point(xyq, r_flag, return_length=True)
    cnt14 = tree.query_ball_point(xyq, 1.4 / scale_arcsec, return_length=True)
    flags = np.zeros(n, dtype=np.int32)
    flags[cnt4 > 1] |= F_NEIGHBOUR
    flags[cnt14 > 1] |= F_GROUP
    peel_xy = []
    do_peel = PEEL if peel is None else peel
    X = allxy[:n, 0].copy(); Y = allxy[:n, 1].copy()
    tree = cKDTree(np.where(allok[:, None], allxy, -1e9))
    wf = bspline_w if (interp or INTERP) == 'bspline' else cubic_w
    nref = 0; npeel_tot = 0
    for i in (range(n) if rows is None else rows):
        if not ok[i]:
            flags[i] |= F_FAIL; continue
        xs, ys = X[i], Y[i]
        xi, yi = int(round(xs)), int(round(ys))
        if xi < 0 or yi < 0 or xi >= nx or yi >= ny:
            flags[i] |= F_FAIL | F_EDGE; continue
        if xs < margin or ys < margin or xs > nx - 1 - margin or ys > ny - 1 - margin:
            flags[i] |= F_EDGE
        x0, x1 = max(xi - hs, 0), min(xi + hs + 1, nx); y0, y1 = max(yi - hs, 0), min(yi + hs + 1, ny)
        d = sci[y0:y1, x0:x1].astype(np.float64); e = err[y0:y1, x0:x1].astype(np.float64); q = dq[y0:y1, x0:x1]
        xp = np.arange(x0, x1, dtype=np.float64); yp = np.arange(y0, y1, dtype=np.float64)
        gx, gy = np.meshgrid(xp, yp)
        rr2 = (gx - xs) ** 2 + (gy - ys) ** 2
        masked = ((q & DQ_FITMASK) != 0) | ~np.isfinite(d) | ~(e > 0)
        if ((q & DQ_SATNL) != 0)[rr2 <= r_flag ** 2].any():
            flags[i] |= F_SATNL
        if ((((q & DQ_HARDBAD) != 0) | ~np.isfinite(d))[rr2 <= r_core ** 2]).any() or masked.mean() > 0.2:
            flags[i] |= F_BADPIX
        good = ~masked
        if good.sum() < 0.25 * (2 * hs + 1) ** 2 or good[rr2 <= (2.0) ** 2].sum() < 5:
            flags[i] |= F_FAIL; continue
        # sources in the fit
        cand = np.array(tree.query_ball_point([xs, ys], (hs + 17) * 1.4143), dtype=int)
        cand = cand[allok[cand]]
        cheb = np.maximum(np.abs(allxy[cand, 0] - xs), np.abs(allxy[cand, 1] - ys))
        free = (cheb <= hs + 6) | ((cheb <= hs + 17) & (fap[cand] > 30.0 * max(fap[i], 20.0 * sigfloor)))
        free |= (cand == i)
        cand = cand[free]
        if len(cand) > kmax:
            dist = np.hypot(allxy[cand, 0] - xs, allxy[cand, 1] - ys)
            keep = np.argsort(dist)[:kmax]
            if i not in cand[keep]:
                keep = np.r_[keep[:-1], int(np.nonzero(cand == i)[0][0])]
            cand = cand[keep]
        prim = int(np.nonzero(cand == i)[0][0])
        P = psf.image(xs + 1.0, ys + 1.0)
        if (interp or INTERP) == 'bspline':
            P = ndi.spline_filter(P, order=3, mode='mirror')
        try:
            sf = StampFit(d, e, good, xp, yp, P, psf, wf, hs, xs, ys)
            sf.set_sources(allxy[cand, 0], allxy[cand, 1])
            sf.solve()
            K0 = len(cand)
            fl = sf.sol[prim]; fe = np.sqrt(sf.cov[prim, prim]) if sf.cov[prim, prim] > 0 else np.nan
            dmin = np.hypot(sf.px - sf.px[prim], sf.py - sf.py[prim]); dmin[prim] = 1e9
            did_ref = False
            if REFINE and np.isfinite(fe) and fl > SNR_REF * fe and K0 <= KREF and dmin.min() > ISO_REF * fw:
                tot = sf.refine([prim], NIT_REF, MAXSHIFT_REF); did_ref = bool(np.any(tot != 0))
            # core chi2 of the primary and companion search in the residual image
            def core_chi():
                core = ((sf.px[prim] - gx.ravel()[sf.gm]) ** 2 + (sf.py[prim] - gy.ravel()[sf.gm]) ** 2) <= r_chi ** 2
                rw = sf.resid_w()
                return float((rw[core] ** 2).sum() / max(core.sum() - 1, 1)) if core.sum() > 3 else np.nan
            npeel = 0
            if do_peel:
                while npeel < NPEEL:
                    cc = core_chi()
                    if not (cc > PEEL_CHI): break
                    R = np.where(good, d - sf.model_full(), 0.0)
                    Rs = ndi.uniform_filter(R, 3, mode='constant')
                    allowed = (((gx - xs) ** 2 + (gy - ys) ** 2) <= (hs - 1.5) ** 2) & good
                    for q_ in range(len(sf.px)):
                        allowed &= ((gx - sf.px[q_]) ** 2 + (gy - sf.py[q_]) ** 2) > 1.5 ** 2
                    allowed &= ((gx - sf.px[prim]) ** 2 + (gy - sf.py[prim]) ** 2) >= (PEEL_DMIN * fw) ** 2
                    if not allowed.any(): break
                    Rm = np.where(allowed, Rs, -np.inf); kk = np.unravel_index(np.argmax(Rm), Rm.shape)
                    sig = Rm[kk] / (np.median(e[good]) / 3.0)
                    if not (sig > PEEL_SIG): break
                    chi_before = sf.chi2()
                    keep_state = (sf.px.copy(), sf.py.copy())
                    sf.add(xp[kk[1]], yp[kk[0]]); sf.solve()
                    mob = [len(sf.px) - 1]
                    if REFINE and fl > SNR_REF * fe: mob.append(prim)
                    sf.refine(mob, NIT_REF, MAXSHIFT_REF)
                    kn = len(sf.px) - 1
                    fnew = sf.sol[kn]; enew = np.sqrt(sf.cov[kn, kn]) if sf.cov[kn, kn] > 0 else np.inf
                    if (chi_before - sf.chi2()) > PEEL_DCHI2 and fnew > 3.0 * enew:
                        npeel += 1
                        fl = sf.sol[prim]; fe = np.sqrt(sf.cov[prim, prim]) if sf.cov[prim, prim] > 0 else np.nan
                    else:
                        sf.drop_last(); sf.set_sources(keep_state[0], keep_state[1]); sf.solve()
                        break
            if npeel and REFINE:
                mob = list(range(K0, len(sf.px))) + ([prim] if fl > SNR_REF * fe else [])
                sf.refine(mob, NIT_REF + 4, MAXSHIFT_REF)
            if npeel:
                pxy = np.c_[sf.px[K0:], sf.py[K0:]]
                peel_xy.append(np.c_[pxy - shift_at(pxy), np.full(len(pxy), i)])
            if REWEIGHT:
                sf.reweight()
        except np.linalg.LinAlgError:
            flags[i] |= F_FAIL; continue
        K = len(sf.px)
        fl = sf.sol[prim]; fe = np.sqrt(sf.cov[prim, prim]) if sf.cov[prim, prim] > 0 else np.nan
        res = sf.resid_w()
        chi2s = float((res ** 2).sum() / max(len(res) - (K + 3), 1))
        core = ((sf.px[prim] - gx.ravel()[sf.gm]) ** 2 + (sf.py[prim] - gy.ravel()[sf.gm]) ** 2) <= r_chi ** 2
        chi2c = float((res[core] ** 2).sum() / max(core.sum() - 1, 1)) if core.sum() > 3 else np.nan
        if debug is not None:
            debug[i] = dict(d=d, mod=sf.model_full(), px=sf.px.copy(), py=sf.py.copy(), prim=prim, x0=x0, y0=y0, K=K, chi2c=chi2c, npeel=npeel)
        out.at[i, 'flux'] = fl; out.at[i, 'ferr'] = fe; out.at[i, 'chi2c'] = chi2c; out.at[i, 'chi2s'] = chi2s
        out.at[i, 'sky'] = sf.sol[K]; out.at[i, 'nsrc'] = K; out.at[i, 'ngood'] = int(sf.gm.sum()); out.at[i, 'npeel'] = npeel
        out.at[i, 'dx'] = sf.px[prim] - allxy[i, 0]; out.at[i, 'dy'] = sf.py[prim] - allxy[i, 1]
        if npeel:
            flags[i] |= F_GROUP
            if np.min(np.hypot(sf.px[K0:] - sf.px[prim], sf.py[K0:] - sf.py[prim])) <= r_flag: flags[i] |= F_NEIGHBOUR
        npeel_tot += npeel; nref += int(did_ref)
        if not (np.isfinite(fl) and np.isfinite(fe)):
            flags[i] |= F_FAIL
        if verbose and i % 5000 == 0:
            print(i, n, time.time() - t0, flush=True)
    info['n_refined'] = nref; info['n_peeled'] = npeel_tot
    info['peel_xy'] = np.vstack(peel_xy) if peel_xy else np.zeros((0, 3))
    out['flags'] = flags
    nonpos = ~(np.isfinite(out['flux'].values) & (out['flux'].values > 0) & np.isfinite(out['ferr'].values) & (out['ferr'].values > 0))
    out.loc[nonpos, 'flags'] = out.loc[nonpos, 'flags'].values | F_NONPOS
    info.update(frame_fail=False, twall=time.time() - t0, hs=hs, psf_fwhm=fw, psf_nacc=psf.nacc, psf_chi2=psf.chi2,
                n=n, nnuis=len(nuis), kmean=float(out['nsrc'].mean()))
    return out, info
