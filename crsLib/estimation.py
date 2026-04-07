import matplotlib.pyplot as plt
import numpy as np

from framework import file_m2k
import scipy.optimize as so
import scipy.signal as ss
import scipy.sparse.linalg as la
import h5py
import pylops as pl
import pylops.optimization as po

from scipy.signal import hilbert, convolve2d

from tqdm import tqdm

import os
from joblib import Parallel, delayed
import h5py
np.int = int
# %%
def bandpass(signal, cutoff, Cutoff, firSize=500., show=False):
    fs = 125e6
    band = [cutoff, Cutoff]
    trans_width = 5e5
    numtaps = int(firSize)
    edges = [0, band[0] - trans_width, band[0], band[1],
             band[1] + trans_width, 0.5 * fs]
    taps = ss.remez(numtaps, edges, [0, 1, 0], fs=fs)

    if show == True:
        w, h = ss.freqz(taps, [1], worN=2000, fs=fs)
        plt.plot(w, np.abs(h))

    bp_sig = np.real(np.convolve(signal, taps, mode='same'))

    return bp_sig

def argsortbscan(bscan):
  x2 = np.abs(hilbert(bscan,axis=0))
  x2 = np.log(x2+1e-10)
  ones = np.ones((500,1))
  xm = convolve2d(x2,ones,mode='same')
  xb = xm > xm.max(axis=0) * .8
  xb = xb * np.flip(np.arange(bscan.shape[0]))[:, np.newaxis]
  ids = np.argsort(xb.argmax(axis=0))
  return ids


# Operadores lineares:
# %%
# Função que retorna um operador linear dado um kernel genérico
def myLinearOp(kernel, operand_len):
    kernel_len = len(kernel)
    n = operand_len
    m = np.max((n, kernel_len))

    if (np.minimum(kernel_len, n) - 1) == 0:
        slNone = None
        slzero = 0
    else:
        slNone = -(np.minimum(kernel_len, n) - 1)
        slzero = (np.minimum(kernel_len, n) - 1)

    def matvec(operand):
        return ss.convolve(operand, kernel, mode='full')[:slNone]

    def rmatvec(operand_):
        operand_pad_ = np.pad(operand_, (0, slzero))
        return ss.correlate(operand_pad_, kernel, mode='valid')

    return la.LinearOperator(shape=(m, n), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

def test_adjoint(operator, rtol=1e-10):
    m, n = operator.shape

    # Random test vectors
    x = np.random.randn(n)
    y = np.random.randn(m)

    # Test adjoint property: <Ax, y> = <x, A*y>
    Ax = operator.matvec(x)
    ATy = operator.rmatvec(y)

    lhs = np.dot(Ax, y)
    rhs = np.dot(x, ATy)

    print(f"<Ax, y> = {lhs:.10f}")
    print(f"<x, AHy> = {rhs:.10f}")
    print(f"Relative error: {abs(lhs - rhs) / max(abs(lhs), abs(rhs)):.2e}")

    return abs(lhs - rhs) / max(abs(lhs), abs(rhs)) < rtol

# %%
def impulse_response(coef, n0, length=50):
    b = [coef[0], coef[1], coef[2]]
    a = [1, -coef[3], -coef[4]]

    impulse = np.zeros(length)
    impulse[n0] = 1

    h = ss.lfilter(b, a, impulse)
    return h

# %%
def solveAxb(A, b, damp=None, niter=100, mthd=None):
    match mthd:
        case 'lsqr':
            x = po.basic.lsqr(A, b, damp=damp, niter=niter, show=False)[0]
        case 'cgls':
            x = po.basic.cgls(A, b, damp=damp, niter=niter, show=False)[0]
        case 'irls':
            A = pl.LinearOperator(A)
            x = po.sparsity.irls(A, b, nouter=niter, epsI=damp, kind="model", **dict(iter_lim=10))[0]
        case _:
            x = so.lsq_linear(A, b).x
    return x


# FIR equiv estimação via leastsquares
def estimate_h_fir(b, a, h_len, mthd=None, damp=1e-5, niter=100):
    A = myLinearOp(a, h_len)
    x = solveAxb(A, b, damp=damp, mthd=mthd, niter=niter)
    return x

# %%
def rollz(v, n=1):
  v_ = np.roll(v, n)
  v_[:n] = 0.
  return v_

# applysys é um operador não linear, desse modo,
# applysysT é um adjunto para um caso específico.
def applysys(x, c, n0=0):
    b_delayed = np.pad(c[:-2], (n0, 0), 'constant', constant_values=(0, 0))
    return ss.lfilter(b_delayed, [1, -c[3], -c[4]], x)

def applysysT(x, y, n0=0):
    A = np.zeros((len(y), 5))
    A[:, 0] = rollz(x, n0)
    A[:, 1] = rollz(x, 1+n0)
    A[:, 2] = rollz(x, 2+n0)
    A[:, 3] = rollz(y, 1)
    A[:, 4] = rollz(y, 2)
    ch = A.T @ y
    return ch

# "estimação uniforme" (misto entre ARX e IIR puro)

def estimate_iir_secondOrder_Uniform(x, y, dtmp, eps_max=15, damp=1e-6, niter=100, mthd=None):
  residMine = np.zeros(eps_max)
  n0s = np.zeros(eps_max)
  coefs = np.zeros((eps_max, 5))
  for n0 in range(eps_max):
    SysMtx = la.LinearOperator((dtmp, 5)
                                , matvec=lambda ww: applysys(x, ww, n0)
                                , rmatvec=lambda ww: applysysT(x, ww, n0))

    coef = solveAxb(SysMtx, y, damp=damp, niter=niter, mthd=mthd)

    y_est = applysys(x, coef, n0)
    Res = np.linalg.norm(y_est - y)

    coefs[n0, :] = coef
    residMine[n0] = Res
    n0s[n0] = n0

  # plt.semilogy(n0s, residuals, 'o-')
  winner = np.argmin(residMine)
  return int(n0s[winner]), coefs[winner], residMine[winner]


# "estimação pontual" (ARX)
def estimate_iir_secondOrder_Pointwise(x, y, dtmp, eps_max=15, damp=1e-6, mthd=None, niter=100):
  residMine = np.zeros(eps_max)
  n0s = np.zeros(eps_max)
  coefs = np.zeros((eps_max, 5))
  for n0 in range(eps_max):
    A = np.zeros((len(y), 5))
    A[:, 0] = rollz(x, n0)
    A[:, 1] = rollz(x, 1+n0)
    A[:, 2] = rollz(x, 2+n0)
    A[:, 3] = rollz(y, 1)
    A[:, 4] = rollz(y, 2)

    SysMtx = la.LinearOperator((dtmp, 5), matvec=lambda ww: A@ww, rmatvec=lambda ww: A.T@ww)
    coef = solveAxb(SysMtx, y, damp=damp, niter=niter, mthd=mthd)

    y_est = applysys(x, coef, n0)
    Res = np.linalg.norm(y_est - y)
    coefs[n0, :] = coef
    residMine[n0] = Res
    n0s[n0] = n0

  winner = np.argmin(residMine)
  return int(n0s[winner]), coefs[winner], residMine[winner]

# %%
def lfilter_operator(b, a, operand_len=None, delay=0):
  n = operand_len

  def matvec(x):
    return ss.lfilter(b, a, x)

  def rmatvec(y):
    return ss.lfilter(b, a, y[::-1])[::-1]

  return la.LinearOperator(shape=(n, n), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

def FIR_linOp(all_h, Nt, Ne):
    total_size = Nt * Ne
    h_len = all_h.shape[0]

    valid_idx = []
    for j in range(Ne):
        for k in range(j, Ne):
            #if j != k:
            valid_idx.append((j, k))

    def matvec(x):
        F = x.reshape(Nt, Ne)
        G = np.zeros_like(F)
        # G recebe F (identidade)
        #G[:, :] = F[:, :]

        for j, k in valid_idx:
            G[:, j] += ss.convolve(F[:, k], all_h[:, j, k], mode='full')[:-(h_len - 1)]
            if j != k:
                G[:, k] += ss.convolve(F[:, j], all_h[:, k, j], mode='full')[:-(h_len - 1)]
        return G.ravel()

    def rmatvec(y):
        G = y.reshape(Nt, Ne)
        F = np.zeros_like(G)
        # F recebe G (identidade)
        #F += G
        for j, k in valid_idx:
            G_padded = np.pad(G[:, j], (0, h_len-1))
            F[:, k] += ss.correlate(G_padded, all_h[:, j, k], mode='valid')

            if j != k:
                G_padded = np.pad(G[:, k], (0, h_len - 1))
                F[:, j] += ss.correlate(G_padded, all_h[:, k, j], mode='valid')

        return F.ravel()

    return la.LinearOperator(shape=(total_size, total_size), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

def IIR_linOp(all_filt, Nt, Ne):
    total_size = Nt * Ne

    valid_idx = []
    for j in range(Ne):
        for k in range(j, Ne):
            #if j != k:
            valid_idx.append((j, k))

    def matvec(x):
        F = x.reshape(Nt, Ne)
        G = np.zeros_like(F)
        # G recebe F (identidade)
        #G[:, :] = F[:, :]
        for j, k in valid_idx:
            # G[:, j] += ss.convolve(F[:, k], all_h[:, j, k], mode='full')[:Nt]
            G[:, j] += all_filt[j, k] * F[:, k]
            if j != k:
                # G[:, k] += ss.convolve(F[:, j], all_h[:, k, j], mode='full')[:Nt]
                G[:, k] += all_filt[k, j] * F[:, j]
        return G.ravel()

    def rmatvec(y):
        G = y.reshape(Nt, Ne)
        F = np.zeros_like(G)
        # F recebe G (identidade)
        #F += G
        for j, k in valid_idx:
            # G_padded = np.pad(G[:, j], (0, h_len-1))
            # F[:, k] += ss.correlate(G_padded, all_h[:, j, k], mode='valid')
            F[:, k] += all_filt[j, k].T * G[:, j]
            if j != k:
                # G_padded = np.pad(G[:, k], (0, h_len - 1))
                # F[:, j] += ss.correlate(G_padded, all_h[:, k, j], mode='valid')
                F[:, j] += all_filt[k, j].T * G[:, k]
        return F.ravel()

    return la.LinearOperator(shape=(total_size, total_size), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

def solve_mthd(model, solve, bscan, Nt, Ne, iscoefs=False, lmbd=1e-5, niter=20, rtol=1e-20, callback=None, x0=None, show=True):
    if iscoefs == True:
        all_filt = np.zeros((Ne, Ne), dtype=la.LinearOperator)

        for idx_i in range(Ne):
            for idx_j in range(Ne):
                coefs = model[:5, idx_i, idx_j]
                n0b = int(model[5, idx_i, idx_j])

                b_delayed = np.pad(coefs[:-2], (n0b, 0), 'constant', constant_values=(0, 0))
                lfilter = lfilter_operator(b_delayed, [1, -coefs[3], -coefs[4]], Nt)

                all_filt[idx_j, idx_i] = lfilter


        match solve:
            case 'standard_cgls':
                H = pl.LinearOperator(IIR_linOp(all_filt, Nt, Ne))
                b = bscan.ravel()
                solution = po.basic.cgls(H, b, damp=lmbd, show=show, niter=niter, callback=callback, x0=x0)

            case 'standard_lsqr':
                H = pl.LinearOperator(IIR_linOp(all_filt, Nt, Ne))
                b = bscan.ravel()
                solution = po.basic.lsqr(H, b, damp=np.sqrt(lmbd), show=show, niter=niter, callback=callback, x0=x0, atol=rtol)

            case _:
                print('solve option invalid')
                solution = None

    else:
        match solve:
            case 'standard_lsqr':
                H = FIR_linOp(model, Nt, Ne)
                b = bscan.ravel()
                solution = po.basic.lsqr(H, b, damp=np.sqrt(lmbd), niter=niter, show=show, callback=callback, x0=x0)
            case 'standard_cgls':
                H = FIR_linOp(model, Nt, Ne)
                b = bscan.ravel()
                solution = po.basic.cgls(H, b, damp=lmbd, niter=niter, show=show, callback=callback, x0=x0)

            case _:
                print('solve option invalid')
                solution = None

    return solution

def get_metrics(bscan_og, msk_dir, debug=False, save_all=False):
    if not debug:
        msk = np.load(msk_dir + 'masks.npz')
        msk_sig = msk['signal']
        msk_crs = msk['cross']
        msk_noise = msk['noise']
        axs = msk['argsort']
    else:
        msk_sig = msk_dir[0]
        msk_crs = msk_dir[1]
        msk_noise = msk_dir[2]
        axs = msk_dir[3]

    bscan = bscan_og[:, axs]

    msks = [msk_sig, msk_crs, msk_noise]

    mean_abs_sqr = [np.mean(np.abs(bscan[aux]) ** 2) for aux in msks]
    std_abs_sqr = [np.std(np.abs(bscan[aux]) ** 2) for aux in msks]
    L2_normalized = [np.linalg.norm(bscan[aux]) / np.sum(aux) for aux in msks]

    cr_x = mean_abs_sqr[0] / mean_abs_sqr[1]
    # cnr_x = np.abs(mean_abs_sqr[0] - mean_abs_sqr[1]) / np.sqrt(std_abs_sqr[0] ** 2 + std_abs_sqr[1] ** 2)
    cnr_x = np.abs(mean_abs_sqr[0] - mean_abs_sqr[1]) / std_abs_sqr[1]
    # cnr_x = np.abs(mean_abs_sqr[0] - mean_abs_sqr[1]) / np.sqrt(mean_abs_sqr[0] ** 2 + mean_abs_sqr[1] ** 2)
    sinr_x = (L2_normalized[0] ** 2) / (L2_normalized[1] ** 2 + L2_normalized[2] ** 2)


    return cr_x, cnr_x, sinr_x, mean_abs_sqr, std_abs_sqr, L2_normalized

def sv_model(x, num_el=127, name='current', msk_dir=None):
    try:
        xx = np.load(name + f'current_x0.npy')
        xx[:-1] = x
        xx[-1] += 1
    except:
        xx = np.zeros(x.shape[0]+1)
        xx[-1] = 1
        xx[:-1] = x

    np.save(name + f'current_x0.npy', xx)

    # if xx[-1] % 20 == 0:
    #     try:
    #         auxx = np.load(name + f'iter_out.npy')
    #         aux = np.zeros((xx[:-1].shape[0], int(xx[-1]//20)))
    #         try:
    #             aux[:, :-1] = auxx
    #         except:
    #             aux[:, 0] = auxx
    #         aux[:, -1] = xx[:-1]
    #         np.save(name + f'iter_out', aux)
    #     except:
    #         np.save(name + f'iter_out', xx[:-1])

    try:
        aux = x.reshape((-1, num_el))
        cr_x, cnr_x, sinr_x, _, _, _ = get_metrics(aux, msk_dir)

        # cnr = np.load(f'{name}' + f'cnr.npy')
        # cr = np.load(f'{name}' + f'cr.npy')
        # sinr = np.load(f'{name}' + f'sinr.npy')

        with h5py.File(name + 'metrics.h5', 'r') as f:
            cnr = np.array(f['cnr'])  # Works for both scalars and arrays
            cr = np.array(f['cr'])
            sinr = np.array(f['sinr'])

        if np.max(cr) <= cr_x:
            np.save(f'{name}' + f'x_best_cr.npy', x)

        if np.max(cnr) <= cnr_x:
            np.save(f'{name}' + f'x_best_cnr.npy', x)

        if np.max(sinr) <= sinr_x:
            np.save(f'{name}' + f'x_best_sinr.npy', x)

        try:
            cnr = np.concatenate([cnr, [cnr_x]])
            cr = np.concatenate([cr, [cr_x]])
            sinr = np.concatenate([sinr, [sinr_x]])
        except:
            cnr = np.concatenate([[cnr], [cnr_x]])
            cr = np.concatenate([[cr], [cr_x]])
            sinr = np.concatenate([[sinr], [sinr_x]])

        with h5py.File(name + 'metrics.h5', 'a') as f:
            for name, data in [('cnr', cnr), ('cr', cr), ('sinr', sinr)]:
                if name in f:
                    del f[name]

                if np.ndim(data) > 0:
                    # Use compression for arrays
                    f.create_dataset(name, data=data, compression="gzip")
                else:
                    # Save as a scalar without compression
                    f.create_dataset(name, data=data)
    except:
        pass

