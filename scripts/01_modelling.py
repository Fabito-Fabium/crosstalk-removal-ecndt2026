import sys
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).parent.parent
sys.path.extend(str(root))
import matplotlib

matplotlib.use('TkAgg')

from crsLib.graphing import *
from crsLib.estimation import *
from crsLib.desatFunc import *
import json

DTYPE = np.float64

with open('config.json', 'r') as file:
    config = json.load(file)

    # Problem-related constants:
    num_elem = config["transducer"]["num_elem"]
    h_len = config["modelling"]["h_len"]
    dtemp = config["preprocessing"]["dtemp"]
    model_type = config["modelling"]["model_type"]

    # Important filenames:
    transducer_type = config["transducer"]["type"]
    central_freq = config["transducer"]["central_freq_MHz"]
    manufacturer = config["transducer"]["manufacturer"]
    output_filename = f"{transducer_type}_{manufacturer}_{num_elem:.0f}_{central_freq:.0f}MHz"

# Important paths:
data_path = root / 'data'
model_path = data_path / "m2k" / output_filename / "calibration"
models_path = data_path / "models"
figures_path = root / "figures"

bscan = np.load(data_path / "processed_data" / (str(output_filename) + ".npy"), allow_pickle=True)

# %%
def estimateModel(i_idx):
    f_i = bscan[:, i_idx, i_idx]

    all_coefsP = np.zeros((6, num_elem, num_elem))
    all_coefsU = np.zeros((6, num_elem, num_elem))

    h_ij_fir = np.zeros((h_len, num_elem, num_elem), dtype=DTYPE)
    h_ij_iirP = np.zeros((h_len, num_elem, num_elem), dtype=DTYPE)
    h_ij_iirU = np.zeros((h_len, num_elem, num_elem), dtype=DTYPE)
    for j_idx in range(num_elem):
        y = bscan[:, j_idx, i_idx]
        x = f_i
        # estimação FIR
        h_ij_fir[:, j_idx, i_idx] = estimate_h_fir(y, x, h_len, mthd='irls', damp=1e-3, niter=40)

        # # estimação IIR Pointwise
        # n0p, coefp, _ = estimate_iir_secondOrder_Pointwise(x, y, DTEMP, eps_max, damp=1e-3, mthd='irls', niter=40)
        # h_ij_iirP[:, j_idx, i_idx] = impulse_response(coefp, n0p, h_len)
        # all_coefsP[:, j_idx, i_idx] = np.concatenate([coefp, [n0p]])
        #
        # # estimação IIR Uniform
        # n0u, coefu, _ = estimate_iir_secondOrder_Uniform(x, y, DTEMP, eps_max, damp=1e-2, mthd='_', niter=500)
        # h_ij_iirU[:, j_idx, i_idx] = impulse_response(coefu, n0u, h_len)
        # all_coefsU[:, j_idx, i_idx] = np.concatenate([coefu, [n0u]])

    return f_i, h_ij_fir, h_ij_iirP, h_ij_iirU, all_coefsP, all_coefsU

BigPack = Parallel(n_jobs=-1)(delayed(estimateModel)(i) for i in tqdm(range(num_elem)))

f_i = np.zeros((dtemp, num_elem), dtype=DTYPE)
h_ij_fir = np.zeros((h_len, num_elem, num_elem), dtype=DTYPE)
h_ij_iirP = np.zeros((h_len, num_elem, num_elem), dtype=DTYPE)
h_ij_iirU = np.zeros((h_len, num_elem, num_elem), dtype=DTYPE)
all_coefsP = np.zeros((6, num_elem, num_elem), dtype=DTYPE)
all_coefsU = np.zeros((6, num_elem, num_elem), dtype=DTYPE)

for i_idx in range(num_elem):
    f_i[:, i_idx] = BigPack[i_idx][0]
    h_ij_fir[:, :, i_idx] = BigPack[i_idx][1][:, :, i_idx]
    h_ij_iirP[:, :, i_idx] = BigPack[i_idx][2][:, :, i_idx]
    h_ij_iirU[:, :, i_idx] = BigPack[i_idx][3][:, :, i_idx]
    all_coefsP[:, :, i_idx] = BigPack[i_idx][4][:, :, i_idx]
    all_coefsU[:, :, i_idx] = BigPack[i_idx][5][:, :, i_idx]

# %%
all_ascan_est_fir = np.zeros_like(bscan)
all_ascan_est_iirP = np.zeros_like(bscan)
all_ascan_est_iirU = np.zeros_like(bscan)

for idx_i in range(num_elem):
    for idx_j in range(num_elem):
        all_ascan_est_fir[:, idx_i, idx_j] = np.convolve(f_i[:, idx_j], h_ij_fir[:, idx_i, idx_j], 'full')[:dtemp]

        coefs = all_coefsP[:5, idx_i, idx_j]
        n0b = int(all_coefsP[5, idx_i, idx_j])

        b_delayed = np.pad(coefs[:-2], (n0b, 0), 'constant', constant_values=(0, 0))
        lfilter = lfilter_operator(b_delayed, [1, -coefs[3], -coefs[4]], dtemp)

        all_ascan_est_iirP[:, idx_j, idx_i] = lfilter * f_i[:, idx_i]

        coefs = all_coefsU[:5, idx_i, idx_j]
        n0b = int(all_coefsU[5, idx_i, idx_j])

        b_delayed = np.pad(coefs[:-2], (n0b, 0), 'constant', constant_values=(0, 0))
        lfilter = lfilter_operator(b_delayed, [1, -coefs[3], -coefs[4]], dtemp)

        all_ascan_est_iirU[:, idx_j, idx_i] = lfilter * f_i[:, idx_i]

# # %%
# interactive_frf(all_ascan_est_fir, title='fir', Func_true=bscan)
# interactive_frf(h_ij_fir, title='fir')#, Func_true=bscan)
# %%
func_true = bscan[:, 2, 5]
func = all_ascan_est_fir[:, 2, 5]
fs=120e6

f = np.linspace(1e4, 120e6, 1000)

w, Func = ss.freqz(func, worN=f, fs=fs)
_, Func_true = ss.freqz(func_true, worN=f, fs=fs)

plt.figure(figsize=(3*1.2, 2*1.2))
plt.semilogx(w, 20 * np.log10(np.abs(Func)), label='Estimated', color='black', linewidth=2)
plt.semilogx(w, 20 * np.log10(np.abs(Func_true)), label='Observed', color='red', linewidth=1, linestyle='--')
plt.axvline(5e6, ls='-', lw=0.5, color='b')
plt.ylim([-37, 85])
plt.fill_between(np.linspace(2.5e6, 7.5e6, 2), np.ones(2) - 100, np.ones(2) + 100, alpha=.25)
plt.xlabel('Frequency / (Hz)')
plt.ylabel('Amplitude / (dB)')
plt.grid(True)
plt.tight_layout()
plt.savefig(figures_path / "fig3b.pdf")

func_true=func_true/np.max(np.abs(func_true))
func=func/np.max(np.abs(func))

plt.figure(figsize=(3*1.2, 2*1.2))
plt.plot(func, label='Estimated', color='black', linewidth=2)
plt.plot(func_true, label='Observed', color='red', linewidth=1, linestyle='--')
plt.xlabel('Time / (Samples)')
plt.ylabel('Normalized Amplitude')
plt.xlim([200, 800])
plt.grid(True)
plt.xticks([300, 400, 500, 600, 700])
plt.tight_layout()
plt.savefig(figures_path / "fig3a.pdf")


plt.figure(figsize=(3*1.2, 2*1.2))
plt.semilogx(w, np.unwrap(np.angle(Func)), label='Estimated', color='black', linewidth=2)
plt.semilogx(w, np.unwrap(np.angle(Func_true)), label='Observed', color='red', linewidth=1, linestyle='--')
plt.xlabel('Frequency / (Hz)')
plt.ylabel('Unwrapped Phase / (rad)')
plt.grid(True)
plt.tight_layout()
plt.legend(loc='upper left')
plt.savefig(figures_path / "fig3c.pdf")

# %%
os.makedirs(models_path, exist_ok=True)
# %%
os.makedirs(str(models_path / output_filename), exist_ok=True)
np.savez(models_path / output_filename / f"model_{model_type}_{h_len}.npz", FIR=h_ij_fir,
         IIRu_trunk=h_ij_iirU, IIRp_trunk=h_ij_iirP,
         IIRu_coefs=all_coefsU, IIRp_coefs=all_coefsP)
print(f"Model create and saved at {models_path / output_filename}/model.npz")
# %%
el = 31
path1 = model_path / f'{el + 1}.m2k'

eliso = file_m2k.read(str(path1), freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0,
                      read_ascan=True, type_insp="contact", water_path=0.0)

ascan = eliso.ascan_data[250:, 0, :num_elem, 0]

t_start = np.argmax(ascan[:, el]) - dtemp // 2
t_end = np.argmax(ascan[:, el]) + dtemp // 2

# %%
plt.figure(figsize=(3*1.2, 2*1.2))
time = eliso.time_grid
img_extent = [1, 65, time[t_end], time[t_start]]
plt.imshow(np.log10(envelope(bscan[:, :, 30])), aspect='auto', extent=img_extent, interpolation='nearest')
plt.xlabel('Channels')
plt.ylabel(r'Time / $(\mu s)$')
plt.tight_layout()
plt.savefig(figures_path / "fig2b.pdf")