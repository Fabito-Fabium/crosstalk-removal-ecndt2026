from pathlib import Path
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
    NUM_ELEM = config["preprocessing"]["num_elem"]
    H_LEN_FIR = config["modelling"]["h_len_fir"]
    DTEMP = config["preprocessing"]["dtemp"]

    # Important filenames:
    npz_name = config["modelling"]["npz_name_template"].replace("{entry}", str(H_LEN_FIR))
    npy_bscan = config["preprocessing"]["npy_bscan_filename"]

# Important paths:
root = Path(__file__).parent.parent
data_path = root / 'data'
model_path = data_path / "m2k" / "linear_imasonic" / "model"
models_path = data_path / "models"
figures_path = root / "figures"

all_ascan = np.load(data_path / str('prePrcs_modelagem/' + npy_bscan), allow_pickle=True)

# %%
def estimateModel(i_idx):
    f_i = all_ascan[:, i_idx, i_idx]

    all_coefsP = np.zeros((6, NUM_ELEM, NUM_ELEM))
    all_coefsU = np.zeros((6, NUM_ELEM, NUM_ELEM))

    h_ij_fir = np.zeros((H_LEN_FIR, NUM_ELEM, NUM_ELEM), dtype=DTYPE)
    h_ij_iirP = np.zeros((H_LEN_FIR, NUM_ELEM, NUM_ELEM), dtype=DTYPE)
    h_ij_iirU = np.zeros((H_LEN_FIR, NUM_ELEM, NUM_ELEM), dtype=DTYPE)
    for j_idx in range(NUM_ELEM):
        y = all_ascan[:, j_idx, i_idx]
        x = f_i
        # estimação FIR
        h_ij_fir[:, j_idx, i_idx] = estimate_h_fir(y, x, H_LEN_FIR, mthd='irls', damp=1e-3, niter=40)

        # # estimação IIR Pointwise
        # n0p, coefp, _ = estimate_iir_secondOrder_Pointwise(x, y, DTEMP, eps_max, damp=1e-3, mthd='irls', niter=40)
        # h_ij_iirP[:, j_idx, i_idx] = impulse_response(coefp, n0p, H_LEN_FIR)
        # all_coefsP[:, j_idx, i_idx] = np.concatenate([coefp, [n0p]])
        #
        # # estimação IIR Uniform
        # n0u, coefu, _ = estimate_iir_secondOrder_Uniform(x, y, DTEMP, eps_max, damp=1e-2, mthd='_', niter=500)
        # h_ij_iirU[:, j_idx, i_idx] = impulse_response(coefu, n0u, H_LEN_FIR)
        # all_coefsU[:, j_idx, i_idx] = np.concatenate([coefu, [n0u]])

    return f_i, h_ij_fir, h_ij_iirP, h_ij_iirU, all_coefsP, all_coefsU

BigPack = Parallel(n_jobs=-1)(delayed(estimateModel)(i) for i in tqdm(range(NUM_ELEM)))

f_i = np.zeros((DTEMP, NUM_ELEM), dtype=DTYPE)
h_ij_fir = np.zeros((H_LEN_FIR, NUM_ELEM, NUM_ELEM), dtype=DTYPE)
h_ij_iirP = np.zeros((H_LEN_FIR, NUM_ELEM, NUM_ELEM), dtype=DTYPE)
h_ij_iirU = np.zeros((H_LEN_FIR, NUM_ELEM, NUM_ELEM), dtype=DTYPE)
all_coefsP = np.zeros((6, NUM_ELEM, NUM_ELEM), dtype=DTYPE)
all_coefsU = np.zeros((6, NUM_ELEM, NUM_ELEM), dtype=DTYPE)

for i_idx in range(NUM_ELEM):
    f_i[:, i_idx] = BigPack[i_idx][0]
    h_ij_fir[:, :, i_idx] = BigPack[i_idx][1][:, :, i_idx]
    h_ij_iirP[:, :, i_idx] = BigPack[i_idx][2][:, :, i_idx]
    h_ij_iirU[:, :, i_idx] = BigPack[i_idx][3][:, :, i_idx]
    all_coefsP[:, :, i_idx] = BigPack[i_idx][4][:, :, i_idx]
    all_coefsU[:, :, i_idx] = BigPack[i_idx][5][:, :, i_idx]

# %%
all_ascan_est_fir = np.zeros_like(all_ascan)
all_ascan_est_iirP = np.zeros_like(all_ascan)
all_ascan_est_iirU = np.zeros_like(all_ascan)

for idx_i in range(NUM_ELEM):
    for idx_j in range(NUM_ELEM):
        all_ascan_est_fir[:, idx_i, idx_j] = np.convolve(f_i[:, idx_j], h_ij_fir[:, idx_i, idx_j], 'full')[:DTEMP]

        coefs = all_coefsP[:5, idx_i, idx_j]
        n0b = int(all_coefsP[5, idx_i, idx_j])

        b_delayed = np.pad(coefs[:-2], (n0b, 0), 'constant', constant_values=(0, 0))
        lfilter = lfilter_operator(b_delayed, [1, -coefs[3], -coefs[4]], DTEMP)

        all_ascan_est_iirP[:, idx_j, idx_i] = lfilter * f_i[:, idx_i]

        coefs = all_coefsU[:5, idx_i, idx_j]
        n0b = int(all_coefsU[5, idx_i, idx_j])

        b_delayed = np.pad(coefs[:-2], (n0b, 0), 'constant', constant_values=(0, 0))
        lfilter = lfilter_operator(b_delayed, [1, -coefs[3], -coefs[4]], DTEMP)

        all_ascan_est_iirU[:, idx_j, idx_i] = lfilter * f_i[:, idx_i]

# # %%
# interactive_frf(all_ascan_est_fir, title='fir', Func_true=all_ascan)
# interactive_frf(h_ij_fir, title='fir')#, Func_true=all_ascan)
# %%
func_true = all_ascan[:, 2, 5]
func = all_ascan_est_fir[:, 2, 5]
fs=120e6

f = np.linspace(1e4, 120e6, 1000)

w, Func = ss.freqz(func, worN=f, fs=fs)
_, Func_true = ss.freqz(func_true, worN=f, fs=fs)

xx = w / (2 * np.pi)

plt.figure(figsize=(3*1.2, 2*1.2))
plt.semilogx(xx, 20 * np.log10(np.abs(Func)), label='Estimated', color='black', linewidth=2)
plt.semilogx(xx, 20 * np.log10(np.abs(Func_true)), label='Observed', color='red', linewidth=1, linestyle='--')
plt.xlabel('Frequency / (Hz)')
plt.ylabel('Amplitude / (dB)')
plt.grid(True)
plt.tight_layout()
plt.savefig(figures_path / "fig4b.pdf")

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
plt.savefig(figures_path / "fig4a.pdf")


plt.figure(figsize=(3*1.2, 2*1.2))
plt.semilogx(xx, np.unwrap(np.angle(Func)), label='Estimated', color='black', linewidth=2)
plt.semilogx(xx, np.unwrap(np.angle(Func_true)), label='Observed', color='red', linewidth=1, linestyle='--')
plt.xlabel('Frequency / (Hz)')
plt.ylabel('Unwrapped Phase / (rad)')
plt.grid(True)
plt.tight_layout()
plt.legend(loc='upper left')
plt.savefig(figures_path / "fig4c.pdf")

# %%
os.makedirs(models_path, exist_ok=True)
# %%
np.savez(models_path / npz_name, FIR=h_ij_fir,
         IIRu_trunk=h_ij_iirU, IIRp_trunk=h_ij_iirP,
         IIRu_coefs=all_coefsU, IIRp_coefs=all_coefsP)
print(f"Model create and saved at {models_path / npz_name}")
# %%
el = 31
path1 = model_path / f'{el + 1}.m2k'

eliso = file_m2k.read(str(path1), freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0,
                      read_ascan=True, type_insp="contact", water_path=0.0)

ascan = eliso.ascan_data[250:, 0, :NUM_ELEM, 0]

t_start = np.argmax(ascan[:, el]) - DTEMP // 2
t_end = np.argmax(ascan[:, el]) + DTEMP // 2

# %%
plt.figure(figsize=(3*1.2, 2*1.2))
time = eliso.time_grid
img_extent = [1, 65, time[t_end], time[t_start]]
plt.imshow(np.log10(envelope(all_ascan[:, :, 30])), aspect='auto', extent=img_extent, interpolation='nearest')
plt.xlabel('Channels')
plt.ylabel(r'Time / $(\mu s)$')
plt.tight_layout()
plt.savefig(figures_path / "fig3b.pdf")