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


# %%
def ascan_prePross(el):
    path1 = model_path / f'{el+1}.m2k'

    eliso = file_m2k.read(str(path1), freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0,
                          read_ascan=True, type_insp="contact", water_path=0.0)

    ascan = eliso.ascan_data[250:, 0, :NUM_ELEM, 0]

    # eliso = file_m2k.read(path1, freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0,
    #                       read_ascan=True, type_insp="contact", water_path=0.0)[1]
    #
    # ascan = eliso.ascan_data[:, 0, :NUM_ELEM, 0]

    # ascan = np.mean(eliso.ascan_data[:, 0, :NUM_ELEM, :], axis=2)



    t_start = np.argmax(ascan[:, el]) - DTEMP//2
    t_end = np.argmax(ascan[:, el]) + DTEMP//2

    current = np.zeros((DTEMP, ascan.shape[1]), dtype=float)
    for i_idx in range(NUM_ELEM):
        max_prim = np.max(abs(ascan[t_start:t_end, i_idx]))
        dec = Decimator(ascan[t_start:t_end, i_idx], max=max_prim)

        # current[:, i_idx] = Reconstruct(dec, ascan[t_start:t_end, i_idx], ascan[t_start:t_end, i_idx])

        desat = Reconstruct(dec, ascan[t_start:t_end, i_idx], ascan[t_start:t_end, i_idx])
        # desat = ascan[t_start:t_end, i_idx]
        # current[:, i_idx] = desat
        current[:, i_idx] = bandpass(desat, 1e6, 9e6, 300, False)

    return current

iter = NUM_ELEM
parallel_tmp = Parallel(n_jobs=-3)(delayed(ascan_prePross)(i) for i in tqdm(range(iter)))

bscans = np.zeros((DTEMP, NUM_ELEM, NUM_ELEM), dtype=float)
# %%
for i in range(iter):
    bscans[:, :, i] = parallel_tmp[i]
# %% ## tentativa de corrigir o problema na aquisição 59 no qual o sinal ideal não está no canal indicado
aux = bscans[:, :, 63-58].copy()
bscans[:, :, 58] = np.flip(aux, axis=1)
# %%
sel_el = 50

path1 = model_path / f'{sel_el}.m2k'

eliso = file_m2k.read(str(path1), freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=None,
                         read_ascan=True, type_insp="contact", water_path=0.0)#[1]

ascan = eliso.ascan_data[600:, 0, :NUM_ELEM, 0]
t_start = np.argmax(ascan[:, sel_el-1]) - DTEMP // 2
t_end = np.argmax(ascan[:, sel_el-1]) + DTEMP // 2
ascan = ascan[t_start:t_end, :]
plt.figure(figsize=(15, 8))
plt.imshow(np.log10(envelope(ascan)+1e-6), aspect='auto', interpolation='nearest')

os.makedirs(root / 'figures/_model_prePrcs/', exist_ok=True)


# %%
plt.figure(figsize=(15, 8))
plt.plot(ascan[:, sel_el-1], label='sat')
plt.plot(bscans[:, sel_el-1, sel_el-1], label='desat')
plt.legend()
plt.title(f'ascan el: {sel_el}')

# %%
os.makedirs(data_path / 'prePrcs_modelagem/', exist_ok=True)
np.save(data_path / str('prePrcs_modelagem/' + npy_bscan), bscans)