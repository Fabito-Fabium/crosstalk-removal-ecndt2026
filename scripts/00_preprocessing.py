import sys
from pathlib import Path
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
    model_h_len = config["modelling"]["h_len"]
    dtemp = config["preprocessing"]["dtemp"]
    transducer_type = config["transducer"]["type"]
    central_freq = config["transducer"]["central_freq_MHz"]
    manufacturer = config["transducer"]["manufacturer"]

# Important paths:
data_path = root / 'data'
model_path = data_path / "m2k" / "linear_imasonic_64_5MHz" / "calibration"
models_path = data_path / "models"
figures_path = root / "figures"


# %%
def ascan_prePross(el):
    path1 = model_path / f'{el+1}.m2k'

    eliso = file_m2k.read(str(path1), freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0,
                          read_ascan=True, type_insp="contact", water_path=0.0)

    ascan = eliso.ascan_data[250:, 0, :num_elem, 0]



    t_start = np.argmax(ascan[:, el]) - dtemp//2
    t_end = np.argmax(ascan[:, el]) + dtemp//2

    current = np.zeros((dtemp, ascan.shape[1]), dtype=float)
    for i_idx in range(num_elem):
        max_prim = np.max(abs(ascan[t_start:t_end, i_idx]))
        dec = Decimator(ascan[t_start:t_end, i_idx], max=max_prim)

        # current[:, i_idx] = Reconstruct(dec, ascan[t_start:t_end, i_idx], ascan[t_start:t_end, i_idx])

        desat = Reconstruct(dec, ascan[t_start:t_end, i_idx], ascan[t_start:t_end, i_idx])
        # desat = ascan[t_start:t_end, i_idx]
        # current[:, i_idx] = desat
        current[:, i_idx] = bandpass(desat, 1e6, 9e6, 300, False)

    return current

iter = num_elem
parallel_tmp = Parallel(n_jobs=-3)(delayed(ascan_prePross)(i) for i in tqdm(range(iter)))

bscans = np.zeros((dtemp, num_elem, num_elem), dtype=float)
# %%
for i in range(iter):
    bscans[:, :, i] = parallel_tmp[i]
# %%
sel_el = 50

path1 = model_path / f'{sel_el}.m2k'

eliso = file_m2k.read(str(path1), freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=None,
                         read_ascan=True, type_insp="contact", water_path=0.0)#[1]

ascan = eliso.ascan_data[600:, 0, :num_elem, 0]
t_start = np.argmax(ascan[:, sel_el-1]) - dtemp // 2
t_end = np.argmax(ascan[:, sel_el-1]) + dtemp // 2
ascan = ascan[t_start:t_end, :]
plt.figure(figsize=(15, 8))
plt.imshow(np.log10(envelope(ascan)+1e-6), aspect='auto', interpolation='nearest')
plt.show()


# %% ## tentativa de corrigir o problema na aquisição 59 no qual o sinal ideal não está no canal indicado
aux = bscans[:, :, 63-58].copy()
bscans[:, :, 58] = np.flip(aux, axis=1)
# %%
plt.figure(figsize=(15, 8))
plt.plot(ascan[:, sel_el-1], label='sat')
plt.plot(bscans[:, sel_el-1, sel_el-1], label='desat')
plt.legend()
plt.title(f'ascan el: {sel_el}')
plt.show()

# %%
os.makedirs(data_path / 'processed_data', exist_ok=True)
np.save(str(data_path / 'processed_data' / f"{transducer_type}_{manufacturer}_{num_elem:.0f}_{central_freq:.0f}MHz.npy"), bscans)