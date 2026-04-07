from pathlib import Path
import matplotlib

matplotlib.use('TkAgg')

from crsLib.graphing import *
from crsLib.estimation import *
from crsLib.desatFunc import *
import json

DTYPE = np.float64
#%% Load important constants:
with open('config.json', 'r') as file:
    config = json.load(file)

    # Problem-related constants:
    NUM_ELEM = config["preprocessing"]["num_elem"]
    H_LEN_FIR = config["modelling"]["h_len_fir"]
    DTEMP = config["preprocessing"]["dtemp"]

    # Important filenames:
    npz_name = config["modelling"]["npz_name_template"].replace("{entry}", str(H_LEN_FIR))
    npy_bscan = config["preprocessing"]["npy_bscan_filename"]

    # Removal parameters:
    removal_solver = config["removal"]["solver"]
    model_type = config["removal"]["model_type"]
    # iscoefs_1 = config["removal"]["iscoefs_1"]
    lmbd_1 = config["removal"]["lmbd_1"]
    niter_1 = config["removal"]["niter_1"]

# Important paths:
root = Path(__file__).parent.parent
data_path = root / 'data'
model_path = data_path / "m2k" / "linear_imasonic" / "model"
models_path = data_path / "models"
figures_path = root / "figures"

all_ascan = np.load(data_path / str('prePrcs_modelagem/' + npy_bscan), allow_pickle=True)


models = np.load(models_path / npz_name, allow_pickle=True)
selected_model_1 = models[model_type]
# %%

os.makedirs(acqq_PATH, exist_ok=True)
og_asc = np.load(acqq_PATH + 'og_bscan.npy')
data_insp = file_m2k.read(path1, freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=None,
                          read_ascan=True, type_insp="contact", water_path=0.0)
eliso = data_insp[1] if isinstance(data_insp, list) else data_insp

bscan = eliso.ascan_data[:, 0, :NUM_ELEM, 0]


def bscan_prePross(idx):
    og_asc = np.zeros_like(bscan)
    max_prim = np.max(abs(bscan[:, idx]))
    dec = Decimator(bscan[:, idx], max=max_prim)

    rec = Reconstruct(dec, bscan[:, idx], bscan[:, idx])
    # og_asc[:, idx] = bandpass(rec, 1e6, 10e6, 300, False)
    og_asc[:, idx] = rec
    return og_asc
#
#
bscans = Parallel(n_jobs=-2)(delayed(bscan_prePross)(i) for i in tqdm(range(NUM_ELEM)))

og_asc = np.sum(bscans, axis=0)


# og_asc = bscan[:, :, 0]

plt.figure(figsize=(15, 8))
plt.imshow(np.log10(envelope(og_asc)+1e-6), aspect='auto', interpolation='nearest')
plt.savefig(acqq_PATH + 'og_bscan.png')

plt.close('all')
np.save(acqq_PATH + 'og_bscan.npy', og_asc)

Nt = og_asc.shape[0]
Ne = og_asc.shape[1]

# %% loop sel 1
md1_PATH = acq_PATH + f'{model_type[:6]}/' + removal_solver + '/{:.2e}/'.format(lmbd_1)
os.makedirs(md1_PATH, exist_ok=True)

metrics_PATH = md1_PATH + 'metrics.h5'

if not (os.path.isfile(metrics_PATH)):
    if os.path.isfile(msk_PATH + 'masks.npz'):
        cr_x, cnr_x, sinr_x, _, _, _ = get_metrics(og_asc, msk_PATH)

        with h5py.File(metrics_PATH, 'a') as f:
            for name, data in [('cnr', cnr_x), ('cr', cr_x), ('sinr', sinr_x)]:
                if name in f:
                    del f[name]
                f.create_dataset(name, data=data)
    else:
        print('There is no selected masks...')

for i in range(1):
    try:
        x_1 = np.load(md1_PATH + 'solve_return.npy')
    except:
        x_1 = og_asc.ravel()

    f_mtx_ = solve_mthd(selected_model_1, removal_solver, og_asc, Nt, Ne, x0=x_1, iscoefs=('coefs' in model_type), niter=niter_1,lmbd=lmbd_1,
                        callback=lambda x: sv_model(x, Ne, md1_PATH, msk_PATH))

    np.save(md1_PATH + 'solve_return.npy', f_mtx_[0])
    try:
        r1norm = np.load(md1_PATH + 'r1norm.npy')
        r1norm = np.concatenate([r1norm, f_mtx_[-1]])
    except:
        r1norm =  f_mtx_[-1]

    np.save(md1_PATH + 'r1norm.npy', r1norm)