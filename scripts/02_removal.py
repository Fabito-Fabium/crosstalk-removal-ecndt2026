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
#%% Load important constants:
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

    # Removal parameters:
    removal_solver = config["removal"]["solver"]["type"]
    lmbd_1 = config["removal"]["solver"]["lmbd_1"]
    niter_1 = config["removal"]["solver"]["niter_1"]

    model_name = f"model_{model_type}_{h_len}"

    transducer_name = f"{transducer_type}_{manufacturer}_{num_elem}_{central_freq}MHz"
# Important paths:
data_path = root / 'data'
models_path = data_path / "models"
figures_path = root / "figures"
result_path = data_path / "results"
test_filename = Path(config['removal']['filename']).with_suffix("")
test_m2k = root / Path(config['removal']['filepath']) / test_filename
bscan = np.load(data_path / "processed_data" / (str(output_filename) + ".npy"), allow_pickle=True)


models = np.load(models_path / transducer_name / str(model_name + ".npz"), allow_pickle=True)
selected_model_1 = models[model_type]
# %%

path = models_path / output_filename
os.makedirs(path, exist_ok=True)

result_model_path = result_path / transducer_name
model_path = result_path / transducer_name

try:
    og_asc = np.load(result_model_path / test_filename / str(model_name + '_og_ascan.npy'))
except FileNotFoundError:
    data_insp = file_m2k.read(str(test_m2k.with_suffix(".m2k")),
                              freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=None, read_ascan=True, type_insp="contact", water_path=0.0)
    eliso = data_insp[1] if isinstance(data_insp, list) else data_insp

    bscan = eliso.ascan_data[:, 0, :num_elem, 0]


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
    bscans = Parallel(n_jobs=-2)(delayed(bscan_prePross)(i) for i in tqdm(range(num_elem)))

    og_asc = np.sum(bscans, axis=0)


    # og_asc = bscan[:, :, 0]

    plt.figure(figsize=(15, 8))
    plt.imshow(np.log10(envelope(og_asc)+1e-6), aspect='auto', interpolation='nearest')
    plt.show()

    np.save(result_model_path / test_filename / str(model_name + '_og_ascan.npy'), og_asc)
    np.save(result_model_path / test_filename / str(model_name + '_og_bscan.npy'), bscans)

Nt = og_asc.shape[0]
Ne = og_asc.shape[1]

# %% loop sel 1

acq_PATH = result_path / transducer_name / test_filename

# md1_PATH = acq_PATH / f'{model_type[:6]}/' / removal_solver + f'/{lmbd_1:.2e}/'
# os.makedirs(md1_PATH, exist_ok=True)

metric_name = str(model_name) + "_" + f"{removal_solver}_lmbd{lmbd_1}_niter{niter_1}"
metrics_path = result_model_path / test_filename / (metric_name + '_metrics.h5')
if os.path.isfile(metrics_path):
    for metric_ in ["current_x0.npy", "metrics.h5", "r1norm.npy", "solve_return.npy", "x_best_cnr.npy", "x_best_cr.npy", "x_best_sinr.npy"]:
        os.remove(result_model_path / test_filename / (metric_name + "_" + metric_))

mask_path = data_path / "masks" / transducer_name / test_filename / 'masks.npz'

# simmilar_results = np.load(acq_PATH / (metric_name

if not (os.path.isfile(metrics_path)):
    if os.path.isfile(mask_path):
        cr_x, cnr_x, sinr_x, _, _, _ = get_metrics(og_asc, str(mask_path))

        with h5py.File(metrics_path, 'a') as f:
            for name, data in [('cnr', cnr_x), ('cr', cr_x), ('sinr', sinr_x)]:
                if name in f:
                    del f[name]
                f.create_dataset(name, data=data)
    else:
        print('There is no selected masks...')

for i in range(1):
    try:
        x_1 = np.load(acq_PATH / (metric_name + '_solve_return.npy'))
        ##
    except FileNotFoundError:
        x_1 = og_asc.ravel()

    f_mtx_ = solve_mthd(selected_model_1, removal_solver, og_asc, Nt, Ne, x0=x_1, iscoefs=('coefs' in model_type), niter=niter_1,lmbd=lmbd_1,
                        callback=lambda x: sv_model(x, Ne, str(acq_PATH / str(metric_name + "_")), str(mask_path)))

    np.save(acq_PATH / (metric_name + '_solve_return.npy'), f_mtx_[0])
    try:
        r1norm = np.load(mask_path / (metric_name + '_r1norm.npy'))
        r1norm = np.concatenate([r1norm, f_mtx_[-1]])
    except:
        r1norm =  f_mtx_[-1]

    np.save(acq_PATH / (metric_name + '_r1norm.npy'), r1norm)