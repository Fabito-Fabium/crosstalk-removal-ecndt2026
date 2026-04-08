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


figures_path = root / "figures"
test_filename = Path(config['removal']['filename']).with_suffix("")
bscan = np.load(root / "data" / "processed_data" / (str(output_filename) + ".npy"), allow_pickle=True)
og_ascans = np.load(root / "data" / "results" / transducer_name / test_filename / str(model_name + "_og_ascan.npy"), allow_pickle=True)
test_result = root / "data" / "results" / transducer_name / test_filename
mask_path = root / 'data' / 'masks' / transducer_name / test_filename

models = np.load(root / "data" / "models" / transducer_name / str(model_name + ".npz"), allow_pickle=True)
selected_model_1 = models[model_type]


#%%
def plotMetrics_per_iter(metric ,title='N/D', ylabel='N/D', logscale=True):
    plt.figure(figsize=(3*1.2, 2*1.2))
    if logscale:
        plt.semilogy(np.arange(len(metric)) + 1, metric, linewidth=2, color='black')
        plt.yticks([10e1, 10e2, 10e3, 10e4], minor=True)
        plt.yticks()
    else:
        plt.plot(np.arange(len(metric)) + 1, metric, linewidth=2, color='black')
    plt.xlabel('Number of Iterations')
    plt.ylabel(ylabel)
    plt.grid(True, which="both")
    # plt.title(title)
    plt.tight_layout()
    plt.show()

metric_name = str(model_name) + "_" + f"{removal_solver}_lmbd{lmbd_1}_niter{niter_1}"

r1norm_1 = (np.load(test_result / str(metric_name + '_r1norm.npy'))[:])
title = f'$||r||_2$ per iteration, \t min: $\\approx${np.min(r1norm_1):.4f}'
plotMetrics_per_iter(r1norm_1, title, ylabel=fr'$\log(||r||_2)$')

plt.savefig(figures_path / 'fig_02.pdf')
# %%
with h5py.File(test_result / str(metric_name + '_metrics.h5'), 'r') as f:
    cnr_arr = np.array(f['cnr'])  # Works for both scalars and arrays
    cr_arr = np.array(f['cr'])
    sinr_arr = np.array(f['sinr'])

argmax_metric = np.argmax([cnr_arr, cr_arr, sinr_arr], axis=1)
# %%
plotMetrics_per_iter(10*np.log10(cnr_arr), title=f'CNR [dB] per iteration, argmax: {argmax_metric[0]}'
                     , ylabel=f'CNR / (dB)', logscale=False)
plt.savefig(figures_path / 'fig_03.pdf')

plotMetrics_per_iter(10*np.log10(cr_arr), title=f'CR [dB] per iteration, argmax: {argmax_metric[1]}'
                     , ylabel=f'CR / (dB)', logscale=False)
plt.savefig(figures_path / 'fig_04.pdf')

plotMetrics_per_iter(10*np.log10(sinr_arr), title=f'SINR [dB] per iteration, argmax: {argmax_metric[2]}'
                     , ylabel=f'SINR / (dB)', logscale=False)
plt.savefig(figures_path / 'fig_05.pdf')
# %%

est_cr = np.load(test_result / str(metric_name + '_x_best_cr.npy')).reshape((-1, num_elem))
est_cnr = np.load(test_result / str(metric_name + '_x_best_cnr.npy')).reshape((-1, num_elem))
est_sinr = np.load(test_result / str(metric_name + '_x_best_sinr.npy')).reshape((-1, num_elem))

md_list = [np.log10(envelope(est_cnr)+1e-6), np.log10(envelope(est_cr)+1e-6),
           np.log10(envelope(est_sinr)+1e-6)]

vmax = np.max(md_list)
vmin = np.min(md_list)
# %%
data_insp = file_m2k.read(str(root / 'data' / 'm2k' / transducer_name / 'test' / test_filename.with_suffix(".m2k")), freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=None,
                          read_ascan=True, type_insp="contact", water_path=0.0)
eliso = data_insp[1] if isinstance(data_insp, list) else data_insp

plt.figure(figsize=(0.8*10, 0.8*6))

plt.subplot(1, 3, 1)
plt.imshow(np.log10(envelope(est_cnr)+1e-6), aspect='auto', interpolation='nearest', vmax=vmax, vmin=vmin,
           extent=[1, 65, eliso.time_grid.max(), eliso.time_grid.min()])

plt.title(f'best cnr, niter= {argmax_metric[0]}')
plt.colorbar()
plt.ylabel(r"Time / $(\mu s)$")
plt.xlabel("Channels")
plt.tight_layout()

plt.subplot(1, 3, 2)
plt.imshow(np.log10(envelope(est_cr)+1e-6), aspect='auto', interpolation='nearest', vmax=vmax, vmin=vmin,
           extent=[1, 65, eliso.time_grid.max(), eliso.time_grid.min()])

# plt.title('Remoção sem regularização')
plt.title(f'best cr, niter= {argmax_metric[1]}')
plt.colorbar()
plt.xlabel("Channels")
plt.tight_layout()

plt.subplot(1, 3, 3)
plt.imshow(np.log10(envelope(est_sinr)+1e-6), aspect='auto', interpolation='nearest', vmax=vmax, vmin=vmin,
           extent=[1, 65, eliso.time_grid.max(), eliso.time_grid.min()])

# plt.title('Remoção sem regularização')
plt.title(f'best sinr, niter= {argmax_metric[2]}')
plt.colorbar()
plt.xlabel("Channels")
plt.tight_layout()

plt.savefig(figures_path / f'fig_06.pdf')