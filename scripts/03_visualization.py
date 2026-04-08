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
metric_name = str(model_name) + "_" + f"{removal_solver}_lmbd{lmbd_1}_niter{niter_1}"

data_insp = file_m2k.read(str(root / 'data' / 'm2k' / transducer_name / 'test' / test_filename.with_suffix(".m2k")), freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=None,
                          read_ascan=True, type_insp="contact", water_path=0.0)
eliso = data_insp[1] if isinstance(data_insp, list) else data_insp

niter = len(np.load(test_result / str(metric_name + '_r1norm.npy')))
# %%
est_1 = np.load(test_result / str(metric_name + '_current_x0.npy'))[:-1].reshape((-1, num_elem))

est_cr = np.load(test_result / str(metric_name + '_x_best_cr.npy')).reshape((-1, num_elem))
est_cnr = np.load(test_result / str(metric_name + '_x_best_cnr.npy')).reshape((-1, num_elem))
est_sinr = np.load(test_result / str(metric_name + '_x_best_sinr.npy')).reshape((-1, num_elem))

md_list = [np.log10(envelope(og_ascans)+1e-6), np.log10(envelope(est_1)+1e-6),
           np.log10(envelope(est_cr)+1e-6), np.log10(envelope(est_cnr)+1e-6),
           np.log10(envelope(est_sinr)+1e-6)]

vmax = np.max(md_list)
vmin = np.min(md_list)
# %%
plt.figure(figsize=(4, 5))
plt.imshow(np.log10(envelope(og_ascans)+1e-6), aspect='auto', interpolation='nearest', vmax=vmax, vmin=vmin, extent=[1, 65, eliso.time_grid.max(), eliso.time_grid.min()])
plt.colorbar()
plt.ylabel(r"Time / $(\mu s)$")
plt.xlabel("Channels")
plt.tight_layout()
# %%
plt.figure(figsize=(4, 2))
pitch = .5
x_transd = np.arange(64) * pitch
x_transd -= np.mean(x_transd)
plt.imshow(np.log10(envelope(est_1)+1e-6), interpolation='nearest', vmax=vmax, vmin=vmin, extent=[x_transd[0], x_transd[-1], eliso.time_grid.max()*1.483/2, eliso.time_grid.min()*1.483/2])
# plt.colorbar()
plt.ylabel(r"mm")
plt.xlabel("mm")
plt.tight_layout()
# plt.savefig(figures_path / 'fig_11.pdf')
# %%
from matplotlib.lines import Line2D

def plot_as_in_ext_abstract(bscan):
    try:
        aux = np.load(mask_path / 'masks.npz')
        mask = aux['signal']
        crs_mask = aux['cross']
        noise_mask = aux['noise']

        img_extent = [1, 65, eliso.time_grid.max(), eliso.time_grid.min()]

        plt.figure(figsize=(0.8*4, 0.8*5))
        plt.imshow(np.log10(envelope(bscan) + 1e-6), aspect='auto', interpolation='nearest',
                   vmax=vmax, vmin=vmin, extent=img_extent)
        plt.colorbar()

        colors = ['red', 'blue', 'orange']
        linewidth = 1.5

        plt.contour(mask, levels=[0.5], colors=colors[0], linewidths=linewidth,
                    extent=img_extent, origin='upper')

        plt.contour(crs_mask, levels=[0.5], colors=colors[1], linewidths=linewidth,
                    extent=img_extent, origin='upper')

        plt.contour(noise_mask, levels=[0.5], colors=colors[2], linewidths=linewidth,
                    extent=img_extent, origin='upper')

        legend_elements = [
            Line2D([0], [0], color=colors[0], lw=2, label='Signal'),
            Line2D([0], [0], color=colors[1], lw=2, label='Crosstalk'),
            Line2D([0], [0], color=colors[2], lw=2, label='Noise')
        ]
        plt.legend(handles=legend_elements, loc='lower left')

        plt.ylabel(r"Time / $(\mu s)$")
        plt.xlabel("Channels")
        plt.tight_layout()
    except:
        print('Não foram definidas as mascaras na presente aquisição')
# %%
plot_as_in_ext_abstract(og_ascans)
plt.savefig(figures_path / 'fig5b.pdf')

plot_as_in_ext_abstract(est_1)
plt.savefig(figures_path / 'fig5c.pdf')
plt.show(block='False')
# %%
cr_all, cnr_all, sinr_all = np.array([get_metrics(aux, str(mask_path) + "/masks.npz")[:3] for aux in [og_ascans, est_1]]).T

titles = ['CR \t\t[dB]' , 'CNR_mod [dB]', 'SINR \t[dB]']
labels = ['Og', model_type[:3]]

metrics = 10*np.log10([cr_all, cnr_all, sinr_all])

for idx_i, title in enumerate(titles):
    print('-----------------')
    print(f'{title}:')
    for idx_j, label in enumerate(labels):
        print(f'{label}: \t{metrics[idx_i][idx_j]}')

    print(f'diff: \t{metrics[idx_i][1] - metrics[idx_i][0]}')

# [msk_sig, msk_crs, msk_noise]
mean_all = np.array([get_metrics(aux, str(mask_path) + "/masks.npz")[3] for aux in [og_ascans, est_1]])
std_all = np.array([get_metrics(aux, str(mask_path) + "/masks.npz")[4] for aux in [og_ascans, est_1]])
L2_all = np.array([get_metrics(aux, str(mask_path) + "/masks.npz")[5] for aux in [og_ascans, est_1]])
