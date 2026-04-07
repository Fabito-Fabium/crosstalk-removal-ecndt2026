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

#%%

data_insp = file_m2k.read(path1, freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=None,
                          read_ascan=True, type_insp="contact", water_path=0.0)
eliso = data_insp[1] if isinstance(data_insp, list) else data_insp


niter = len(np.load(md1_PATH + 'r1norm.npy'))
# %%
og_asc = np.load(acqq_PATH + 'og_bscan.npy')
est_1 = np.load(md1_PATH + 'current_x0.npy')[:-1].reshape((-1, NUM_ELEM))

est_cr = np.load(md1_PATH + 'x_best_cr.npy').reshape((-1, NUM_ELEM))
est_cnr = np.load(md1_PATH + 'x_best_cnr.npy').reshape((-1, NUM_ELEM))
est_sinr = np.load(md1_PATH + 'x_best_sinr.npy').reshape((-1, NUM_ELEM))

md_list = [np.log10(envelope(og_asc)+1e-6), np.log10(envelope(est_1)+1e-6),
           np.log10(envelope(est_cr)+1e-6), np.log10(envelope(est_cnr)+1e-6),
           np.log10(envelope(est_sinr)+1e-6)]

vmax = np.max(md_list)
vmin = np.min(md_list)
# %%
plt.figure(figsize=(4, 5))
plt.imshow(np.log10(envelope(og_asc)+1e-6), aspect='auto', interpolation='nearest', vmax=vmax, vmin=vmin, extent=[1, 65, eliso.time_grid.max(), eliso.time_grid.min()])
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
plt.savefig(figures_path / 'fig_11.pdf')
# %%
from matplotlib.lines import Line2D

def plot_as_in_ext_abstract(bscan):
    try:
        aux = np.load(msk_PATH + 'masks.npz')
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
plot_as_in_ext_abstract(og_asc)
plt.savefig(figures_path / 'fig_00.pdf')
plot_as_in_ext_abstract(est_1)
plt.savefig(figures_path / 'fig_01.pdf')
plt.show(block='False')
# %%
cr_all, cnr_all, sinr_all = np.array([get_metrics(aux, msk_PATH)[:3] for aux in [og_asc, est_1]]).T

titles = ['CR \t\t[dB]' , 'CNR_mod [dB]', 'SINR \t[dB]']
labels = ['Og', MODEL_TYPE[:3]]

metrics = 10*np.log10([cr_all, cnr_all, sinr_all])

for idx_i, title in enumerate(titles):
    print('-----------------')
    print(f'{title}:')
    for idx_j, label in enumerate(labels):
        print(f'{label}: \t{metrics[idx_i][idx_j]}')

    print(f'diff: \t{metrics[idx_i][1] - metrics[idx_i][0]}')

# [msk_sig, msk_crs, msk_noise]
mean_all = np.array([get_metrics(aux, msk_PATH)[3] for aux in [og_asc, est_1]])
std_all = np.array([get_metrics(aux, msk_PATH)[4] for aux in [og_asc, est_1]])
L2_all = np.array([get_metrics(aux, msk_PATH)[5] for aux in [og_asc, est_1]])
