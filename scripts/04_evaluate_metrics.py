from .__config import *
# %%
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


r1norm_1 = (np.load(md1_PATH + 'r1norm.npy')[:])
title = f'$||r||_2$ per iteration, \t min: $\\approx${np.min(r1norm_1):.4f}'
plotMetrics_per_iter(r1norm_1, title, ylabel=fr'$\log(||r||_2)$')

plt.savefig(f'{rm_dir}/fig_02.pdf')
# %%
with h5py.File(md1_PATH + 'metrics.h5', 'r') as f:
    cnr_arr = np.array(f['cnr'])  # Works for both scalars and arrays
    cr_arr = np.array(f['cr'])
    sinr_arr = np.array(f['sinr'])

argmax_metric = np.argmax([cnr_arr, cr_arr, sinr_arr], axis=1)
# %%
plotMetrics_per_iter(10*np.log10(cnr_arr), title=f'CNR [dB] per iteration, argmax: {argmax_metric[0]}'
                     , ylabel=f'CNR / (dB)', logscale=False)
plt.savefig(f'{rm_dir}/fig_03.pdf')

plotMetrics_per_iter(10*np.log10(cr_arr), title=f'CR [dB] per iteration, argmax: {argmax_metric[1]}'
                     , ylabel=f'CR / (dB)', logscale=False)
plt.savefig(f'{rm_dir}/fig_04.pdf')

plotMetrics_per_iter(10*np.log10(sinr_arr), title=f'SINR [dB] per iteration, argmax: {argmax_metric[2]}'
                     , ylabel=f'SINR / (dB)', logscale=False)
plt.savefig(f'{rm_dir}/fig_05.pdf')
# %%
est_cr = np.load(md1_PATH + 'x_best_cr.npy').reshape((-1, num_el))
est_cnr = np.load(md1_PATH + 'x_best_cnr.npy').reshape((-1, num_el))
est_sinr = np.load(md1_PATH + 'x_best_sinr.npy').reshape((-1, num_el))

md_list = [np.log10(envelope(est_cnr)+1e-6), np.log10(envelope(est_cr)+1e-6),
           np.log10(envelope(est_sinr)+1e-6)]

vmax = np.max(md_list)
vmin = np.min(md_list)
# %%
try:
    eliso = file_m2k.read(path1, freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=None,
                          read_ascan=True, type_insp="contact", water_path=0.0)[1]
except:
    eliso = file_m2k.read(path1, freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=None,
                          read_ascan=True, type_insp="contact", water_path=0.0)

plt.figure(figsize=(0.8*10, 0.8*6))

plt.subplot(1, 3, 1)
plt.imshow(np.log10(envelope(est_cnr)+1e-6), aspect='auto', interpolation='nearest', vmax=vmax, vmin=vmin
           , extent=[1, 65, eliso.time_grid.max(), eliso.time_grid.min()])

plt.title(f'best cnr, niter= {argmax_metric[0]}')
plt.colorbar()
plt.ylabel(r"Time / $(\mu s)$")
plt.xlabel("Channels")
plt.tight_layout()

plt.subplot(1, 3, 2)
plt.imshow(np.log10(envelope(est_cr)+1e-6), aspect='auto', interpolation='nearest', vmax=vmax, vmin=vmin
           , extent=[1, 65, eliso.time_grid.max(), eliso.time_grid.min()])

# plt.title('Remoção sem regularização')
plt.title(f'best cr, niter= {argmax_metric[1]}')
plt.colorbar()
plt.xlabel("Channels")
plt.tight_layout()

plt.subplot(1, 3, 3)
plt.imshow(np.log10(envelope(est_sinr)+1e-6), aspect='auto', interpolation='nearest', vmax=vmax, vmin=vmin
           , extent=[1, 65, eliso.time_grid.max(), eliso.time_grid.min()])

# plt.title('Remoção sem regularização')
plt.title(f'best sinr, niter= {argmax_metric[2]}')
plt.colorbar()
plt.xlabel("Channels")
plt.tight_layout()

plt.savefig(f'{rm_dir}/fig_06.pdf')