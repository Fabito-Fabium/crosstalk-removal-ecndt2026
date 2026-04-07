
# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from framework.post_proc import envelope
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
plt.ion()
# %%
def bscan_plot(bscan, title='BSCAN plot', save=None):
    plt.figure(figsize=(15, 8))
    plt.imshow(np.log10(envelope(bscan) + 1e-6), aspect='auto', interpolation='nearest')
    plt.title(title)
    if save != None:
        plt.savefig(f'{save}.png')

# %%
def impulse_response_graph_ALL(og_bscan, estimated_bscan, rt_idx, num_idx, title="title", fignum=0):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6), num=fignum)

    plt.subplots_adjust(bottom=0.2)
    fig.suptitle(title, fontsize=16, y=0.98)

    current_index = 0


    ax_left.imshow(20 * np.log10(envelope(og_bscan[:, :, current_index] + 1e-6)), vmin=-80, vmax=60, aspect='auto',
                   interpolation='nearest')
    ax_right.imshow(20 * np.log10(envelope(estimated_bscan[:, :, current_index]) + 1e-6), aspect='auto',
                    interpolation='nearest', vmin=-80, vmax=60)


    def set_initial_limits(index):

        ax_left.set_title(f'Original B-scan; el {rt_idx[index]}')
        ax_right.set_title(f'B-scan with estimated mean-value removal; el {rt_idx[index]}')


    set_initial_limits(current_index)


    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(slider_ax, 'element', 1, num_idx, valinit=1, valstep=1)

    def update(val):
        ax_right.clear()
        index = int(slider.val) - 1
        current_index = index

        if index >= og_bscan.shape[2]:
            print(f"Error: Index {index} out of bounds for data1")
            return

        ax_left.imshow(20*np.log10(envelope(og_bscan[:, :, current_index] + 1e-6)), aspect='auto', interpolation='nearest', vmin=-80, vmax=80)
        ax_right.imshow(20*np.log10(envelope(estimated_bscan[:, :, current_index]) + 1e-6), aspect='auto', interpolation='nearest', vmin=-80, vmax=80)

        set_initial_limits(current_index)
        fig.canvas.draw()


    slider.on_changed(update)

    plt.show()

def frf_graph_h(func, axes=None, N=50, x_type='f', fs=120e6):
    f = np.linspace(1e4, 120e6, 1000);
    w, Func = ss.freqz(func, [1], f, fs=fs)

    if x_type == 'w':
        xlabel = r'$\omega$ [rad/s]'
        xx = w
    else:
        xlabel = 'f [Hz]'
        xx = w / (2 * np.pi)

    if axes is None:
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 2, 3)
        ax3 = plt.subplot(2, 2, 4)
    else:
        ax1, ax2, ax3 = axes

    # Clear and replot each subplot
    ax1.clear()
    ax1.semilogx(xx, 20 * np.log10(np.abs(Func)))
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Magnitude [dB]')
    ax1.set_title(r'|$H(\omega)$|')
    ax1.grid(True)

    ax2.clear()
    ax2.stem(func)
    ax2.set_xlabel('n')
    ax2.set_ylabel('h[n]')
    ax2.set_title('Impulse Response')
    ax2.grid(True)

    ax3.clear()
    ax3.semilogx(xx, np.angle(Func))
    ax3.set_xlabel(xlabel)
    ax3.set_title(r'$\angle H(\omega)$')
    ax3.grid(True)

def frf_graph_f(func, func_true, axes=None, N=20, x_type='f', fs=120e6):
    f = np.linspace(1e4, 120e6, 1000)

    w, Func = ss.freqz(func, worN=f, fs=fs)
    _, Func_true = ss.freqz(func_true, worN=f, fs=fs)

    if x_type == 'w':
        xlabel = r'$\omega$ [rad/s]'
        xx = w
    else:
        xlabel = 'f [Hz]'
        xx = w / (2 * np.pi)

    if axes is None:
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 2, 3)
        ax3 = plt.subplot(2, 2, 4)
    else:
        ax1, ax2, ax3 = axes

    # Clear and replot each subplot
    ax1.clear()
    ax1.semilogx(xx, 20 * np.log10(np.abs(Func)), label='est')
    ax1.semilogx(xx, 20 * np.log10(np.abs(Func_true)), label='real')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Magnitude [dB]')
    ax1.set_title(r'|$G(\omega)$|')
    ax1.grid(True)
    ax1.legend()

    ax2.clear()
    ax2.plot(func, label='est')
    ax2.plot(func_true, label='real')
    ax2.set_xlabel('n')
    ax2.set_ylabel('g')
    ax2.set_title('Estimated vs Real')
    ax2.grid(True)
    ax2.legend()

    ax3.clear()
    ax3.semilogx(xx, np.angle(Func), label='est')
    ax3.semilogx(xx, np.angle(Func_true), label='real')
    ax3.set_xlabel(xlabel)
    ax3.set_title(r'$\angle G(\omega)$')
    ax3.grid(True)
    ax3.legend()

def interactive_frf(Func, N=20, x_type='f', fs=120e6, title='IIR', Func_true=None):

    # Get dimensions from Func shape
    num_el0 = Func.shape[1]  # Second axis size
    num_el1 = Func.shape[2]  # Third axis size

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    axes = (ax1, ax2, ax3)

    # Add counter display
    counter_ax = fig.add_axes([0.1, 0.02, 0.8, 0.05])
    counter_ax.axis('off')  # Hide axes frame
    counter_ax.text(0.5, 0.1, 'Navigation: ←/→ to change affected elements, ↑/↓ to change the element casting crosstalk',
                                ha='center', va='center', fontsize=10, alpha=0.7)

    # Current indices
    idx0 = 0  # For second axis
    idx1 = 0  # For third axis

    try:
        diff_norm = np.linalg.norm(Func[:, idx0, idx1] - Func_true[:, idx0, idx1])/len(Func[:, idx0, idx1])
        frf_graph_f(Func[:, idx0, idx1], Func_true[:, idx0, idx1], axes, N, x_type, fs)
        fig.suptitle(f'{title}, acquisition {idx1 + 1};  $g_{{{idx0 + 1}}}, mse: {diff_norm}$', fontsize=17)

        def update_plot():
            diff_norm = np.linalg.norm(Func[:, idx0, idx1] - Func_true[:, idx0, idx1])/len(Func[:, idx0, idx1])
            frf_graph_f(Func[:, idx0, idx1], Func_true[:, idx0, idx1], axes, N, x_type, fs)
            fig.suptitle(f'{title}, acquisition {idx1 + 1};  $g_{{{idx0 + 1}}}, mse =  {diff_norm}$', fontsize=17)
            fig.canvas.draw_idle()
    except:
        frf_graph_h(Func[:, idx0, idx1], axes, N, x_type, fs)
        fig.suptitle(f'{title}, $h_{{{idx0 + 1}, {idx1 + 1}}}$', fontsize=20)

        def update_plot():
            """Update plot with current indices"""
            frf_graph_h(Func[:, idx0, idx1], axes, N, x_type, fs)
            fig.suptitle(f'{title}, $h_{{{idx0 + 1}, {idx1 + 1}}}$', fontsize=20)
            fig.canvas.draw_idle()


    # Keyboard navigation function
    def on_key(event):
        nonlocal idx0, idx1
        update_needed = False

        if event.key == 'right':
            if idx0 < num_el0 - 1:
                idx0 += 1
                update_needed = True
            else:
                idx0 = -1

        elif event.key == 'left':
            if idx0 > 0:
                idx0 -= 1
                update_needed = True
            else:
                idx0 = num_el1
        elif event.key == 'up':
            if idx1 < num_el1 - 1:
                idx1 += 1
                update_needed = True
            else:
                idx1 = -1
        elif event.key == 'down':
            if idx1 > 0:
                idx1 -= 1
                update_needed = True
            else:
                idx1 = num_el1

        if update_needed:
            update_plot()

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()
