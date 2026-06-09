import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from daltonlens import simulate


def fig_to_rgb_array(fig):
    fig.canvas.draw()

    # get RGBA buffer (new matplotlib way)
    buf = np.asarray(fig.canvas.buffer_rgba())

    # drop alpha channel
    rgb = buf[..., :3]

    return rgb


def rgb_array_to_fig(img_array, title=None):
    fig, ax = plt.subplots()
    ax.imshow(img_array)
    if title:
        ax.set_title(title)
    ax.axis("off")
    return fig


def simulate_plot_with_daltonlens(plot_func, severity=1.0):
    """
    Render a matplotlib plot under different color vision deficiencies
    using DaltonLens (Machado 2009 model).
    """

    # original plot
    fig = plot_func()
    img = fig_to_rgb_array(fig)

    sim = simulate.Simulator_Machado2009()

    variants = {
        #"normal": img,
        "deuteranopia": sim.simulate_cvd(img, simulate.Deficiency.DEUTAN, severity),
        "protanopia": sim.simulate_cvd(img, simulate.Deficiency.PROTAN, severity),
        "tritanopia": sim.simulate_cvd(img, simulate.Deficiency.TRITAN, severity),
    }

    # display grid
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, (name, vimg) in zip(axes, variants.items()):
        ax.imshow(vimg)
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.show()