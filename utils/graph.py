import pandas as pd
import numpy as np
from utils import utils as pp
import matplotlib.pyplot as plt

# credit to @ImportanceOfBeingErnest
def adjust_title(ax):
    title = ax.title
    ax.figure.canvas.draw()
    def _get_t():
        ax_width = ax.get_window_extent().width
        ti_width = title.get_window_extent().width
        return ax_width/ti_width

    while _get_t() <= 1 and title.get_fontsize() > 1:
        title.set_fontsize(title.get_fontsize()-1)

