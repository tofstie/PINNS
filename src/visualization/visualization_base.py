import os

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from src.visualization.visualization_parameters import VisualizationParameters, FigureParameters, DataParameters

class VisualizationBase:
    def __init__(self, params: VisualizationParameters):
        self.params = params
        if self.params.style == 'ggplot':
            plt.style.use('ggplot')
        plt.rc('axes',
               facecolor = '#FFFFFF',
               edgecolor = 'k',
               axisbelow = True,
               grid = False,
               labelcolor = 'k')
        plt.rcParams['axes.formatter.limits'] = (-4,4)
        plt.rcParams['axes.formatter.offset_threshold'] = 5
        plt.rc('xtick', color = 'k')
        plt.rc('ytick', color = 'k')
        plt.rc('legend', frameon = False, fontsize = 'medium')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 8
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['text.usetex'] = True
        colors = cycler(color=self.params.color_list)
        plt.rc('axes', prop_cycle=colors)

    def visualize(self):
        for figure in self.params.figures:
            if figure.type == "figure":
                fig, ax = plt.subplots(1,1,figsize = figure.figsize,dpi = figure.dpi)
                for data in figure.data:
                    if data.plot_type == "plot":
                        ax.plot(data.x, data.y, label = data.label, marker = data.marker)
                    elif data.plot_type == "scatter":
                        ax.scatter(data.x, data.y, label = data.label, marker = data.marker, s = data.marker_size)
                ax.set_ylabel(figure.ylabel[0])
                ax.set_xlabel(figure.xlabel[0])
                ax.set_yscale(figure.yscale[0])
                ax.set_xscale(figure.xscale[0])
                if figure.use_legend:
                    if figure.use_bbox:
                        ax.legend(loc = figure.legend_loc, bbox_to_anchor = figure.bbox_location)
                    else:
                        ax.legend(loc = figure.legend_loc)
            elif figure.type == "subplot":
                fig, axes = plt.subplots(figure.subplot_rows, figure.subplot_cols,figsize=figure.figsize, dpi = figure.dpi)
                for idx, data in zip(figure.figure_idx, figure.data):
                    if data.plot_type == "plot":
                        axes[idx].plot(data.x, data.y, label = data.label, marker = data.marker)
                    elif data.plot_type == "scatter":
                        axes[idx].scatter(data.x, data.y, label = data.label, marker = data.marker, s = data.marker_size)
                for idx, xlabel, xscale in zip(range(0,len(figure.xlabel)),figure.xlabel, figure.xscale):
                    axes[idx].set_xlabel(xlabel)
                    axes[idx].set_xscale(xscale)
                for idx, ylabel, yscale in zip(range(0,len(figure.ylabel)),figure.ylabel, figure.yscale):
                    axes[idx].set_ylabel(ylabel)
                    axes[idx].set_yscale(yscale)
                if figure.use_legend:
                    if figure.use_bbox:
                        axes[0].legend(loc = figure.legend_loc, bbox_to_anchor = figure.bbox_location)
                    else:
                        axes[0].legend(loc = figure.legend_loc)
        plt.show()
    def save(self, path):
        pass
