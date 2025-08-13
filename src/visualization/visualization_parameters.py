class VisualizationParameters:
    def __init__(self):
        self.color_list = [
            '#ED1B2F',
            '#b9b4b3',
            '#1896cb',
            '#B2D235'
        ]
        self.style = 'ggplot'

        self.figures = []

    def set_color_cycle(self,list_of_colors):
        self.color_list = list_of_colors

class FigureParameters:
    def __init__(self):
        # Figure Settings
        self.type = "figure"
        self.figsize = (3,3)
        self.dpi = 600
        self.subplot_rows = 0
        self.subplot_cols = 0

        # Legend Settings
        self.use_legend = True
        self.legend_loc = 'upper right'
        self.use_bbox = False
        self.bbox_location = (0,0)
        self.bbox_inches = 'tight'

        # Axes Settings
        self.ylabel = []
        self.xlabel = []
        self.xscale = []
        self.yscale = []

        # Data Settings
        self.data = []
        self.figure_idx = []

        # Plotting Settings
    def add_data(self, data, figure_idx):
        self.data.append(data)
        self.figure_idx.append(figure_idx)

    def add_x_axes_settings(self, xlabel, xscale = "linear"):
        self.xlabel.append(xlabel)
        self.xscale.append(xscale)

    def add_y_axes_settings(self, ylabel, yscale = "linear"):
        self.ylabel.append(ylabel)
        self.yscale.append(yscale)

class DataParameters:
    def __init__(self, x = None, y = None, z = None, label = "", marker = "", plot_type = "plot"):
        # Data
        self.x = x
        self.y = y
        self.z = z
        # Styling
        self.label = label
        self.marker = marker
        self.plot_type = plot_type
        self.marker_size = 1

    def set_2D_data(self, x, y, label, plot_type = "plot", marker = ""):
        self.x = x
        self.y = y
        self.label = label
        self.plot_type = plot_type
        self.marker = marker

    def set_3D_data(self, x, y, z, label, plot_type = "scatter", marker = ""):
        self.x = x
        self.y = y
        self.z = z
        self.label = label
        self.plot_type = plot_type
        self.marker = marker

