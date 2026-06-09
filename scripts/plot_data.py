import pandas as pd
import numpy as np
import re
import os
import random

from os.path import join

from bokeh.io import show, save, output_file
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool, LinearAxis, Range1d, BoxAnnotation, Legend
from bokeh.plotting import figure

def parse_list_timestamps(str_ts_list, tz='UTC'):
    """
    Convert a list of pandas timestamps represented as a string to a
    readable format i.e. pandas timestamp object
    """
    # convert string dates to readable time stamps
    fault_dates = str_ts_list.strip('][')
    fault_dates = re.split(r'(?<=\)\)), ', fault_dates)
    f_dates = [re.split(r'(?<=\)), ', ts[1:-1]) for ts in fault_dates]

    f_dts = [pd.to_datetime(ts, format="Timestamp('%Y-%m-%d %H:%M:%S%z', tz='{}')".format(tz)) for ts in f_dates]

    return f_dts


def plot_vav_data(df_vav=None, csv_path=None, fault_dates=None, file_name='', fig_folder='./', plt_adj_w=1, plt_adj_h=1):
    """
    Plot VAV data from mortar
    """

    if df_vav is not None:
        # use passed dataframe
        vlv_dat = df_vav.copy()

        # remove timezone info for plotting
        vlv_dat.index = vlv_dat.index.tz_localize(None)
    elif vlv_dat is not None:
        # read csv
        vlv_dat = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Rename columns for consistency
    col_rename = {
        'vav-Discharge_Air_Flow_Sensor': 'air_flow',
        'vav-Discharge_Air_Temperature_Sensor': 'dnstream_ta',
        'vav-Damper_Position_Sensor': 'dmp_po',
        'room-Zone_Air_Temperature_Sensor': 'room_ta',
        'oat-Outside_Air_Temperature_Sensor': 'oat',
        'ahu-Discharge_Air_Temperature_Sensor': 'upstream_ta'
        }
    
    vlv_dat = vlv_dat.rename(columns=col_rename)

    # calculate temperature difference
    if 'temp_diff' not in vlv_dat.columns:
        vlv_dat['temp_diff'] = vlv_dat['dnstream_ta'] - vlv_dat['upstream_ta']


    # define plot data and parameters
    y_overlimit = 0.05

    left_y_max = np.ceil(vlv_dat.loc[:, ['room_ta', 'oat', 'upstream_ta', 'dnstream_ta']].max().max())
    left_y_min = np.floor(vlv_dat.loc[:, ['room_ta', 'oat', 'upstream_ta', 'dnstream_ta']].min().min())

    right_sec_y_max = np.ceil(vlv_dat['temp_diff'].max())
    right_sec_y_min = np.floor(vlv_dat['temp_diff'].min())

    sub_cols = ['upstream_ta', 'dnstream_ta', 'dmp_po', 'temp_diff', 'room_ta', 'oat']
    if 'air_flow' in vlv_dat.columns:
        right_y_max = np.ceil(vlv_dat['air_flow'].max())
        right_y_min = np.floor(vlv_dat['air_flow'].min())
        sub_cols.append('air_flow')

    subset_dat = vlv_dat.loc[:, sub_cols]
    index_name = subset_dat.index.name

    if index_name == "" or index_name is None:
        index_name = "Time"
        subset_dat.index.name = index_name

    src = ColumnDataSource(subset_dat)

    # make the plot
    max_date_idx = min(480, len(vlv_dat.index)-1)
    p = figure(plot_height=300*plt_adj_h, plot_width=800*plt_adj_w, tools='xpan', toolbar_location=None,
                x_axis_type='datetime', x_axis_location='above',
                x_range=(vlv_dat.index[0], vlv_dat.index[max_date_idx]),
                y_range = Range1d(start=left_y_min*(1-y_overlimit), end=left_y_max*(1+y_overlimit)),
                background_fill_color='#ffffff'
                )
    p.yaxis.axis_label = 'Temperature [F]'

    # highlight problem areas
    if fault_dates is not None:
        fault_hilight = []
        for ts in fault_dates:
            box_ann = BoxAnnotation(left=ts[0], right=ts[1], fill_color='#db8370', fill_alpha=0.15)
            fault_hilight.append(box_ann)
            p.add_layout(box_ann)

    # line plots
    p.step(index_name, 'upstream_ta', source=src, color='#7093db', line_width=2, legend_label="Upstream temp")
    p.step(index_name, 'dnstream_ta', source=src, color='#db7093', line_width=2, legend_label="Downstream temp")


    # add room and oat temp if available
    if 'room_ta' in vlv_dat.columns:
        p.step(index_name, 'room_ta', source=src, color='#e28a10', line_width=2, line_dash='2 2', legend_label="Room temp")
    if 'oat' in vlv_dat.columns:
        p.step(index_name, 'oat', source=src, color="#d1e210", line_width=2, line_dash='1 1', legend_label="OAT")


    if 'air_flow' in vlv_dat.columns:
        p.extra_y_ranges = {"dmpPos": Range1d(start=-1, end=101),
                            "airFlow": Range1d(start=right_y_min*(1-y_overlimit), end=right_y_max*(1+y_overlimit))
                            }
        p.add_layout(LinearAxis(y_range_name='airFlow', axis_label='Air flow [cfm]'), 'right')
        p.step(index_name, 'air_flow', source=src, color='#93db70', y_range_name='airFlow', line_width=2, legend_label="Airflow rate")
    else:
        p.extra_y_ranges = {"dmpPos": Range1d(start=-1, end=101)}

    # add valve data
    p.add_layout(LinearAxis(y_range_name='dmpPos', axis_label='Valve position [%]'), 'left')
    p.step(index_name, 'dmp_po', source=src, color='#9a9a9a', line_width=0.5, line_dash='4 4', y_range_name='dmpPos', legend_label="Damper position")

    # add legend
    p.add_layout(Legend(), 'right')
    p.legend.click_policy = "hide"

    p.legend.label_text_font_size = "9px"
    p.legend.label_height = 5
    p.legend.glyph_height = 5
    p.legend.spacing = 5

    # range selector tool
    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select = figure(title="",
                    plot_height=100*plt_adj_h, plot_width=800*plt_adj_w, 
                    y_range=Range1d(start=-1, end=101),
                    x_axis_type="datetime",
                    tools="", toolbar_location=None, background_fill_color="#ffffff")
    select.yaxis.axis_label = 'Damper'

    select.step(index_name, 'dmp_po', source=src, color='#70dbb8', legend_label="Damper position")
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool

    select.extra_y_ranges = {"tempDiff": Range1d(start=right_sec_y_min*(1-y_overlimit), end=right_sec_y_max*(1+y_overlimit))}
    select.add_layout(LinearAxis(y_range_name='tempDiff', axis_label='TDiff [F]'), 'right')
    select.step(index_name, 'temp_diff', source=src, color='#b870db', y_range_name='tempDiff', legend_label="Temp Difference")

    # add legend
    select.add_layout(Legend(), 'right')
    select.legend.click_policy = "hide"

    select.legend.label_text_font_size = "9px"
    select.legend.label_height = 5
    select.legend.glyph_height = 5
    select.legend.spacing = 5

    if fault_dates is not None:
        for box_ann in fault_hilight:
            select.add_layout(box_ann)


    # save plot
    if file_name != '':
        plot_name = f"{file_name}-tseries.html" 
    else:
        head, tail = os.path.split(csv_path)
        plot_name = '{}-timeseries.html'.format(tail.split('.csv')[0])
    output_file(join(fig_folder, plot_name))
    save(column(p, select))