import brickschema
import pandas as pd
import numpy as np
from os.path import join
import os
import sys

# Point this at your local sMAP Python client checkout.
SMAP_PYTHON_DIR = os.environ.get("SMAP_PYTHON_DIR", "/path/to/your/smap/python")
sys.path.append(SMAP_PYTHON_DIR)
sys.path.append(os.path.join(SMAP_PYTHON_DIR, "smap"))

from smap.archiver.client import SmapClient
from smap.contrib import dtutil

# create plots
from bokeh.palettes import Spectral8, Category20
from bokeh.io import show, save, output_file
from bokeh.layouts import column
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, RangeTool, LinearAxis, Range1d, BoxAnnotation, Legend

_debug = 1

def _query_hw_consumers(g):
    """
    Retrieve hot water consumers in the building, their respective
    boiler(s), and relevant hvac zones.
    """
    # query direct and indirect hot water consumersus
    hw_consumers_query = """SELECT DISTINCT * WHERE {
    ?boiler     rdf:type/rdfs:subClassOf?   brick:Hot_Water_Loop .
    ?boiler     brick:feeds+                ?t_unit .
    ?t_unit     rdf:type                    ?equip_type .
    ?mid_equip  brick:feeds                 ?t_unit .
    ?t_unit     brick:feeds+                ?room_space .
    ?room_space rdf:type/rdfs:subClassOf?   brick:HVAC_Zone .

        FILTER NOT EXISTS {
            ?subtype ^a ?t_unit ;
                (rdfs:subClassOf|^owl:equivalentClass)* ?equip_type .
            filter ( ?subtype != ?equip_type )
            }
    }
    """
    if _debug: print("Retrieving hot water consumers for each boiler.\n")

    q_result = g.query(hw_consumers_query)
    df_hw_consumers = pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars])

    return df_hw_consumers


def _clean_metadata(df_hw_consumers):
    """
    Cleans metadata dataframe to have unique hot water consumers with
    most specific classes associated to other relevant information.
    """

    unique_t_units = df_hw_consumers.loc[:, "t_unit"].unique()
    direct_consumers_bool = df_hw_consumers.loc[:, 'mid_equip'] == df_hw_consumers.loc[:, 'boiler']

    direct_consumers = df_hw_consumers.loc[direct_consumers_bool, :]
    indirect_consumers = df_hw_consumers.loc[~direct_consumers_bool, :]

    # remove any direct hot consumers listed in indirect consumers
    for unit in direct_consumers.loc[:, "t_unit"].unique():
        indir_test = indirect_consumers.loc[:, "t_unit"] == unit

        # update indirect consumers df
        indirect_consumers = indirect_consumers.loc[~indir_test, :]

    # label type of hot water consumer
    direct_consumers.loc[:, "consumer_type"] = "direct"
    indirect_consumers.loc[:, "consumer_type"] = "indirect"

    hw_consumers = pd.concat([direct_consumers, indirect_consumers])
    hw_consumers = hw_consumers.drop(columns=["subtype"]).reset_index(drop=True)

    return hw_consumers


def search_for_entities(g, class_type, point_list, relationship="brick:hasPoint"):
    """
    Return entities with the defined class type
    """

    if isinstance(point_list, list):
        points = " ".join(point_list)

    type_query = f"""SELECT DISTINCT * WHERE {{
        VALUES          ?req_point {{ {points} }}
        ?entity         rdf:type/rdfs:subClassOf?   {class_type} .
        ?entity         {relationship}              ?entity_points .
        ?entity_points  rdf:type/rdfs:subClassOf?   ?req_point .
        ?entity         brick:isPartOf?             ?larger_comp .
        ?larger_comp    rdf:type                    ?larger_comp_class .

        FILTER NOT EXISTS {{
            ?subtype ^a ?larger_comp ;
                (rdfs:subClassOf|^owl:equivalentClass)* ?larger_comp_class .
            filter ( ?subtype != ?larger_comp_class )
            }}
    }}
    """

    q_result = g.query(type_query)

    df = pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars])
    #df = df.drop_duplicates(subset=['point_name']).reset_index(drop=True)

    return df



def return_entity_points(g, entity, point_list):
    """
    Return defined brick point class for piece of equipment
    """
    
    if isinstance(point_list, list):
        points = " ".join(point_list)

    # query to return certain points of other points
    term_query = f"""SELECT DISTINCT * WHERE {{
        VALUES ?req_point {{ {points} }}
        ?point_name     rdf:type                        ?req_point .
        ?point_name     brick:isPointOf                 ?t_unit .
        ?point_name     brick:bacnetPoint               ?bacnet_id .
        ?point_name     brick:hasUnit?                  ?val_unit .
        ?bacnet_id      brick:hasBacnetDeviceInstance   ?bacnet_instance .
        ?bacnet_id      brick:hasBacnetDeviceType       ?bacnet_type .
        ?bacnet_id      brick:accessedAt                ?bacnet_net .
        ?bacnet_net     dbc:connstring                  ?bacnet_addr .
        }}"""

    # execute the query
    q_result = g.query(term_query, initBindings={"t_unit": entity})

    df = pd.DataFrame(q_result, columns=[str(s) for s in q_result.vars])
    df = df.drop_duplicates(subset=['point_name']).reset_index(drop=True)

    return df


def get_paths_from_tags(tags):
    paths = {key: tags[key]["Path"] for key in tags}
    paths = pd.DataFrame.from_dict(paths, orient='index', columns=['path'])
    # new_cols = ["empty", "site", "bms", "bacnet_instance", "bms2", "point_name"] ## old path implementation
    new_cols = ["empty", "site", "bacnet_device", "point_name", "bacnet_instance", "property_name"] ## new path implementation

    # adjustments to dataframe
    paths[new_cols] = paths.path.str.split("/", expand=True)
    paths = paths.drop(columns=["empty"])

    return paths


def plot_multiple_entities(metadata, data, start, end, filename, exclude_str=None, ylimits=None):

    # MERGE duplicate data from same path but different uuids
    df_grps = metadata.groupby(by=['path'])
    df_dups = metadata.duplicated(subset = ['path'], keep='first')
    df_uniq = metadata.loc[~df_dups, :]

    for grp_key in df_grps.groups.keys():
        cur_grp = df_grps.get_group(grp_key)
        cur_index = cur_grp.index.values

        merge_dat = data[cur_index[0]]
        for dt_idx in cur_index[1:]:
            cur_dat = data[dt_idx]

            if len(cur_dat) > 0:
                merge_dat = np.concatenate((merge_dat, cur_dat), axis=0)

        remain_idx = df_uniq.loc[df_uniq['index'].isin(cur_grp['index']), :].index.values

        # sort merged data set and save
        sort_idx = np.argsort(merge_dat[:, 0])
        data[remain_idx[0]] = merge_dat[sort_idx]
    
    metadata = df_uniq

    plots = []
    for ii, point_type in enumerate(metadata['req_point'].unique()):
        # if "Position" in point_type:
        #     y_plot_range = Range1d(start=0, end=101)
        # else:
        #     y_plot_range = Range1d(start=0, end=1.1)

        # plot settings
        plt_colors = Category20[20]

        x_range_str_time = pd.to_datetime(start, unit='s', utc=True).tz_convert('US/Pacific').tz_localize(None)
        x_range_end_time = pd.to_datetime(end, unit='s', utc=True).tz_convert('US/Pacific').tz_localize(None)

        if ii == 0:
            x_plot_range = (x_range_str_time, x_range_end_time)
        else:
            x_plot_range = plots[0].x_range

        p = figure(
            plot_height=300, plot_width=1500,
            x_axis_type="datetime", x_axis_location="below",
            x_range=x_plot_range,
            # y_range=y_plot_range
            )
        p.add_layout(Legend(), 'right')

        in_data = metadata["req_point"].isin([point_type])

        in_data_index = in_data[in_data].index
        df_subset = [data[x] for x in in_data_index]

        for i, dd in enumerate(df_subset):
            if exclude_str is not None:
                if any([nm in metadata.loc[in_data_index[i], "point_name_x"] for nm in exclude_str]):
                    continue
            p.step(
                pd.to_datetime(dd[:, 0], unit='ms', utc=True).tz_convert("US/Pacific").tz_localize(None),
                dd[:, 1], legend_label=metadata.loc[in_data_index[i], "point_name_x"],
                color = plt_colors[i % len(plt_colors)], line_width=2,
                mode = 'after'
                )

        y_axis_label = str(point_type).split("#")[1]
        p.yaxis.axis_label = y_axis_label

        if ylimits is not None:
            y_plot_range = Range1d(start=ylimits[0], end=ylimits[1])
            p.y_range = y_plot_range

        p.legend.click_policy = "hide"

        p.legend.label_text_font_size = "9px"
        p.legend.label_height = 5
        p.legend.glyph_height = 5
        p.legend.spacing = 5

        plots.append(p)

    output_file(filename)
    save(column(plots))

    return plots



def plot_boiler_temps(boiler_points_to_download, boiler_data, filename, ctrlr_sp=None, req_num=None):

    # plot settings
    plt_colors = Spectral8

    x_range_str_time = pd.to_datetime(start, unit='s', utc=True).tz_convert('US/Pacific').tz_localize(None)
    x_range_end_time = pd.to_datetime(end, unit='s', utc=True).tz_convert('US/Pacific').tz_localize(None)

    p = figure(
            plot_height=300, plot_width=1500,
            x_axis_type="datetime", x_axis_location="below",
            x_range=(x_range_str_time, x_range_end_time),
            y_range=Range1d(start=0, end=200)
            )
    p.add_layout(Legend(), 'right')
    p.yaxis.axis_label = "Boiler temperatures"

    for i, dd in enumerate(boiler_data):
        p.step(
            pd.to_datetime(dd[:, 0], unit='ms', utc=True).tz_convert("US/Pacific").tz_localize(None),
            dd[:, 1], legend_label=boiler_points_to_download.iloc[i]["point_name_x"],
            color = plt_colors[i % len(plt_colors)], line_width=2,
            mode = 'after'
            )

    # add extra plot lines
    if ctrlr_sp is not None:
        p = add_ctrl_data(p, ctrlr_sp)

    if req_num is not None:
        new_p = add_req_num_data(p, req_num)
        new_p.yaxis.axis_label = "HW Requests"

        plots = [p, new_p]
    else:
        plots = [p]

    for plt in plots:
        plt.legend.click_policy = "hide"

        plt.legend.label_text_font_size = "9px"
        plt.legend.label_height = 5
        plt.legend.glyph_height = 5
        plt.legend.spacing = 5

    output_file(filename)
    save(column(plots))

    return plots


def get_data_from_smap(points_to_download, paths, smap_client, start, end):
    data_ids = points_to_download["bacnet_instance"]
    avail_to_download = paths["bacnet_instance"].isin(data_ids)
    data_paths = paths.loc[avail_to_download, :]

    # combine the data frames
    df_combine = pd.merge(data_paths.reset_index(), points_to_download, how="right", on="bacnet_instance")

    # get data from smap
    data = smap_client.data_uuid(df_combine["index"], start, end, cache=False)

    return df_combine, data


def convert_smap_to_pandas(smap_dat_arr, col_labels=None):
    """
    Convert a dataset downloaded from smap to a pandas dataframe
    """

    df = []
    for i, dd in enumerate(smap_dat_arr):
        # df_timestamps = pd.to_datetime(dd[:, 0], unit='ms', utc=True).tz_convert("US/Pacific").tz_localize(None) # does not take into account daylight savings time change
        df_timestamps = pd.to_datetime(dd[:, 0], unit='ms', utc=True).tz_convert("US/Pacific")
        if col_labels is not None:
            cur_df = pd.DataFrame(dd[:, 1], index=df_timestamps, columns=[col_labels[i]])
        else:
            cur_df = pd.DataFrame(dd[:, 1], index=df_timestamps)

        # add cur_df to container
        df.append(cur_df)

    # combine all individual timeseries
    all_dfs = pd.concat(df, axis=1)

    return all_dfs


if __name__ == "__main__":
    # database settings
    url = "http://178.128.64.40:8079"
    keyStr = "B7qm4nnyPVZXbSfXo14sBZ5laV7YY5vjO19G"
    where = "Metadata/SourceName = 'Field Study 5a'"

    # set file names
    exp_brick_model_file = "./sdh_2024_shacl_expanded.ttl"

    # set save folder names
    plot_folder = "./figures"

    # time interval for to download data
    start = dtutil.dt2ts(dtutil.strptime_tz("05-01-2025", "%m-%d-%Y"))
    end   = dtutil.dt2ts(dtutil.strptime_tz("08-31-2025", "%m-%d-%Y"))

    # initiate smap client and download tags
    smap_client = SmapClient(url, key=keyStr)
    tags = smap_client.tags(where, asdict=True)

    # retrieve relevant tags from smap database
    paths = get_paths_from_tags(tags)


    # Download data using uuids directly
    # 1319e6fb-efd6-575c-a9f7-c43c8e1e37d8
    data = smap_client.data_uuid(['1319e6fb-efd6-575c-a9f7-c43c8e1e37d8'], start, end, cache=False)
    import pdb; pdb.set_trace()


    # load schema files
    g = brickschema.Graph()
    g.load_file(exp_brick_model_file)

    # query hot water consumers and clean metadata
    df_hw_consumers = _query_hw_consumers(g)
    import pdb; pdb.set_trace()
    df_hw_consumers = _clean_metadata(df_hw_consumers)

    #############################
    ##### Return hw consumer ctrl points
    #############################
    vlvs = ["brick:Position_Sensor", "brick:Valve_Command"]
    df_vlvs = []
    for t_unit in df_hw_consumers["t_unit"].unique():
        df_vlvs.append(return_entity_points(g, t_unit, vlvs))

    df_vlvs = pd.concat(df_vlvs).reset_index(drop=True)
    df_vlvs["bacnet_instance"] = df_vlvs["bacnet_instance"].astype(int).astype(str)

    # download data from smap
    ctrl_points_to_download, hw_ctrl_data = get_data_from_smap(df_vlvs, paths, smap_client, start, end)


    # create plot
    fig_file = join(plot_folder, "hw_consumer_ctrl.html")
    ctrl_plots = plot_multiple_entities(ctrl_points_to_download, hw_ctrl_data, start, end, fig_file, exclude_str=["REV", "DPR", "D-O"])


    #############################
    ##### Return hw consumer discharge temperatures
    #############################

    dischrg_temps = ["brick:Supply_Air_Temperature_Sensor", "brick:Embedded_Temperature_Sensor"]

    df_dischrg_temps = []
    for t_unit in df_hw_consumers["t_unit"].unique():
        df_dischrg_temps.append(return_entity_points(g, t_unit, dischrg_temps))

    df_dischrg_temps = pd.concat(df_dischrg_temps).reset_index(drop=True)
    df_dischrg_temps["bacnet_instance"] = df_dischrg_temps["bacnet_instance"].astype(int).astype(str)

    # download data from smap
    # TODO: there is a value error when cache is set to true
    dischrg_temps_to_download, dischrg_temps_data = get_data_from_smap(df_dischrg_temps, paths, smap_client, start, end)

    # create plots
    fig_file = join(plot_folder, "hw_consumer_discharge_temps.html")
    dischrg_temps_plots = plot_multiple_entities(dischrg_temps_to_download, dischrg_temps_data, start, end, fig_file)