import argparse
import pandas as pd
import numpy as np
import pickle
from common_py_utils import yaml_cfg_parser
from os.path import exists
from scipy.stats import mode
from scipy.signal import medfilt
from typing import List, IO, Union
from braingeneers.utils import smart_open
import zipfile
import io
import itertools
import functools
import collections
import multiprocessing
import hanging_threads
import os
import re
import dateutil


# hanging_threads.start_monitoring(seconds_frozen=60)
state_prob_col_dict = {0: 'probability_wake', 1: 'probability_nrem', 2: 'probability_rem'}
state_dict = {0: 'wake', 1: 'nrem', 2: 'rem'}
PARAMS_DIR = 'parameter_files/'
PARALLELISM = 8


class ExtendAction(argparse.Action):
    """
    This is available by default in python 3.8+, this code doesn't assume python 3.8+
    https://stackoverflow.com/questions/41152799/argparse-flatten-the-result-of-action-append
    """
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []
        items.extend(values)
        setattr(namespace, self.dest, items)


def arg_parser():
    parser = argparse.ArgumentParser(
        description='Convert a single neural data file from the default time-first to channels-first format.'
    )
    parser.register('action', 'extend', ExtendAction)

    parser.add_argument(
        '--dataset-params-file', type=str, required=True,
        help='Dataset parameter file, files found in "src/parameter_files/dataset-*.yaml". Example: dataset-CAF42'
    )
    parser.add_argument(
        '--calling-params-file', type=str, required=True,
        help='Flicker calling parameter file, files found in "src/parameter_files/flicker-calling-*.yaml". Example: flicker-calling-standard.yaml'
    )
    parser.add_argument(
        '--version-format-string', '-v', action='extend', nargs='+', required=True, dest='versions',
        help='The dataset version, this argument should be called multiple times, typically 3, to specify each of the '
             'runs used in flicker calling. Example: '
             '--version wnr-v14-perregion-c24k-{channels_from:d}-{channels_to:d}-run1 --version wnr-v14-perregion-c24k-{channels_from:d}-{channels_to:d}-run2 --version wnr-v14-perregion-c24k-{channels_from:d}-{channels_to:d}-run3'
    )
    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output path and filename for result CSV. '
             'Example: s3://hengenlab/CAF26/flicker-calling/Results/wnr-v14-perregion-c24k-0-64/flicker-calling-standard-CAF26-wnr-v14-perregion-c24k.csv'
    )
    parser.add_argument(
        '--proportion_wake_window_sec', type=float, required=False, default=[60*60], action='extend', nargs='+',
        help='Enrichment column added, a window over which the proportion of wake is calculated from labels. '
             'If this is specified multiple times multiple columns will be added with different window sizes.'
    )

    args = vars(parser.parse_args())
    calling_params = yaml_cfg_parser.parse_yaml_cfg(f'{PARAMS_DIR}/{args["calling_params_file"]}', is_file=True)
    dataset_params = yaml_cfg_parser.parse_yaml_cfg(f'{PARAMS_DIR}/{args["dataset_params_file"]}', is_file=True)

    assert len(args['versions']) >= 3, 'At least 3 runs of the same model are required for flicker calling. ' \
                                       'See command line help for correct --version usage.'

    return {
        **args,
        **calling_params,
        **dataset_params['datasetparams'],
        **dataset_params['flickercalling'],
        'dataset_name': dataset_params['DATASET_NAME'],
    }


def main(versions: List[str], wide_filter_secs: int, narrow_filter_secs: int, threshold: float,
         video_fps: int, regions: list, dataset_name: str,
         transition_labels: str, output: str, proportion_wake_window_sec: float, **_):
    wide_filter_width = wide_filter_secs * video_fps
    narrow_filter_width = narrow_filter_secs * video_fps
    assert output.endswith('.zip'), 'Output file should end with .zip extension.'

    #
    # Call Flickers
    #
    print('PROGRESS: Begin flicker calling')

    # Prepare the function and parameters for starmap (necessary to parallelize this)
    f = functools.partial(
        process_region_and_state,
        regions=regions,
        versions=versions,
        dataset_name=dataset_name,
        wide_filter_width=wide_filter_width,
        narrow_filter_width=narrow_filter_width,
        threshold=threshold,
        transition_labels=transition_labels,
        video_fps=video_fps,
    )
    f_params = itertools.product(state_prob_col_dict.keys(), range(len(regions)))

    # Starmap, process regions+states in parallel
    with multiprocessing.Pool(PARALLELISM) as pool:
        finished_regional_substate_probs_dicts = pool.starmap(f, f_params)  # for debugging use itertools.starmap(f, params) to run sequentially in one process.
    finished_regional_substate_probs_dict = dict(collections.ChainMap(*finished_regional_substate_probs_dicts))  # merge the dictionaries together

    # convert to pandas dataframe
    df = convert_to_dataframe(
        finished_regional_substate_probs_dict=finished_regional_substate_probs_dict,
        regions=regions,
        dataset_name=dataset_name,
        version=versions[0].format(CHANNEL_FROM=0, CHANNEL_TO=64),
    )

    # exclude flickers that were in unconfident regions
    df = exclude_unconfident_region_flickers(df=df, regions=regions)

    # Enrich df with windowed proportion of wake|sleep
    assert isinstance(proportion_wake_window_sec, list)
    for ws in proportion_wake_window_sec:
        df = enrich_df_windows_proportion_wake_sleep(df, windows_size_sec=ws, fps=video_fps)

    # Save the dataframe
    print(f'PROGRESS: Saving dataframe to {output}')
    with smart_open.open(output, 'wb') as f:
        bio = io.BytesIO()
        df.to_csv(bio, index=False, header=True, na_rep='NaN')
        archive_name = os.path.basename(output).split('.zip')[0]
        with zipfile.ZipFile(f, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_archive:
            with zip_archive.open(archive_name, 'w') as zf:
                zf.write(bio.getvalue())


def exclude_unconfident_region_flickers(df: pd.DataFrame, regions: List[List[Union[str, int]]]):
    """ Some flickers will have been called in low confidence surrounding regions, this is a fix for that issue. """
    for region in regions:
        col_flicker = f'{region[0]}-{region[1]}-{region[2]}-flicker-state'
        col_surrounding = f'{region[0]}-{region[1]}-{region[2]}-surrounding-state'
        mask_unconfident = df[col_surrounding] == -1
        df[col_flicker][mask_unconfident] = -1

    return df


def enrich_df_windows_proportion_wake_sleep(df: pd.DataFrame, windows_size_sec: float, fps: int):
    window_size = int(round(windows_size_sec * fps))
    is_wake_series = df['label_wnr_012'] == 0
    proportion_wake = is_wake_series.rolling(window_size).mean()
    df[f'proportion_wake_{windows_size_sec}s_window'] = proportion_wake
    return df


def process_region_and_state(state: int, region_ix: int, regions: List[List[Union[str, int]]], versions: List[str],
                             dataset_name: str, wide_filter_width: int, narrow_filter_width: int,
                             threshold: float, transition_labels: str, video_fps: int):
    assert isinstance(threshold, float)
    region = regions[region_ix][0]
    channel_from = regions[region_ix][1]
    channel_to = regions[region_ix][2]

    # Prepare substates
    regional_substate_probs_dict = prepare_prob_timeseries(
        versions=versions,
        dataset_name=dataset_name,
        filter_width=narrow_filter_width,
        region=region,
        channel_from=channel_from,
        channel_to=channel_to,
        chosen_substate=state,
        threshold=threshold,
    )

    # Prepare states
    regional_state_probs_dict = prepare_prob_timeseries(
        versions=versions,
        dataset_name=dataset_name,
        filter_width=wide_filter_width,
        region=region,
        channel_from=channel_from,
        channel_to=channel_to,
        chosen_substate=state,
        threshold=threshold,
    )

    # Finish autocalling
    finished_regional_substate_probs_dict = finish_autocalling(
        regional_substate_probs_dict=regional_substate_probs_dict,
        regional_state_probs_dict=regional_state_probs_dict,
        transition_labels_filepath=transition_labels,
        state=state,
        region=region,
        channel_from=channel_from,
        channel_to=channel_to,
        fps=video_fps,
    )

    return finished_regional_substate_probs_dict


def finish_autocalling(regional_substate_probs_dict: dict, regional_state_probs_dict: dict,
                       transition_labels_filepath: str, state: int, region: str, channel_from: int, channel_to: int,
                       fps: int):

    finished_regional_substate_probs_dict = {}

    with smart_open.open(transition_labels_filepath, 'rb') as f:
        f_in_memory = io.BytesIO(f.read())

        if '.xlsx' in transition_labels_filepath:
            calls_df = pd.read_excel(f_in_memory, engine='openpyxl')
        elif '.csv' in transition_labels_filepath:
            calls_df = pd.read_csv(f_in_memory)
        else:
            raise ValueError("Calls file can only be csv or xlsx")

    trans_df = calls_df  # all events in these labeled sheets should be excluded
    trans_df = trans_df[(trans_df['start'] != '*') & (trans_df['end'] != '*')]

    print(f'PROGRESS: Finish autocalling for Region {region}, State {state}:{state_dict[state]}')
    region_substate_probs = regional_substate_probs_dict[region]
    region_state_probs = regional_state_probs_dict[region]

    region_blip_probs = np.where(((region_substate_probs == 1) & (region_state_probs == 0)), region_substate_probs, 0)
    #                              is-a-flicker                   surrounding-state-not-same-as-flicker   then flicker_probs else not called

    region_blip_probs = exclude_transitions(region_blip_probs, trans_df, fps=fps)

    finished_regional_substate_probs_dict[('prob-flicker', state, region, channel_from, channel_to)] = region_blip_probs
    finished_regional_substate_probs_dict[('prob-surrounding', state, region, channel_from, channel_to)] = region_state_probs
    finished_regional_substate_probs_dict['label_wnr_012'] = regional_substate_probs_dict['label_wnr_012']
    # finished_regional_substate_probs_dict['predicted_wnr_012'] = regional_substate_probs_dict['predicted_wnr_012']
    finished_regional_substate_probs_dict['time_of_day_sec'] = regional_substate_probs_dict['time_of_day_sec']

    return finished_regional_substate_probs_dict


def exclude_transitions(substate_probs, trans_df, fps):
    for index,row in trans_df.iterrows():
        row_start = int(time_to_row(row['start'], fps))
        row_end = int(time_to_row(row['end'],fps))
        substate_probs[row_start:row_end] = 0

    return substate_probs


def time_to_row(sec, fps):
    return fps * sec


def convert_to_dataframe(finished_regional_substate_probs_dict: dict, regions: List[List[Union[str, int]]], dataset_name: str, version: str):
    flicker_states = {}
    surrounding_states = {}
    for r in regions:
        flicker_state = np.zeros_like(finished_regional_substate_probs_dict[('prob-flicker', 0, r[0], r[1], r[2])]).astype(np.int32) - 1
        surrounding_state = np.zeros_like(finished_regional_substate_probs_dict[('prob-surrounding', 0, r[0], r[1], r[2])]).astype(np.int32) - 1

        # aggregate probabilities of WAKE|NREM|REM to a single called value
        for s in state_dict.keys():
            flicker_state[finished_regional_substate_probs_dict[('prob-flicker', s, r[0], r[1], r[2])] == 1.0] = s
            surrounding_state[finished_regional_substate_probs_dict[('prob-surrounding', s, r[0], r[1], r[2])] == 1.0] = s
        flicker_states[f'{r[0]}-{r[1]}-{r[2]}-flicker-state'] = flicker_state
        surrounding_states[f'{r[0]}-{r[1]}-{r[2]}-surrounding-state'] = surrounding_state

    non_region_column_names = list(sorted([k for k in finished_regional_substate_probs_dict.keys() if isinstance(k, str)]))
    non_region_columns = {k: v for k, v in finished_regional_substate_probs_dict.items() if isinstance(k, str)}

    output_dataframe = pd.DataFrame(
        data={**non_region_columns, **flicker_states, **surrounding_states},
        columns=[*non_region_column_names, *sorted(list(flicker_states.keys()) + list(surrounding_states.keys()))]
    )

    return output_dataframe


@functools.lru_cache
def load_predictions_file(url_or_filepath) -> pd.DataFrame:
    with smart_open.open(url_or_filepath, 'rb') as f:
        z = zipfile.ZipFile(f)
        filename_list = z.namelist()
        assert len(filename_list) == 1, f'Only 1 file was expected in the predictions zip, found: {filename_list}'
        b = z.read(filename_list[0])
        df = pd.read_csv(io.BytesIO(b))
    return df


def _get_time_of_day_sec(neural_filename: str):
    # Example input: Headstages_512_Channels_int16_2021-06-07_11-05-32.bin
    regex = r'Headstages_\d+_Channels_int16_\d+-\d+-\d+_(?P<hour>\d+)-(?P<min>\d+)-(?P<sec>\d+).bin'
    hour_min_sec = re.match(regex, neural_filename).groups()  # this timestamp is off by 0 to 5 min.
    sec = int(hour_min_sec[0]) * 3600 + int(hour_min_sec[1]) * 60 + int(hour_min_sec[2])
    return sec


def prepare_prob_timeseries(versions: List[str], dataset_name: str, filter_width: int,
                            region: str, channel_from: int, channel_to: int,
                            chosen_substate: int, threshold: float):

    chosen_col = state_prob_col_dict[chosen_substate]
    regional_substates_probs_dict = {}

    region_substate_probs_list = []

    versions = [v.format(CHANNEL_FROM=channel_from, CHANNEL_TO=channel_to) for v in versions]

    for version in versions:
        print(f'PROGRESS: Prepare prob timeseries for Region {region}, Version {version}, Filter {filter_width}, State {chosen_substate}')
        filepath = f's3://hengenlab/{dataset_name}/Runs/{version}/Results/predictions_{dataset_name}.csv.zip'
        df = load_predictions_file(filepath)
        regional_substates_probs_dict['label_wnr_012'] = df['label_wnr_012'].to_numpy()
        # regional_substates_probs_dict['predicted_wnr_012'] = df['predicted_wnr_012'].to_numpy()  # todo not correct because it doesn't account for all 3 runs
        regional_substates_probs_dict['time_of_day_sec'] = df['neural_filename'].map(_get_time_of_day_sec).add(df['neural_offset'].divide(7500000).multiply(300))  # filename gives us the base time + offset gives us the 5 minute offset from the timestamp, assumes fs=25000

        chosen_substate_probs = df[chosen_col]
        if threshold is not None:
            chosen_substate_probs = np.where(chosen_substate_probs > threshold,chosen_substate_probs, -1)

        other_substate_probs = [df[other_col].tolist() for other_col in state_prob_col_dict.values() if other_col != chosen_col]
        other_substate_probs = np.vstack(other_substate_probs)
        other_substate_probs = np.max(other_substate_probs, axis=0)
        compare_substate_probs = chosen_substate_probs - other_substate_probs
        top_state = np.where(compare_substate_probs > 0, 1, 0)
        region_substate_probs_list.append(top_state)
    region_substate_probs_array = np.vstack(region_substate_probs_list)
    region_moded_substate_probs = np.ndarray.flatten(mode(region_substate_probs_array,axis=0)[0])

    if filter_width is not None:
        # with two values in a list median is same as mode and runs faster
        filtered_regional_substate_probs = median_filter(region_moded_substate_probs,filter_width)
        regional_substates_probs_dict[region] = filtered_regional_substate_probs
    else:
        regional_substates_probs_dict[region] = region_moded_substate_probs

    return regional_substates_probs_dict


def median_filter(probs_list, filter_width):
    if filter_width % 2 == 0:
        filter_width += 1
    return medfilt(np.asarray(probs_list),filter_width)


if __name__ == '__main__':
    main(**arg_parser())
