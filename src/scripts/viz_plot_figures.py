""" Gathers results from all Runs, save JSON files and runs plots. """
from typing import List, Tuple, Dict, Union
from braingeneers.utils import s3wrangler
from braingeneers.utils import smart_open
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline
import plotly.figure_factory as ff
import plotly.express as px
import logging
import sys
import numpy as np
import json
import argparse
import itertools
import io
import re
import os
import math
import random
import functools
import pprint
import recordtype
from enum import Enum, EnumMeta
import multiprocessing
import yaml
import zipfile
import scipy
import scipy.stats
import sklearn
import sklearn.metrics
import diskcache
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from tenacity import *
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import hashlib
from braingeneers.utils import s3wrangler
from common_py_utils import common_utils
import statsmodels.stats.anova as anova
from pymer4.models import Lmer
from common_py_utils.common_utils import SPIKING_CHANNELS, NON_SPIKING_CHANNELS, GAIN, PROBE_REGION_CHANNEL_LIST, Regions, Probe


BUCKET = 'hengenlab'
BASEPATH_DATA = 's3://hengenlab'
BASEPATH_PARAMETER_FILES = '../parameter_files'
BASEPATH_FIGURES = '../../figures'
BASEPATH_AUDIT = '../../figures/audit'
BASEPATH_CACHE = '../../tmp/viz_plot_figures'
BASEPATH_MANUAL_DATA = '../../figures/data'
N_THREADS = 16


# Regex definitions for version strings and other matching patterns used here
class RegexDef(Enum):
    DEFAULT_SIZE = r'.*-sz-(?P<size>\d+)$'
    DEFAULT_RUN_GROUP = r'(?P<rungroup>.*)-sz-.*'
    PERREGION = r'wnr-v14-perregion-c64k-(?P<channel>\d+)-\d+-run(?P<run>\d+)'
    SAMPLE_SIZE = r'wnr-v14-run5-ch-(?P<channel>\d+)-sz-(?P<size>\d+)$'
    WHOLE_BRAIN = r'wnr-v14-whole-ch-(?P<channel>\d+)-\d+-sz-(?P<size>\d+)$'
    HIGHPASS = r'wnr-v14-c24k-highpass-(?P<hz>\d+)hz-region-(?P<channel>\d+)-(?P<size>\d+)$'
    HIGHPASS_SAMPLE_SIZE = r'wnr-v14-sizehighpass-ch-(?P<channel>\d+)-sz-(?P<size>\d+)$'
    HIGHPASS_SAMPLE_SIZE_DATASET_SHUFFLE = r'wnr-v14-run7.1-szhighwshuff-ch-(?P<channel>\d+)-sz-(?P<size>\d+)$'
    HIGHPASS_WITHIN_SAMPLE_SHUFFLE = r'wnr-v14-szhighpersampleshuff-ch-(?P<channel>\d+)-sz-(?P<size>\d+)$'
    ARCHIVE_REGEX = r'archive'
    BALANCED_ACCURACY_REGEX = r'.*(?P<bac>\d\.\d\d): Balanced Accuracy Score.*'
    FLICKER_CALLING_COLUMN = r'(?P<region>.+)-(?P<channel_from>\d+)-(?P<channel_to\d+)-(?P<type>flicker|surrounding)-state'


class FlickerFlavor(Enum):
    WAKE_WITHIN_NREM = ('WAKE', 'NREM')
    WAKE_WITHIN_REM = ('WAKE', 'REM')
    NREM_WITHIN_WAKE = ('NREM', 'WAKE')
    NREM_WITHIN_REM = ('NREM', 'REM')
    REM_WITHIN_WAKE = ('REM', 'WAKE')
    REM_WITHIN_NREM = ('REM', 'NREM')
    WAKE_WAKE_ANOMALY = ('WAKE', 'WAKE')
    NREM_NREM_ANOMALY = ('NREM', 'NREM')
    REM_REM_ANOMALY = ('REM', 'REM')


COLOR_MAP = {
    Regions.CA1: '#003F5C',
    Regions.V1V2: '#2F4B7C',
    Regions.S1: '#665191',
    Regions.M1M2: '#A05195',
    Regions.CP: '#D45087',
    Regions.SC: '#F95D6A',
    Regions.ACAD: '#FF7C43',
    Regions.RSPV: '#FFA600',
    Regions.LGN: '#F4DB30',
    Regions.NAC: '#506CE9',
    'wake': 'rgb(204,204,152)',
    'nrem': 'rgb(8,162,74)',
    'rem': 'rgb(38,92,46)',
}

# Data structures - data is passed around as a record type
Version = recordtype.recordtype(typename='Version', field_names='dataset_name version')
ModalAccuracy = recordtype.recordtype(typename='ModalAccuracy', field_names='region dataset_name versions modal_bac standard_bac accuracy f1micro f1macro modal_confusion_matrix standard_confusion_matrix')
HumanAccuracy = recordtype.recordtype(typename='HumanAccuracy', field_names='dataset_name bac modal_bac individual_sleep_state_path n_datapoints modal_confusion_matrix')
BalancedAccuracy = recordtype.recordtype(typename='BalancedAccuracy', field_names='dataset_name bac version')

DATASET_NAMES = ['CAF26', 'CAF34', 'CAF42', 'CAF69', 'EAB40', 'EAB50_5state', 'CAF99', 'CAF106', 'CAF77']
DATASET_ANON_NAMES_MAP = {
    'CAF26': 'Animal 1',
    'CAF34': 'Animal 2',
    'CAF42': 'Animal 3',
    'CAF69': 'Animal 4',
    'EAB40': 'Animal 5',
    'EAB50_5state': 'Animal 6',
    'CAF99': 'Animal 7',
    'CAF106': 'Animal 8',
    'CAF77': 'Animal 9',
}

WNR_MAP = {0: 'wake', 1: 'nrem', 2: 'rem'}

# Plotting Constants
MIN_ACCURACY = 0.34
MOVEMENT_THRESHOLD = 0.75
LONG_MEDIAN_WINDOW = 50 * 15
SHORT_MEDIAN_WINDOW = 10
LONG_MEDIAN_THRESHOLD = 0.8
SHORT_MEDIAN_THRESHOLD = 0.95

cache = diskcache.Cache(f'{BASEPATH_CACHE}/diskcache/', size_limit=int(300e9))
logging.basicConfig(stream=sys.stderr, level=logging.WARN)
logger = logging.getLogger(__name__)
pd.options.plotting.backend = "plotly"
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 250)
pd.set_option('display.width', 500)


def _get_n_stars(p_val: float):
    return 4 if p_val <= 0.0001 else 3 if p_val <= 0.001 else 2 if p_val <= 0.01 else 1 if p_val <= 0.05 else 0


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parsed_args = vars(parser.parse_args())
    return parsed_args


def _hash(*args):
    b = ''.join([str(arg) for arg in args]).encode('utf8')
    hash_str = hashlib.sha256(b).hexdigest()
    return hash_str


class PrintTiming:
    """ Easy way to print progress message + timing info """
    t0 = None

    def __init__(self, msg: str):
        print(msg, end=' ')

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'...{time.time() - self.t0:0.2f} sec')


def _get_versions(dataset_names: List = None, run_version_regex: (str, RegexDef) = None,
                  exclude_archive: bool = True, exclude_bad_probes: bool = False) -> List[Version]:
    """
    Get results from summary_statistics_$DATASET_NAME.txt files and
    returns the results in object/json form.

    :param dataset_names: A list of $DATASET_NAME values.
    :param run_version_regex: filter by model version name using a regex string. None will get all available.
    :param show_results: Display results from the query on stdout.
    :return: object/json form: {$DATASET_NAME: {$VERSION: {'bac': balanced-acc-score}}}
    """
    run_version_regex = run_version_regex if isinstance(run_version_regex, str) else run_version_regex.value

    with multiprocessing.pool.ThreadPool(processes=N_THREADS) as pool:
        # Query VERSION names, returns list of tuples [(dataset_name, version), ...]
        versions_all = list(itertools.chain.from_iterable(
            pool.map(lambda dn: itertools.product([dn], _list_versions(dn)), dataset_names)
        ))

        versions_filtered = versions_all

        # Filter by run_version_regex
        versions_filtered = [
            (dn, v) for dn, v in versions_filtered
            if re.match(run_version_regex, v, re.IGNORECASE)
        ]

        # Optionally exclude "archive" paths by regex
        versions_filtered = versions_filtered if not exclude_archive else [
            (dn, v) for dn, v in versions_filtered
            if not re.match(RegexDef.ARCHIVE_REGEX.value, v, re.IGNORECASE)
        ]

        # Optionally exclude bad probes (this assumes a "channel" capture group is defined in the regex)
        versions_filtered = versions_filtered if not exclude_bad_probes else [
            (dn, v) for dn, v in versions_filtered
            if not _is_bad_probe(dn, _get_channel(v, run_version_regex))
        ]

        # Apply regex to filter, returns tuples: [(dataset_name, version), ...]
        versions_filtered = [(dn, v) for dn, v in versions_filtered if re.search(run_version_regex, v)] \
            if run_version_regex is not None else versions_filtered

        runs = [Version(*r) for r in versions_filtered]
        return runs


def _enum_ordinal(e: EnumMeta, v: Enum):
    """ Returns the ordinal (index) of an Enum, example: _enum_ordinal(Region, Region.CA1) """
    for i, v0 in eumerate(e):
        if v0 == v:
            return i
        else:
            continue


# @retry(stop=stop_after_delay(3600), wait=wait_fixed(5), before=after_log(logger, logging.WARN))
def _list_versions(dataset_name: str) -> List[str]:
    """ Queries S3 path dataset_name and returns Run versions (paths under .../Run/: [version1, version2, ...] """
    path = f'{BASEPATH_DATA}/{dataset_name}/Runs/'
    s3urls = s3wrangler.list_directories(path)
    versions = map(lambda p: os.path.basename(os.path.normpath(p)), s3urls)
    return versions


def _read_summary_stats_file(dataset_name: str, version: str) -> str:
    cache_filepath = f'{BASEPATH_CACHE}/{dataset_name}-{version}-summary-statistics.txt'
    try:
        # Check for file in cache first
        with smart_open.open(cache_filepath, 'r') as f:
            summary_stats_txt = f.read()
    except FileNotFoundError:
        # Download and save in local cache
        path = f'{BASEPATH_DATA}/{dataset_name}/Runs/{version}/Results/summary_statistics_{dataset_name}.txt'
        print(f'INFO: Downloading {path}')
        buff = io.BytesIO()
        s3wrangler.download(path=path, local_file=buff, use_threads=False)
        summary_stats_txt = buff.getvalue().decode()
        with smart_open.open(cache_filepath, 'w') as f:
            f.write(summary_stats_txt)

    return summary_stats_txt


def _parse_summary_statistics(summary_stats_txt: str):
    balanced_accuracy_line = re.search(RegexDef.BALANCED_ACCURACY_REGEX.value, summary_stats_txt)
    assert balanced_accuracy_line is not None, f'Balanced accuracy regex failed in text: {summary_stats_txt}'
    balanced_accuracy_score = float(balanced_accuracy_line.group('bac'))
    return balanced_accuracy_score


def _get_region(dataset_name: str, channel: int) -> Regions:
    probes = [
        p for p in PROBE_REGION_CHANNEL_LIST
        if p.dataset_name == dataset_name and p.channel_from <= channel < p.channel_to
    ]
    assert len(probes) == 1, f'No region found for {dataset_name}, channel {channel}'
    return probes[0].region


def _get_channel(version: str, version_regex: (str, RegexDef) = RegexDef.SAMPLE_SIZE) -> int:
    r = version_regex.value if isinstance(version_regex, RegexDef) else version_regex
    return int(re.search(r, version).group('channel'))


def _get_size(version: str, version_regex: RegexDef = RegexDef.DEFAULT_SIZE) -> int:
    r = version_regex.value if isinstance(version_regex, Enum) else version_regex
    return int(re.search(r, version).group('size'))


# @cache.memoize()  # todo temp fix for cluster, re-enable for local use
def _get_raw_data_sample(filepath: str, n_channels: int, offset: int, size: int):
    """ Reads and caches a segment of raw data from a specified file (s3 or local). """
    with smart_open.open(filepath, 'rb') as f_orig:
        f_orig.seek(8 + 2 * n_channels * offset)  # Seek to correct offset
        b = f_orig.read(size * 2 * n_channels)  # Read file from cache location
        data = np.frombuffer(b, dtype=np.int16).reshape((n_channels, -1), order='F')
        return data * GAIN


def _parse_regex(version: str, regex: (str, Enum), group_name: str, cast_int: bool = False):
    regex = regex.value if isinstance(regex, Enum) else regex
    result = re.search(regex, version).group(group_name)
    result = int(result) if cast_int else result
    return result


def _get_run_group(dataset_name: str, version: str) -> str:
    run_group = dataset_name + _parse_regex(version, RegexDef.DEFAULT_RUN_GROUP, group_name='rungroup', cast_int=False)
    return run_group


def _is_spiking(dataset_name: str, channel: int):
    is_spiking = channel in SPIKING_CHANNELS[dataset_name]
    is_non_spiking = channel in NON_SPIKING_CHANNELS[dataset_name]
    assert is_spiking != is_non_spiking, \
        f'{dataset_name}, channel {channel} must be XOR, ' \
        f'got is_spiking={is_spiking}, is_non_spiking={is_non_spiking}'
    return is_spiking


def _is_bad_probe(dataset_name: str, channel: int):
    is_bad_probe = _get_region(dataset_name, channel) == Regions.BADPROBE
    return is_bad_probe


def _get_test_videos(dataset_name: str):
    filepath = f'{BASEPATH_PARAMETER_FILES}/dataset-{dataset_name}.yaml'
    with smart_open.open(filepath, 'r') as f:
        dataset_params = yaml.safe_load(f)
    test_video_files = dataset_params['datasetparams']['test_video_files']
    return test_video_files


@cache.memoize()
def _pull_predictions(dataset_name: str, version: str, testset_only: bool = True) -> np.ndarray:
    """ Downloads the prediction file for a given dataset_name/version as numpy array. """
    original_filepath = f'{BASEPATH_DATA}/{dataset_name}/Runs/{version}/Results/predictions_{dataset_name}.csv.zip'
    with smart_open.open(original_filepath, 'rb') as f:
        with zipfile.ZipFile(f) as z:
            buff = z.read(f'predictions_{dataset_name}.csv')
            p = np.recfromcsv(fname=io.BytesIO(buff))
            test_video_files = _get_test_videos(dataset_name)
            test_video_files_bytes = [v.encode('utf8') for v in test_video_files]
            p = p[np.isin(p['video_filename'], test_video_files_bytes)] if testset_only else p
            return p


def _compute_modal_accuracy(dataset_name: str, versions: List[str], mode_length_fps: int):
    """
    Computes modal accuracy from a list of np.recfromcsv arrays of
    the predictions files for each probe of a given dataset.
    """
    # Get predictions files
    f = functools.partial(_pull_predictions, dataset_name)
    predictions = map(f, versions)

    results = list(predictions)
    predictions_fine = [result['predicted_wnr_012'] for result in results]
    labels_fine = [result['label_wnr_012'] for result in results]

    predictions_split = [np.array_split(pf, len(pf)//mode_length_fps) for pf in predictions_fine]
    labels_split = [np.array_split(lf, len(lf)//mode_length_fps) for lf in labels_fine]

    # merge the splits - 2759, (196,) - on eab40 example
    predictions_split_merge = list(map(list, zip(*predictions_split)))
    predictions_split_concat = [np.concatenate(x) for x in predictions_split_merge]
    predictions_coarse_mode = np.array([scipy.stats.mode(x).mode[0] for x in predictions_split_concat])

    labels_split_merge = list(map(list, zip(*labels_split)))
    labels_split_concat = [np.concatenate(x) for x in labels_split_merge]
    labels_coarse_mode = np.array([scipy.stats.mode(x).mode[0] for x in labels_split_concat])

    accuracy = sklearn.metrics.accuracy_score(y_true=labels_coarse_mode, y_pred=predictions_coarse_mode)
    f1micro = sklearn.metrics.f1_score(y_true=labels_coarse_mode, y_pred=predictions_coarse_mode, average='micro')
    f1macro = sklearn.metrics.f1_score(y_true=labels_coarse_mode, y_pred=predictions_coarse_mode, average='macro')
    bac = sklearn.metrics.balanced_accuracy_score(y_true=labels_coarse_mode, y_pred=predictions_coarse_mode)
    standard_bac = sklearn.metrics.balanced_accuracy_score(y_true=np.concatenate(labels_fine), y_pred=np.concatenate(predictions_fine))

    modal_confusion_matrix = sklearn.metrics.confusion_matrix(y_true=labels_coarse_mode, y_pred=predictions_coarse_mode, labels=[0, 1, 2])
    standard_confusion_matrix = sklearn.metrics.confusion_matrix(y_true=np.concatenate(labels_fine), y_pred=np.concatenate(predictions_fine), labels=[0, 1, 2])
    assert modal_confusion_matrix.shape == (3, 3)
    assert standard_confusion_matrix.shape == (3, 3)

    return dataset_name, versions, float(bac), float(standard_bac), float(accuracy), float(f1micro), float(f1macro), modal_confusion_matrix, standard_confusion_matrix


def _exists(filepath: str):
    exists = s3wrangler.does_object_exist(filepath) if filepath.startswith('s3://') else os.path.isfile(filepath)
    return exists


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Regions):
            return f'TYPE Region {obj.value}'
        elif isinstance(obj, (Probe, Version, ModalAccuracy, HumanAccuracy, BalancedAccuracy)):
            # noinspection PyProtectedMember
            return (f'TYPE {type(obj).__name__} FIELDS {" ".join(obj._fields)}',) + tuple(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        else:
            return json.JSONEncoder.default(self, obj)


def _save_json(data: List[tuple], save_path: (str, object)):
    # data_normalized = [(type(d).__name__,) + tuple(d) for d in data]
    f = smart_open.open(save_path, 'w') if isinstance(save_path, str) else save_path
    json.dump(data, f, cls=JsonEncoder, indent=2)
    f.close()


def _save_figure(fig: go.Figure, filepath: str, save_formats: Tuple[str] = ('svg', 'html', 'pdf')):
    if 'svg' in save_formats:
        fig.write_image(f'{filepath}.svg')
    if 'png' in save_formats:
        fig.write_image(f'{filepath}.png')
    if 'html' in save_formats:
        fig.write_html(f'{filepath}.html')
    if 'pdf' in save_formats:
        fig.write_image(f'{filepath}.pdf')


def _get_fps(dataset_name: str, default_fps: int = 15):
    """ Get FPS (frames per sec) from parameter file. """
    with open(f'{BASEPATH_PARAMETER_FILES}/dataset-{dataset_name}.yaml') as f:
        params = yaml.safe_load(f)
        fps = params['datasetparams'].get('video_fps', default_fps)
    return fps


@cache.memoize()
def _load_npy_file(filepath: str) -> np.ndarray:
    """ Loads a numpy file from local or S3. """
    assert isinstance(filepath, str), f'String expected, got {type(filepath)}'
    with smart_open.open(filepath, 'rb') as f:
        data = np.load(f)
    return data


def _transform_mode(a: (np.ndarray, list, tuple), mode_length_samples: int) -> np.ndarray:
    """ Converts a list of values to the mode of those values of size modal_length. """
    mode_splits = np.array_split(a, np.arange(mode_length_samples, len(a), mode_length_samples))
    mode_values = [scipy.stats.mode(ms).mode[0] for ms in mode_splits]
    return mode_values


def _pull_human_bac(dataset_name: str, mode_length_samples: int) -> List[HumanAccuracy]:
    """ Pull human sleep score balance accuracy from files in s3://hengenlab/$DATASET_NAME/SleepStateIndividual/ """

    # consensus numpy file
    sleep_state_consensus_base_path = f'{BASEPATH_DATA}/{dataset_name}/SleepState/'
    sleep_state_consensus_path = s3wrangler.list_objects(sleep_state_consensus_base_path)
    assert len(sleep_state_consensus_path) == 1, f'Expected 1 sleep state consensus file, got: {sleep_state_consensus_path}'

    # individual sleep state files
    sleep_state_individual_path = f'{BASEPATH_DATA}/{dataset_name}/SleepStateIndividual/'
    sleep_state_paths = s3wrangler.list_objects(sleep_state_individual_path)

    # Load all numpy files in parallel
    with multiprocessing.Pool(N_THREADS) as pool:
        ss_consensus_individuals = list(pool.map(_load_npy_file, sleep_state_consensus_path + sleep_state_paths))

    # exclude -1 values
    ss_consensus_individuals_trimmed = [s[s != -1] for s in ss_consensus_individuals]

    ss_consensus_individuals_trimmed_shapes = [s.shape for s in ss_consensus_individuals_trimmed]
    assert np.all(np.equal(ss_consensus_individuals_trimmed_shapes[0], ss_consensus_individuals_trimmed_shapes)), \
        f'Not all sleep state files for {dataset_name} have the same length after -1 values are removed.\n' \
        f'Got shapes: {ss_consensus_individuals_trimmed_shapes}\n' \
        f'For sleep state files: {sleep_state_consensus_path + sleep_state_paths}'

    # apply mode
    ss_consensus_individuals_mode = list(map(
        functools.partial(_transform_mode, mode_length_samples=mode_length_samples),
        ss_consensus_individuals_trimmed
    ))

    balanced_accuracy_scores = [
        sklearn.metrics.balanced_accuracy_score(
            y_true=ss_consensus_individuals_trimmed[0],
            y_pred=ssc
        )
        for ssc in ss_consensus_individuals_trimmed[1:]
    ]

    modal_balanced_accuracy_scores = [
        sklearn.metrics.balanced_accuracy_score(
            y_true=ss_consensus_individuals_mode[0],
            y_pred=ssc
        )
        for ssc in ss_consensus_individuals_mode[1:]
    ]

    modal_confusion_matrix = np.array([
        sklearn.metrics.confusion_matrix(
            y_true=ss_consensus_individuals_mode[0],
            y_pred=ssc,
            labels=[1, 2, 3]
        )
        for ssc in ss_consensus_individuals_mode[1:]
    ])
    assert modal_confusion_matrix.shape[1:] == (3, 3)

    result = [
        HumanAccuracy(dataset_name, bac, modal_bac, ssp, npy.shape[0], mcm)
        for ssp, bac, modal_bac, npy, mcm in zip(
            sleep_state_paths,
            balanced_accuracy_scores,
            modal_balanced_accuracy_scores,
            ss_consensus_individuals[1:],
            modal_confusion_matrix,
        )
    ]

    return result


def compute_modal_bac(versions: List[Version], version_regex: RegexDef, mode_length_sec: int = 30):
    assert all([isinstance(v, Version) for v in versions])

    mode_length_fps = [mode_length_sec * _get_fps(dataset_name=v.dataset_name) for v in versions]

    # compute modal accuracies per each Version
    with multiprocessing.Pool(N_THREADS) as pool:
        modal_bac = pool.starmap(
            _compute_modal_accuracy,
            [(v.dataset_name, [v.version], m) for v, m in zip(versions, mode_length_fps)]
        )

    modal_bac_results = [
        ModalAccuracy(_get_region(mb[0], _get_channel(mb[1][0], version_regex)).value, *mb)
        for mb in modal_bac
    ]

    return modal_bac_results


def pull_models_bac_by_version(version_channel_regex: (str, Enum), exclude_bad_probes=True):
    version_channel_regex = version_channel_regex.value if isinstance(version_channel_regex, Enum) else version_channel_regex

    # get Versions for all per-region models
    versions = _get_versions(dataset_names=DATASET_NAMES, run_version_regex=version_channel_regex)

    # filter bad probes
    versions_nobadprobes = [
        p for p in versions
        if not _is_bad_probe(p.dataset_name, _get_channel(p.version, version_channel_regex))
    ] if exclude_bad_probes else versions

    # compute accuracies per each Version
    with multiprocessing.Pool(N_THREADS) as pool:
        summary_stats_files = list(pool.starmap(
            _read_summary_stats_file,
            [(v.dataset_name, v.version) for v in versions_nobadprobes],
        ))

        bac_only = pool.map(
            _parse_summary_statistics,
            summary_stats_files,
        )

    assert len(bac_only) == len(versions_nobadprobes)
    bac_results = [BalancedAccuracy(v.dataset_name, b, v.version) for b, v in zip(bac_only, versions_nobadprobes)]

    return bac_results


def plot_modal_bac(modal_region_results, tuple_dataset_name_human_accuracy):
    fig = go.Figure()

    regions = [r.value for r in Regions if r.name != 'BADPROBE']

    # Human scoring is done on a per-animal basis.
    # N = 3 humans in stderr calculation
    human_raw = [ha.modal_bac for ha in itertools.chain(*[dnha[1] for dnha in tuple_dataset_name_human_accuracy])]
    human_y = np.mean(human_raw)
    human_error_y = np.std(human_raw, ddof=0) / 3

    # CNN accuracies are the mean of each of the 3 models trained per region
    # stderr is calculated here with N = number of animals, NOT number of model runs (we average across model runs)
    cnn_modal_accuracy_grouped_by_region = [list(g) for _, g in itertools.groupby(sorted(modal_region_results, key=lambda ma: [r.value for r in Regions].index(ma.region)), key=lambda ma: ma.region)]
    cnn_modal_accuracy_grouped_by_run = [ [list(g) for _, g in itertools.groupby(magr, key=lambda ma: f'{ma.dataset_name}{ma.region}{ma.versions[0][:-1]}')] for magr in cnn_modal_accuracy_grouped_by_region]
    cnn_raw = [ [ np.mean([ma.modal_bac for ma in run_group]) for run_group in region_group] for region_group in cnn_modal_accuracy_grouped_by_run]
    cnn_y = [np.mean(x) for x in cnn_raw]
    cnn_error_y = [np.std(x, ddof=0) / np.sqrt(len(x)) for x in cnn_raw]

    cnn_all_y = np.mean([mrr.modal_bac for mrr in modal_region_results])
    cnn_all_error_y = np.std([mrr.modal_bac for mrr in modal_region_results], ddof=0) / np.sqrt(np.sum([len(run_len) for run_len in cnn_modal_accuracy_grouped_by_run]))

    cnn_n = [len(run_len) for run_len in cnn_modal_accuracy_grouped_by_run]

    anova = scipy.stats.f_oneway(*cnn_raw)
    ttest_human_vs_cnn = scipy.stats.ttest_ind(list(itertools.chain(*cnn_raw)), human_raw)

    fig.add_trace(go.Bar(
        name='Mean CNN',
        x=['CNN'],
        y=(cnn_all_y,),
        error_y=dict(type='data', array=(cnn_all_error_y,)),
        texttemplate=[f' N = {sum([len(c) for c in cnn_raw])}'],
        textposition="inside",
        textangle=-90,
        textfont_color="white",
        insidetextanchor="start",
        insidetextfont=dict(size=17),
    ))
    fig.add_trace(go.Bar(
        name='Mean Human',
        x=['Human'],
        y=(human_y,),
        error_y=dict(type='data', array=(human_error_y,)),
        texttemplate=f' N = 3 (per each of 10 animals)',
        textposition="inside",
        textangle=-90,
        textfont_color="white",
        insidetextanchor="start",
        insidetextfont=dict(size=17),
    ))

    # Spacer
    fig.add_trace(go.Bar(
        name=' ',
        x=[' '],
        y=([0]),
        showlegend=False,
    ))

    fig.add_trace(go.Bar(
        name='CNN by brain-region',
        x=regions,
        y=cnn_y,
        error_y=dict(type='data', array=cnn_error_y),
        texttemplate=[f' N = {n}' for n in cnn_n],
        textposition="inside",
        textangle=-90,
        textfont_color="white",
        insidetextanchor="start",
        insidetextfont=dict(size=17),
    ))

    fig.update_layout(barmode='group')
    fig.update_layout(yaxis=dict(title='Balanced Accuracy Score'))
    fig.update_layout(xaxis=dict(title='Human/CNN overage average and CNN per each brain region'))
    fig.update_layout(xaxis=dict(tickangle=45))

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-accuracy/figure1-cnn-vs-human-per-region')

    audit_data = {
        'cnn_notes': 'CNN accuracies are the mean of each of the 3 models trained per region. '
                     'stderr is calculated here with N = number of animals, NOT number of model runs '
                     '(we average across model runs)',
        'regions': regions, 'cnn_raw': cnn_raw, 'cnn_y': cnn_y, 'cnn_error_y': cnn_error_y,
        'cnn_anova': {'pvalue': anova.pvalue, 'statistic': anova.statistic},
        'cnn_human_ttest': {'pvalue': ttest_human_vs_cnn.pvalue, 'statistic': ttest_human_vs_cnn.statistic},
        'human_notes': 'Human scoring is done on a per-animal region. '
                       'Each human score is replicated per each region in each animal. '
                       'N = 3 humans in stderr calculation. ',
        'human_raw': human_raw, 'human_y': human_y, 'human_error_y': human_error_y,
    }
    _save_json(audit_data, f'{BASEPATH_AUDIT}/figure-accuracy.json')


def compute_human_accuracy_by_region_and_dataset(mode_length_samples: int):
    """
    Computes human accuracy per each region.
    Human accuracies are per dataset, so each accuracy measurement is duplicated per each
    region in the dataset.
    """
    regions = tuple(set([r.value for r in Regions if r.name != 'BADPROBE']))
    dataset_name_region_map = {ds: [p.region for p in PROBE_REGION_CHANNEL_LIST if p.dataset_name == ds] for ds in DATASET_NAMES}
    tuple_dataset_name_human_accuracy = list(zip(DATASET_NAMES, map(
        functools.partial(_pull_human_bac, mode_length_samples=mode_length_samples),
        DATASET_NAMES,
    )))
    tuple_region_human_accuracy = list(itertools.chain(*[list(itertools.product(dataset_name_region_map[ds], ha)) for ds, ha in tuple_dataset_name_human_accuracy]))
    list_tuples_human_accuracy_by_region = [[trha for trha in tuple_region_human_accuracy if trha[0].value == r] for r in regions]

    return list_tuples_human_accuracy_by_region, tuple_dataset_name_human_accuracy


def plot_human_cnn_confusion_matrices(perregion_modal_bac_results, tuple_dataset_name_human_accuracy):
    #
    # Human confusion matrix
    #
    human_confusion_matrix_sum = np.sum([human_accuracy.modal_confusion_matrix for each_dataset in tuple_dataset_name_human_accuracy for human_accuracy in each_dataset[1]], axis=0)
    human_confusion_matrix_norm = human_confusion_matrix_sum / np.sum(human_confusion_matrix_sum, axis=1)[:, np.newaxis]

    z_text = np.round(human_confusion_matrix_norm, 3).astype(np.str)
    fig = ff.create_annotated_heatmap(
        z=np.flipud(human_confusion_matrix_norm),
        colorscale='bluyl',
        x=['WAKE', 'NREM', 'REM'],
        y=['REM', 'NREM', 'WAKE'],
        annotation_text=np.flipud(z_text),
        zmin=0, zmax=1, zauto=False,
    )
    fig['data'][0]['showscale'] = True
    fig.update_xaxes(title='Human consensus score')
    fig.update_yaxes(title=f'Human predictions')

    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 16
        fig.layout.annotations[i].font.family = 'arial'

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-confusion-matrices/figure1-human-confusion-matrix')

    #
    # CNN confusion matrix
    #
    cnn_confusion_matrix_sum = np.sum([ma.modal_confusion_matrix for ma in perregion_modal_bac_results], axis=0)
    cnn_confusion_matrix_norm = cnn_confusion_matrix_sum / np.sum(cnn_confusion_matrix_sum, axis=1)[:, np.newaxis]

    z_text = np.round(cnn_confusion_matrix_norm, 3).astype(np.str)
    fig = ff.create_annotated_heatmap(
        z=np.flipud(cnn_confusion_matrix_norm),
        colorscale='bluyl',
        x=['WAKE', 'NREM', 'REM'],
        y=['REM', 'NREM', 'WAKE'],
        annotation_text=np.flipud(z_text),
    )
    fig['data'][0]['showscale'] = True
    fig.update_xaxes(title='Human consensus score')
    fig.update_yaxes(title=f'CNN predictions')

    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 16
        fig.layout.annotations[i].font.family = 'arial'

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-confusion-matrices/figure1-cnn-confusion-matrix')


def plot_confusion_matrices_per_sample_size(persamplesize_modal_bac_results):

    # Note: Using standard confusion matrix for per-sample-size plots
    for modal_accuracy in persamplesize_modal_bac_results:
        zdata = modal_accuracy.standard_confusion_matrix / np.sum(modal_accuracy.standard_confusion_matrix, axis=1)[:, np.newaxis]
        z_text = np.round(zdata, 3).astype(np.str)
        fig = ff.create_annotated_heatmap(
            z=np.flipud(zdata),
            colorscale='bluyl',
            x=['WAKE', 'NREM', 'REM'],
            y=['REM', 'NREM', 'WAKE'],
            annotation_text=np.flipud(z_text),
            zmin=0, zmax=1, zauto=False,
        )
        fig['data'][0]['showscale'] = True
        fig.update_xaxes(title='Human consensus score')
        fig.update_yaxes(title=f'CNN predictions')
        fig.update_layout(title=dict(text=f'{modal_accuracy.dataset_name} - {modal_accuracy.versions[0]} - {modal_accuracy.region} - BAC: {modal_accuracy.standard_bac:.3f}', yanchor='top'))
        fig.update_layout(margin=dict(l=20, r=20, t=175, b=50), width=700, height=600)

        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font.size = 16
            fig.layout.annotations[i].font.family = 'arial'

        _save_figure(fig, f'{BASEPATH_FIGURES}/figure-confusion-matrices-per-sample-size/'
                          f'confusion-matrix-{modal_accuracy.dataset_name}-{modal_accuracy.versions[0]}-{modal_accuracy.region}-{modal_accuracy.standard_bac:.3f}')


def plot_sample_size_bar_plot(runs_per_sample_size: List[BalancedAccuracy], min_accuracy: int):
    spiking_channels = list(itertools.chain(*[itertools.product([k], SPIKING_CHANNELS[k]) for k in SPIKING_CHANNELS.keys()]))
    non_spiking_channels = list(itertools.chain(*[itertools.product([k], NON_SPIKING_CHANNELS[k]) for k in NON_SPIKING_CHANNELS.keys()]))
    version_channel_sz_regex = r'wnr-v14-run5-ch-(\d+)-sz-(\d+)'

    # sanity check assertion that each channel is spiking XOR not-spiking
    for ba in runs_per_sample_size:
        is_spiking = (ba.dataset_name, int(ba.version.split('-')[-3])) in spiking_channels
        is_non_spiking = (ba.dataset_name, int(ba.version.split('-')[-3])) in non_spiking_channels
        assert is_spiking != is_non_spiking, \
            f'{ba.dataset_name}, channel {ba.version.split("-")[-3]} must be XOR, ' \
            f'got is_spiking={is_spiking}, is_non_spiking={is_non_spiking}'

    for filter_txt, filter_list in \
            ('spiking', spiking_channels), \
            ('non-spiking', non_spiking_channels), \
            ('all', spiking_channels + non_spiking_channels):

        # filter for spiking vs. non spiking and run the plot 3 times
        runs_per_sample_size_filtered = [
            ba for ba in runs_per_sample_size
            if (ba.dataset_name, int(ba.version.split('-')[-3])) in filter_list
        ]

        per_region_samples = {region.value: [
            run for run in runs_per_sample_size_filtered
            if region == _get_region(
                run.dataset_name,
                int(next(re.finditer(version_channel_sz_regex, run.version)).group(1))
            )]
            for region in Regions if region != Regions.BADPROBE
        }
        _save_json(per_region_samples, f'{BASEPATH_AUDIT}/per-region-samples-{filter_txt}')

        # Get minimum sample size per each region
        runs_per_region = {region.value: [ba for ba in per_region_samples[region.value]] for region in Regions if region != Regions.BADPROBE}
        run_groups_per_region = {region: list(set(['-'.join([ba.dataset_name] + ba.version.split('-')[:-2]) for ba in runs_per_region[region]])) for region in runs_per_region}
        ba_channel_sample_sizes_per_region = {region: [(ba, int(next(re.finditer(version_channel_sz_regex, ba.version)).group(1)), int(next(re.finditer(version_channel_sz_regex, ba.version)).group(2))) for ba in region_list] for region, region_list in runs_per_region.items()}
        channel_sample_sizes_per_region_run = {
            region: {
                rgpr: [
                    (b, c, s) for b, c, s in list_ba_ch_sz
                    if '-'.join([b.dataset_name] + b.version.split('-')[:-2]) == rgpr
                ] for rgpr in run_groups_per_region[region]
            }
            for region, list_ba_ch_sz in ba_channel_sample_sizes_per_region.items()
        }

        # tick_vals = list(range(len([r for r in Regions if r != Regions.BADPROBE])))
        #
        # tick_text = [region.value for region in Regions if region != Regions.BADPROBE]
        bar_yter = [
            (
                # bar Y values
                np.mean([
                    np.min([np.inf] + [sz for ba, ch, sz in ba_ch_sz_list if ba.bac >= min_accuracy])
                    for run_group_str, ba_ch_sz_list in run_dict.items()
                    if np.isfinite(np.min([np.inf] + [sz for ba, ch, sz in ba_ch_sz_list if ba.bac >= min_accuracy]))  # exclude channels that failed at all input sizes
                ]),
                # bar debug text string
                (region, '<br>'.join([f'{k}' for k, v in run_dict.items()])),
                # bar standard error
                np.std([
                    np.min([np.inf] + [sz for ba, ch, sz in ba_ch_sz_list if ba.bac >= min_accuracy])
                    for run_group_str, ba_ch_sz_list in run_dict.items()
                    if np.isfinite(np.min([np.inf] + [sz for ba, ch, sz in ba_ch_sz_list if ba.bac >= min_accuracy]))  # exclude channels that failed at all input sizes
                ]) / np.sqrt(np.sum([
                    1 for run_group_str, ba_ch_sz_list in run_dict.items()
                    if np.isfinite(np.min([np.inf] + [sz for ba, ch, sz in ba_ch_sz_list if ba.bac >= min_accuracy]))
                ])),
                region,
            )
            for region, run_dict in channel_sample_sizes_per_region_run.items()
        ]
        # Sort by bar height
        bar_yter = sorted(bar_yter, key=lambda x: x[0])

        region_order_map = {r: i for i, (y, txt, e, r) in enumerate(bar_yter)}

        # np.inf (or 100000) indicates a case when no values meet min_accuracy, so we say 24k model succeeded in all cases and becomes min
        scatter_xyt = sorted([
            (
                region_order_map[run_group] + (random.random() - 0.5) / 2,
                np.min([100000] + [sz for ba, _, sz in list_ba_ch_sz if ba.bac >= min_accuracy]),
                f'{list_ba_ch_sz[0][0].dataset_name}-{"-".join(list_ba_ch_sz[0][0].version.split("-")[:-2])}-{"-".join([f"{sz}:{ba.bac}" for ba, _, sz in list_ba_ch_sz if ba.bac >= min_accuracy])}',
            )
            for run_group, dict_group_ba_ch_sz in channel_sample_sizes_per_region_run.items()
            for _, list_ba_ch_sz in dict_group_ba_ch_sz.items()
        ])

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=np.arange(len(bar_yter)),
                y=[y for y, txt, e, r in bar_yter],
                error_y=dict(type='data', array=[e for y, txt, e, r in bar_yter]),
                name='Mean',
                marker_color=[COLOR_MAP[Regions(r)] for y, txt, e, r, in bar_yter],

            )
        )

        fig.update_xaxes(
            tickvals=np.arange(len(bar_yter)),
            ticktext=[r for y, txt, e, r in bar_yter],
        )
        fig.update_yaxes(title=f'Min samples for better than chance (>{int(min_accuracy * 100)}%)')
        fig.update_layout(width=1500, height=500)

        _save_figure(fig, f'{BASEPATH_FIGURES}/figure-sample-size/sample-size-bar-plot-without-scatter-{filter_txt}')

        for ds in DATASET_NAMES:
            fig.add_trace(
                go.Scatter(
                    x=[x for x, y, txt in scatter_xyt if txt.startswith(ds)],
                    y=[y for x, y, txt in scatter_xyt if txt.startswith(ds)],
                    text=[txt for x, y, txt in scatter_xyt if txt.startswith(ds)],
                    mode='markers',
                    name=f'Min Samples {ds}',
                    marker=dict(
                        # color='#ec838a',
                        size=10,
                    ),
                )
            )

        # Validate all runs are accounted for
        sample_size_counts = list(itertools.chain(*[
            [
                (
                    region,
                    run,
                    len(x1), len(x1) == len(list(list(channel_sample_sizes_per_region_run.items())[0][1].items())[0][1])
                )
                for run, x1 in x0.items()
            ]
            for region, x0 in channel_sample_sizes_per_region_run.items()
        ]))
        assert np.all([c for _, _, _, c in sample_size_counts]), f'Sample count validation failed:\n{sample_size_counts}'

        _save_figure(fig, f'{BASEPATH_FIGURES}/figure-sample-size/sample-size-bar-plot-{filter_txt}')
        _save_json(ba_channel_sample_sizes_per_region, f'{BASEPATH_AUDIT}/sample-size-per-region-{filter_txt}.json')
        _save_json(channel_sample_sizes_per_region_run, f'{BASEPATH_AUDIT}/sample-size-per-run-{filter_txt}.json')
        _save_json(bar_yter, f'{BASEPATH_AUDIT}/sample-size-bar-yter-{filter_txt}.json')
        _save_json(scatter_xyt, f'{BASEPATH_AUDIT}/sample-size-scatter-xyte-{filter_txt}.json')
        _save_json(sample_size_counts, f'{BASEPATH_AUDIT}/sample-size-counts-{filter_txt}.json')


def compute_anova(sample_size_per_run_spiking, sample_size_per_run_non_spiking, min_accuracy):
    anova_data_frame = pd.DataFrame(
        [
            (
                run_group_name.split('-')[0],
                run_group_name,
                np.min([sz for ba, ch, sz in run_group_tuple if ba[2] >= min_accuracy]),
                'spiking',
                region,
            )
            for region, region_run_group_dict in sample_size_per_run_spiking.items()
            for run_group_name, run_group_tuple in region_run_group_dict.items()
        ] + [
            (
                run_group_name.split('-')[0],
                run_group_name,
                np.min([sz for ba, ch, sz in run_group_tuple if ba[2] >= min_accuracy]),
                'non_spiking',
                region,
            )
            for region, region_run_group_dict in sample_size_per_run_non_spiking.items()
            for run_group_name, run_group_tuple in region_run_group_dict.items()
        ], columns=['animal', 'model', 'min_size', 'is_spiking', 'region'])
    anova_data_frame.to_csv(f'{BASEPATH_AUDIT}/anova-data-frame.csv')


def plot_sample_size_whole_brain(runs_whole_brain_per_sample_size):
    data = pd.DataFrame(
        [(ba.bac if ba.bac > 0 else 0.33, ba.dataset_name, _get_size(ba.version)) for ba in runs_whole_brain_per_sample_size],
        columns=['bac', 'dataset_name', 'sample_size']
    )
    per_sample_size_mean = data.groupby(by='sample_size').mean()
    per_sample_size_stderr = data.groupby(by='sample_size').sem()

    fig = go.Figure()

    for dataset_name in DATASET_NAMES:
        per_dataset_df = data[data['dataset_name'] == dataset_name].groupby(by='sample_size')
        per_dataset_mean = per_dataset_df.mean()

        fig.add_trace(
            go.Scatter(
                x=per_dataset_mean.bac.axes[0].asi8,
                y=np.array(per_dataset_mean.bac.array),
                mode='lines+markers',
                name=f'{DATASET_ANON_NAMES_MAP[dataset_name]}',
                line=dict(
                    color='rgba(200, 200, 200, 0.6)',
                    width=1.5,
                    dash='dash'
                ),
                marker=dict(
                    size=3,
                ),
            )
        )
        annotation_ix = np.where(np.array(per_dataset_mean.bac.array) > 0.33)[0][0]
        fig.add_annotation(
            x=math.log10(per_dataset_mean.bac.axes[0].asi8[annotation_ix]),
            y=np.array(per_dataset_mean.bac.array)[annotation_ix],
            text=f'{DATASET_ANON_NAMES_MAP[dataset_name]}',
            showarrow=False,
            yshift=0,
            font=dict(
                size=8,
                color="Black"
            )
        )

    fig.add_trace(
        go.Scatter(
            x=per_sample_size_mean.bac.axes[0].asi8,
            y=np.array(per_sample_size_mean.bac.array),
            error_y=dict(
                type='data',
                array=np.array(per_sample_size_stderr.bac.array),
                visible=True,
            ),
            mode='lines+markers',
            name=f'Mean balanced accuracy',
            line=dict(
                color='#ec838a',
                width=3,
            ),
            marker=dict(
                size=8,
            ),
        )
    )
    fig.add_annotation(
        x=math.log10(per_sample_size_mean.bac.axes[0].asi8[-1] - 24576),
        y=np.array(per_sample_size_mean.bac.array)[-1] - 0.005,
        text=f'Mean',
        showarrow=False,
        yshift=0,
        font=dict(
            size=8,
            color="Black"
        )
    )

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[100000, 10000, 1000, 100, 10, 1],
            ticktext=['4s', '400ms', '40ms', '4ms', '0.4ms', '0.04ms']
        )
    )

    fig.update_layout(xaxis_title='Sample input size, log scale', yaxis_title='Balanced Accuracy')
    fig.update_xaxes(autorange='reversed')
    fig.update_xaxes(type='log')
    fig.update_layout(width=1000, height=800)
    fig.update_layout(title='Whole brain model balanced accuracy as a function of sample size')

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-sample-size-whole-brain/sample-size-whole-brain')


def plot_proportion_of_model_above_chance(runs_per_sample_size, min_bac: float, sample_size: int):
    df = pd.DataFrame([
        (
            ba.dataset_name,
            ba.bac,
            ba.version,
            _get_run_group(ba.dataset_name, ba.version),
            _get_channel(ba.version),
            _get_size(ba.version),
            _get_region(ba.dataset_name, _get_channel(ba.version)).value,
            ba.bac >= min_bac,
        )
        for ba in runs_per_sample_size
    ], columns=['dataset_name', 'bac', 'version', 'run_group', 'channel', 'sample_size', 'region', 'is_min_bac'])
    # Group by region
    df_grouped = df[df['sample_size'] == sample_size].groupby('region')
    # Aggregate by proportion above chance
    df_agg_by_region = df_grouped['is_min_bac'].agg(**{
        'n_above_chance': np.sum,
        'n_samples': np.size,
        'proportion_above_chance': lambda x: np.sum(x)/x.shape[0]
    })
    df_sort_by_region = df_agg_by_region.sort_values(by='proportion_above_chance', ascending=False)

    fig = df_sort_by_region.plot(
        y='proportion_above_chance',
        kind='bar',
        title=f'Proportion of single-channel models above chance at {sample_size} samples',
    )
    fig.update_layout(
        xaxis_title='Region',
        yaxis_title=f'Proportion models > {min_bac} balanced accuracy',
    )
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-sample-size/proportion-above-chance-{sample_size}-samples')


def plot_highpass_runs(highpass_modal_bac_results: List[BalancedAccuracy]):
    data = pd.DataFrame([
        (
            ba.bac if ba.bac > 0 else 0.33,
            ba.dataset_name,
            _get_channel(version=ba.version, version_regex=RegexDef.HIGHPASS),
            _get_region(ba.dataset_name, _get_channel(version=ba.version, version_regex=RegexDef.HIGHPASS)).value,
            _parse_regex(ba.version, RegexDef.HIGHPASS, group_name='hz', cast_int=True),
        )
        for ba in highpass_modal_bac_results
    ], columns=['bac', 'dataset_name', 'probe_start_ch', 'region', 'highpass_hz'])

    data_per_region = data.sort_values(by='highpass_hz').groupby(by='region')

    fig = go.Figure()

    for region, data_by_region in data_per_region:
        data_group_by_hz = data_by_region.groupby(by='highpass_hz')
        # Plot mean line
        fig.add_trace(go.Scatter(
            name=f'{region}',
            x=list(data_group_by_hz.groups),
            y=list(data_group_by_hz['bac'].mean()),
            line=dict(
                width=3,
                color=COLOR_MAP[Regions(region)],
                shape='spline',
                smoothing=0.3,
            ),
        ))

        # Plot scatter
        scatter_data = data[data['region'] == region]
        fig.add_trace(go.Scatter(
            x=scatter_data['highpass_hz'].to_numpy() + (np.random.rand(len(scatter_data)) - 0.5) * 200,
            y=scatter_data['bac'],
            showlegend=False,
            mode='markers',
            marker=dict(
                color=COLOR_MAP[Regions(region)],
                size=8,
                line=dict(
                    width=2,
                    color=COLOR_MAP[Regions(region)],
                ),
            ),
        ))

    fig.update_layout(
        width=1500, height=1200,
        xaxis_title='Highpass filtered data in Hz',
        yaxis_title='Balanced accuracy score',
    )
    fig.update_xaxes(tickmode='array', tickvals=[500, 1000, 2000, 3000, 5000])

    _save_figure(fig, f'{BASEPATH_FIGURES}/highpass/highpass-per-region')
    with smart_open.open(f'{BASEPATH_AUDIT}/highpass-per-region.json', 'w') as f:
        f.write(data.to_json())


def _convert_int16_to_mv(x):
    return (x - 512) * 64 * GAIN


def _generate_histogram_data(per_sample_histogram_df: pd.DataFrame, region: str = None):
    """ Filters and aggregates data by region. """
    if region is not None:
        filter_by_region = per_sample_histogram_df['region'] == region
        per_sample_histogram_df = per_sample_histogram_df[filter_by_region]
    df_by_sleep_state = per_sample_histogram_df.groupby(by='sleep_state')
    df_sum = df_by_sleep_state.sum()

    hist_data = [
        df.iloc[1:].to_numpy() / np.sum(df.iloc[1:].to_numpy())
        for ix, df in df_sum.iterrows()
    ]

    return hist_data


def plot_per_state_region_distributions(per_sample_histogram_df: pd.DataFrame):

    # Single channel plots
    plot_params = [('EAB40', 178, 'MOp'), ('CAF69', 39, 'ACA'), ('EAB40', 192, 'MOp'), ('CAF26', 88, 'CA1'), ('CAF34', 163, 'CA1'), ('CAF99', 414, 'RSPv')]

    group_labels = ['WAKE', 'NREM', 'REM']

    fig = make_subplots(
        rows=9, cols=2,
        specs=[
            [{'colspan': 2}, None],     # all brain regions
            [{}, {}],                   # single channels
            [{}, {}],
            [{}, {}],
            [{}, {}],                   # per brain region, two per row
            [{}, {}],
            [{}, {}],
            [{}, {}],
            [{}, {}],
        ],
        subplot_titles=['All brain regions'] +
                       [f'{dn} - Channel {ch} - Region {r}' for dn, ch, r in plot_params] +
                       [f'Region {r.value}' for r in Regions if r != Regions.BADPROBE],
    )

    #
    # All brian regions
    #
    hist_data = _generate_histogram_data(per_sample_histogram_df, None)
    assert len(hist_data) == 3

    for i, (hdata, label) in enumerate(zip(hist_data, group_labels)):
        fig.add_trace(go.Scatter(
            name=f'{label}-ALL',
            x=_convert_int16_to_mv(np.arange(1024)),
            y=hdata,
            fill='tozeroy',
            fillcolor=f'rgba({205 if i == 0 else 92}, {205 if i == 1 else 92}, {205 if i == 2 else 92}, 0.5)',
            hoveron='points',
            line_color=f'rgba({205 if i == 0 else 92}, {205 if i == 1 else 92}, {205 if i == 2 else 92}, 1.0)',
            line=dict(width=0.5)
        ), row=1, col=1)

    #
    # Single channels
    #
    for j, (dataset_name, channel, region) in enumerate(plot_params):
        filter_mask = (per_sample_histogram_df['dataset_name'] == dataset_name) & (per_sample_histogram_df['channel'] == channel)
        per_channel_histogram_df = per_sample_histogram_df[filter_mask]
        hist_data = _generate_histogram_data(per_channel_histogram_df, None)
        assert len(hist_data) == 3

        row_start = 2
        for i, (hdata, label) in enumerate(zip(hist_data, group_labels)):
            fig.add_trace(go.Scatter(
                name=f'{label}-{region}-{dataset_name}-ch{channel}',
                x=_convert_int16_to_mv(np.arange(1024)),
                y=hdata,
                fill='tozeroy',
                fillcolor=f'rgba({205 if i == 0 else 92}, {205 if i == 1 else 92}, {205 if i == 2 else 92}, 0.5)',
                hoveron='points',
                line_color=f'rgba({205 if i == 0 else 92}, {205 if i == 1 else 92}, {205 if i == 2 else 92}, 1.0)',
                line=dict(width=0.5)
            ), row=(j//2) + row_start, col=j % 2 + 1)

    #
    # split out by region
    #
    regions = [r.value for r in Regions if r != Regions.BADPROBE]
    for j, region in enumerate(regions):
        hist_data = _generate_histogram_data(per_sample_histogram_df, region)
        assert len(hist_data) == 3

        for i, (hdata, label) in enumerate(zip(hist_data, group_labels)):
            row_start = 5  # The row the plots start on, rows are 1-indexed
            fig.add_trace(go.Scatter(
                name=f'{label}-{region}',
                x=_convert_int16_to_mv(np.arange(1024)),
                y=hdata,
                fill='tozeroy',
                fillcolor=f'rgba({205 if i==0 else 92}, {205 if i==1 else 92}, {205 if i==2 else 92}, 0.5)',
                hoveron='points',
                line_color=f'rgba({205 if i==0 else 92}, {205 if i==1 else 92}, {205 if i==2 else 92}, 1.0)',
                showlegend=True,
                line=dict(width=0.5)
            ), row=(j//2) + row_start, col=j % 2 + 1)

    fig.update_xaxes(range=[-1000, 1000], title=f'Millivolts (mv)')
    fig.update_yaxes(title='Histogram, normalized count')
    fig.update_layout(bargap=0, height=2000, width=1000, title='Voltage distribution - All brain regions')

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-sample-size-distributions/sample-size-distribution')


def _runs_list_to_df(runs_list: list, regex_def: RegexDef):
    """ accepts the output of pull_models_bac_by_version and converts the list to a pandas dataframe CSV style. """
    as_flat_list = [(
        r.dataset_name,
        r.version,
        r.bac,
        _get_channel(r.version, regex_def),
        _get_region(r.dataset_name, _get_channel(r.version, regex_def)).value,
        _get_size(r.version),
        _get_run_group(r.dataset_name, r.version),
        _is_spiking(r.dataset_name, _get_channel(r.version, regex_def)),
    ) for r in runs_list]
    df = pd.DataFrame(as_flat_list, columns=['dataset_name', 'version', 'balanced_accuracy', 'channel', 'region', 'input_size', 'run_group', 'is_spiking'])
    return df


def plot_raw_data(data: np.ndarray, title: str, filename: str, plot_sizes: (list, tuple), plot_ranges: (list, tuple)):
    """ Plot raw data at varying sample sizes """
    fig = make_subplots(
        rows=(len(plot_sizes) + 1) // 2,
        cols=2,
        subplot_titles=[f'Samples: {plot_size}' for plot_size in plot_sizes],
    )

    for i, (plot_size, plot_range) in enumerate(zip(plot_sizes, plot_ranges)):
        row = i // 2 + 1
        col = i % 2 + 1
        fig.add_trace(go.Scatter(
            name=f'{plot_size}',
            y=data[:plot_size],
            marker=dict(color='#604013')
        ), row=row, col=col)
        fig.update_yaxes(range=[-plot_range, plot_range], row=row, col=col)

    fig.update_layout(height=2000, width=1000, title=title)

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-raw-data/{filename}')


def plot_single_channel_distributions(dataset_name, channel, region, per_sample_histogram_df, basedir='per-channel'):
    filter_mask = (per_sample_histogram_df['dataset_name'] == dataset_name) & \
                  (per_sample_histogram_df['channel'] == channel)
    per_channel_histogram_df = per_sample_histogram_df[filter_mask]
    hist_data = _generate_histogram_data(per_channel_histogram_df, None)
    assert len(hist_data) == 3

    fig = go.Figure()

    for i, hdata in enumerate(hist_data):
        fig.add_trace(go.Scatter(
            name=f'{["WAKE", "NREM", "REM"][i]}',
            x=_convert_int16_to_mv(np.arange(1024)),
            y=hdata,
            fill='tozeroy',
            fillcolor=f'rgba({205 if i == 0 else 92}, {205 if i == 1 else 92}, {205 if i == 2 else 92}, 0.5)',
            hoveron='points',
            line_color=f'rgba({205 if i == 0 else 92}, {205 if i == 1 else 92}, {205 if i == 2 else 92}, 1.0)',
            line=dict(width=0.5)
        ))

    fig.update_layout(title=f'Region-{region.value}-{dataset_name}-ch-{channel}')
    fig.update_layout(width=1000, height=800)
    fig.update_xaxes(range=[-200, 200], title=f'Millivolts (mv)')
    fig.update_yaxes(title='Histogram, normalized count')

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-sample-size-distributions/{basedir}/distribution-{dataset_name}-ch-{channel}')


def plot_bar_highpass_sample_size_with_shuffle(runs_highpass_sample_size, runs_highpass_sample_size_with_shuffle):
    hpssws_ch_sz = [
        (
            r.dataset_name,
            _get_channel(r.version, RegexDef.HIGHPASS_SAMPLE_SIZE_DATASET_SHUFFLE),
            _get_size(r.version, RegexDef.HIGHPASS_SAMPLE_SIZE_DATASET_SHUFFLE),
        )
        for r in runs_highpass_sample_size_with_shuffle
    ]
    # Filter out highpass runs that don't have a shuffle run
    runs_highpass_sample_size_matching = [
        r for r in runs_highpass_sample_size
        if (r.dataset_name,
            _get_channel(r.version, RegexDef.HIGHPASS_SAMPLE_SIZE),
            _get_size(r.version, RegexDef.HIGHPASS_SAMPLE_SIZE)) in hpssws_ch_sz
    ]
    assert len(runs_highpass_sample_size_matching) == len(runs_highpass_sample_size_with_shuffle), \
        'Mismatched size between runs_highpass_sample_size_matching & runs_highpass_sample_size_with_shuffle'

    # filter runs for (unshuffled & shuffled) for models (>4k), (1k to 4k), and (<1k) (6 sets)
    high = 4196
    low = 1024
    runs_unshuffled_high = [r for r in runs_highpass_sample_size_matching if _get_size(r.version, RegexDef.HIGHPASS_SAMPLE_SIZE) > high]
    runs_unshuffled_med = [r for r in runs_highpass_sample_size_matching if low <= _get_size(r.version, RegexDef.HIGHPASS_SAMPLE_SIZE) <= high]
    runs_unshuffled_low = [r for r in runs_highpass_sample_size_matching if _get_size(r.version, RegexDef.HIGHPASS_SAMPLE_SIZE) < high]
    runs_shuffled_high = [r for r in runs_highpass_sample_size_with_shuffle if _get_size(r.version, RegexDef.HIGHPASS_SAMPLE_SIZE_DATASET_SHUFFLE) > high]
    runs_shuffled_med = [r for r in runs_highpass_sample_size_with_shuffle if low <= _get_size(r.version, RegexDef.HIGHPASS_SAMPLE_SIZE_DATASET_SHUFFLE) <= high]
    runs_shuffled_low = [r for r in runs_highpass_sample_size_with_shuffle if _get_size(r.version, RegexDef.HIGHPASS_SAMPLE_SIZE_DATASET_SHUFFLE) < high]

    _save_json({
        'runs_unshuffled_high': runs_unshuffled_high,
        'runs_unshuffled_med': runs_unshuffled_med,
        'runs_unshuffled_low': runs_unshuffled_low,
        'runs_shuffled_high': runs_shuffled_high,
        'runs_shuffled_med': runs_shuffled_med,
        'runs_shuffled_low': runs_shuffled_low,
    }, f'{BASEPATH_AUDIT}/plot-bar-highpass-unshuffled-shuffled.json')

    df_unshuffled_high = pd.DataFrame([r.bac for r in runs_unshuffled_high])
    df_unshuffled_med = pd.DataFrame([r.bac for r in runs_unshuffled_med])
    df_unshuffled_low = pd.DataFrame([r.bac for r in runs_unshuffled_low])
    df_shuffled_high = pd.DataFrame([r.bac for r in runs_shuffled_high])
    df_shuffled_med = pd.DataFrame([r.bac for r in runs_shuffled_med])
    df_shuffled_low = pd.DataFrame([r.bac for r in runs_shuffled_low])

    fig = go.Figure()

    # chance line
    fig.add_shape(type='line', x0=0.5, x1=3.5, y0=0.33, y1=0.33, line=dict(color='black', dash='dash', width=2), xref='x', yref='y')

    # bar plot
    fig.add_trace(go.Bar(
        x=[1, 2, 3],
        y=[float(df_unshuffled_high.mean()), float(df_unshuffled_med.mean()), float(df_unshuffled_low.mean())],
        text=[f'{float(df_unshuffled_high.mean()):.2f}', f'{float(df_unshuffled_med.mean()):.2f}', f'{float(df_unshuffled_low.mean()):.2f}'],
        name='Unshuffled',
    ))
    fig.add_trace(go.Bar(
        x=[1, 2, 3],
        y=[float(df_shuffled_high.mean()), float(df_shuffled_med.mean()), float(df_shuffled_low.mean())],
        text=[f'{float(df_shuffled_high.mean()):.2f}', f'{float(df_shuffled_med.mean()):.2f}', f'{float(df_shuffled_low.mean()):.2f}'],
        name='Shuffled',
    ))

    fig.update_layout(barmode='group', title='')
    fig.update_yaxes(range=[0, 1], title='Balanced Accuracy')
    fig.update_xaxes(range=[0.5, 3.5])

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-bar-highpass-shuffle/figure-bar-highpass-shuffle')


# noinspection PyUnusedLocal
@cache.memoize()
def _get_cacheable_file_with_modified(s3_or_local_path: str, last_modified_timestamp: str):
    print(f'DEBUG> {last_modified_timestamp}:  {s3_or_local_path}')

    with smart_open.open(s3_or_local_path, 'rb') as f:
        bio = io.BytesIO(f.read())

    return bio


def get_cacheable_file(s3_or_local_path: str) -> io.BytesIO:
    """
    Downloads a cacheable file from S3 or local path, stores it in the local disk cache.
    Checks last modified timestamp on the file for changes, downloads a new copy when it's changed
    """

    # get last modified timestamp so the file is re downloaded on changes
    if s3_or_local_path.startswith('s3://'):
        last_modified = str(s3wrangler.describe_objects(s3_or_local_path)[s3_or_local_path]['LastModified'])
    else:
        last_modified = str(time.ctime(os.stat(s3_or_local_path)[8]))

    return _get_cacheable_file_with_modified(s3_or_local_path, last_modified)


def _open_zipfile(bio: io.BytesIO) -> io.BytesIO:
    """ Open a zip file from a stream and return a binary stream of unzipped data. Assumes a single file zipfile """
    z = zipfile.ZipFile(bio)
    zfnames = z.namelist()
    assert len(zfnames) == 1
    f = z.open(zfnames[0], 'r')
    raw_bytes = io.BytesIO(f.read())
    return raw_bytes


def get_flicker_dataframes(flicker_calling_params: str, dataset_names: str, model_size: str, use_cache: bool = True):
    urls = [f's3://hengenlab/{dataset_name}/flicker-calling/Results/{flicker_calling_params}-{dataset_name}-wnr-v14-perregion-{model_size}.csv.zip' for dataset_name in dataset_names]
    last_modified = [str(s3obj['LastModified']) for url, s3obj in s3wrangler.describe_objects(urls).items()]
    dataframes = cache.get(('__get_flicker_dataframes__', urls, last_modified)) if use_cache else None

    if dataframes is None:
        # Download flicker calling CSVs
        with multiprocessing.Pool(N_THREADS) as pool:
            csv_zip_bytesio_list = pool.map(get_cacheable_file, urls)
            csv_bytesio_list = pool.map(_open_zipfile, csv_zip_bytesio_list)
            dataframes = pool.map(pd.read_csv, csv_bytesio_list)

            assert len(dataframes) == len(dataset_names)
            for dataframe, dataset_name in zip(dataframes, dataset_names):
                dataframe.attrs['dataset_name'] = dataset_name
                dataframe.attrs['fps'] = _get_fps(dataset_name)
        cache[('__get_flicker_dataframes__', urls, last_modified)] = dataframes

    return dataframes


def get_flickers_by_flavor(flicker_dataframes: List[pd.DataFrame], window_size_sec: int = 3600, as_dataframe: bool = False) -> List[Tuple[FlickerFlavor, float]]:
    """
    Returns list of flicker flavors and the proportion of wake within the window
    for a given column.

    flicker_state: pd.Series, surrounding_region_state: pd.Series, result: (pd.Series, pd.DataFrame), dataset_name: str, df: pd.DataFrame
    """
    def _get_flickers_by_flavor(flicker_state: pd.Series, surrounding_region_state: pd.Series, result: (pd.Series, pd.DataFrame), dataset_name: str, df: pd.DataFrame):
        mask_flickers = (flicker_state.to_numpy() != -1).astype(np.int32)
        flicker_starts = mask_flickers[1:] - mask_flickers[:-1] == 1
        flicker_ixs = np.where(flicker_starts)[0] + 1

        results = []

        for flicker_ix in flicker_ixs:
            # get predictions from region surrounding the flicker
            flicker_start = flicker_ix
            flicker_end = flicker_state.iloc[flicker_ix:].ne(flicker_state[flicker_ix]).argmax() + flicker_start

            # correct for edge case when flicker is called to the end of file
            flicker_end = len(flicker_state) if flicker_end == flicker_start else flicker_end
            assert all(flicker_state[flicker_start:flicker_end].eq(flicker_state[flicker_start]))  # sanity check

            flicker_surrounding_state = surrounding_region_state[flicker_start:flicker_end].mode()[0]

            flicker_flavor = WNR_MAP[flicker_state.iloc[flicker_ix]].upper()
            flicker_flavor_surrounding_region = WNR_MAP[flicker_surrounding_state].upper()
            flicker_flavor = FlickerFlavor((flicker_flavor, flicker_flavor_surrounding_region))

            results.append((
                flicker_flavor,                 # flicker flavor
                result.iloc[flicker_ix],        # time since last state change
                flicker_state.name,             # column name
                dataset_name,                   # dataset name
                flicker_ix,                     # flicker start index, location in predictions file
                df,                             # original dataframe in full
                flicker_end - flicker_start     # flicker length
            ))

        return results

    args = [
        (
            df[column_name],
            df[column_name.split('-flicker-state')[0] + '-surrounding-state'],
            df[f'proportion_wake_{window_size_sec}s_window'],
            df.attrs['dataset_name'],
            df,
        )
        for df in flicker_dataframes
        for column_name in df.columns if 'flicker-state' in column_name and not column_name.startswith('BADPROBE-')
    ]
    flickers_by_flavor = list(itertools.chain(*itertools.starmap(_get_flickers_by_flavor, args)))

    if as_dataframe:
        flickers_by_flavor = pd.DataFrame({
            'flicker_flavor': [f[0].name for f in flickers_by_flavor],
            f'proportion_wake_{window_size_sec}s_window': [f[1] for f in flickers_by_flavor],
            'column_name_flicker_calling': [f[2] for f in flickers_by_flavor],
            'dataset_name': [f[3] for f in flickers_by_flavor],
            'ix': [f[4] for f in flickers_by_flavor],
            'flicker_length': [f[6] for f in flickers_by_flavor],
        })

    return flickers_by_flavor


# def get_time_per_wnr_state_sec(flicker_dataframes: List[pd.DataFrame], n_bins: int):
#     """ normalize by total time per state (W|N|R) and per quantile as defined by n_bins """
#     time_per_wnr_state_sec = {0: [0.0] * n_bins, 1: [0.0] * n_bins, 2: [0.0] * n_bins}
#
#     for df in flicker_dataframes:
#         for state in [0, 1, 2]:
#             for i, (a, b) in enumerate(zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[:-1])):
#                 percent_time_in_state = df['time_since_last_state_change_sec'] / \
#                                         (df['time_since_last_state_change_sec'] + df['time_to_next_state_change_sec'])
#                 mask = (df['label_wnr_012'] == state) & (a < percent_time_in_state) & (percent_time_in_state <= b)
#                 time_per_wnr_state_sec[state][i] += df['label_wnr_012'][mask].count() * (1 / df.attrs['fps'])
#
#     return time_per_wnr_state_sec


# def plot_flickers_rate_by_time_in_state(flicker_dataframes: List[pd.DataFrame], bins: int):
#     """ xaxis: percent | absolute """
#     flickers_by_flavor = get_flickers_by_flavor(flicker_dataframes)
#
#     data = dict()
#     data_normalized = dict()
#     normix = {'WAKE': 0, 'NREM': 1, 'REM': 2}
#     max_time_since_last_state_change_sec = pd.concat(df['time_since_last_state_change_sec'] for df in flicker_dataframes).quantile(0.95) if xaxis == 'absolute' else None
#
#     # calculate histogram per each flicker flavor
#     for flicker_flavor, proportion_wake, column_name, dataset_name, flicker_ix, df, flicker_length in [f for f in flickers_by_flavor if not np.isnan(f[1])]:
#         data[flicker_flavor] = data.get(flicker_flavor, [0] * bins)  # initialize counts per each flicker flavor
#         # if xaxis == 'percent':
#         percent_time_in_state = df.iloc[flicker_ix]['time_since_last_state_change_sec'] / (df.iloc[flicker_ix]['time_since_last_state_change_sec'] + df.iloc[flicker_ix]['time_to_next_state_change_sec'])
#         bin_ix = min(int(percent_time_in_state * bins), bins - 1)  # min(..., bins-1) accounts for edge case of 1.0 values
#         # elif xaxis == 'absolute':
#         #     abs_time_in_state = df.iloc[flicker_ix]['time_since_last_state_change_sec']
#         #     bin_ix = min(int(abs_time_in_state / max_time_since_last_state_change_sec * bins), bins - 1)
#         # else:
#         #     raise ValueError(xaxis)
#         data[flicker_flavor][bin_ix] += 1
#
#     time_per_wnr_state_sec = get_time_per_wnr_state_sec(flicker_dataframes, n_bins=bins)
#     for flicker_flavor in data.keys():
#         data_normalized[flicker_flavor] = data[flicker_flavor] / time_per_wnr_state_sec[normix[flicker_flavor.value[1]]] * 60
#
#     flicker_flavors = [
#         FlickerFlavor.WAKE_WITHIN_NREM, FlickerFlavor.REM_WITHIN_NREM,
#         FlickerFlavor.WAKE_WITHIN_REM, FlickerFlavor.NREM_WITHIN_REM,
#         FlickerFlavor.NREM_WITHIN_WAKE, FlickerFlavor.REM_WITHIN_WAKE,
#     ]
#
#     if xaxis == 'percent':
#         x = [
#             f'{int((x-1/bins/2)*100)}-{int((x+1/bins/2)*100)}'
#             for x in np.linspace(0, 1, bins + 1)[:-1] + 1/bins/2
#         ]
#     elif xaxis == 'absolute':
#         x = [
#             f'{int(x-max_time_since_last_state_change_sec/bins/2)}s-{int(x+max_time_since_last_state_change_sec/bins/2)}s'
#             if x < max_time_since_last_state_change_sec
#             else f'{int(x-max_time_since_last_state_change_sec/bins/2)}s+'
#             for x in np.linspace(0, max_time_since_last_state_change_sec, bins + 1)[:-1] + max_time_since_last_state_change_sec/bins/2
#         ]
#     else:
#         raise ValueError(xaxis)
#
#     fig = go.Figure(
#         data=[
#             go.Bar(
#                 name='-within-'.join(flicker_flavor.value),
#                 x=x,
#                 y=data_normalized[flicker_flavor]
#             )
#             for flicker_flavor in flicker_flavors
#         ],
#         layout=go.Layout(
#             width=1000, height=700,
#         )
#     )
#
#     fig.update_xaxes(title=f'{xaxis.title()} time in state')
#     fig.update_yaxes(title='Flickers per minute rate')
#     fig.update_layout(
#         barmode='group',
#         title='Flickers rate per minute as a function of time spent in state, normalized by total time spent in '
#               'each state (e.g. WAKE-within-NREM normalized to time in NREM state'
#     )
#
#     _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers/flickers-rate-by-time-in-state-{xaxis}')


# def plot_flickers_proportion_wake(flicker_dataframes: List[pd.DataFrame], bins: int):
#
#     flickers_by_flavor = get_flickers_by_flavor(flicker_dataframes)
#
#     # histogram count
#     data = {}
#     data_normalized = {}
#     normix = {'WAKE': 0, 'NREM': 1, 'REM': 2}
#
#     for flicker_flavor, proportion_wake, column_name, dataset_name, flicker_ix, df, flicker_length in [f for f in flickers_by_flavor if not np.isnan(f[1])]:
#         data[flicker_flavor] = data.get(flicker_flavor, [0] * bins)  # initialize counts per each flicker flavor
#         bin_ix = min(int(proportion_wake * bins), bins-1)  # min(..., bins-1) accounts for edge case of 1.0 values
#         data[flicker_flavor][bin_ix] += 1
#
#     time_per_wnr_state_sec = get_time_per_wnr_state_sec(flicker_dataframes, n_bins=bins)
#
#     for flicker_flavor in data.keys():
#         data_normalized[flicker_flavor] = data[flicker_flavor] / time_per_wnr_state_sec[normix[flicker_flavor.value[1]]] * 60
#
#     # Sort by total flickers per min across all proportions
#     # flicker_flavors = [f[0] for f in sorted(list(data_normalized.items()), key=lambda x: np.sum(x[1]), reverse=True)]
#     flicker_flavors = [
#         FlickerFlavor.WAKE_WITHIN_NREM, FlickerFlavor.REM_WITHIN_NREM,
#         FlickerFlavor.WAKE_WITHIN_REM, FlickerFlavor.NREM_WITHIN_REM,
#         FlickerFlavor.NREM_WITHIN_WAKE, FlickerFlavor.REM_WITHIN_WAKE,
#     ]
#
#     # bar plot - normalized by the total time spent in the surrounding state (e.g. NREM for WAKE-within-NREM)
#     spacing = np.linspace(0, 1, bins + 1)
#     x_labels = [f'{spacing[i]:0.1f} - {spacing[i + 1]:0.1f}' for i in range(0, bins)]
#     fig = go.Figure(
#         data=[
#             go.Bar(
#                 name='-within-'.join(flicker_flavor.value),
#                 x=x_labels,
#                 y=data_normalized[flicker_flavor]
#             )
#             for flicker_flavor in flicker_flavors
#         ]
#     )
#
#     fig.update_xaxes(title='Proportion of past hour spent in wake')
#     fig.update_yaxes(title='Flickers per minute across all datasets')
#     fig.update_layout(barmode='group', title='Flickers rate per minute, normalized by the surrounding state '
#                                              '(e.g. WAKE-within-NREM normalized to time in NREM state')
#
#     _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers/flickers-by-proportion-wake')


# def plot_flickers_proportion_wake_aggregated(flicker_dataframes: List[pd.DataFrame], bins: int, region_filter: Regions = None):
#     """ aggregates down to only 2 bars, wake and sleep. """
#
#     flickers_by_flavor = get_flickers_by_flavor(flicker_dataframes)
#     # flickers_by_flavor_df = pd.DataFrame({
#     #     'flicker_flavor': [f[0].name for f in flickers_by_flavor],
#     #     'time_since_last_state_change': [f[1] for f in flickers_by_flavor],
#     #     'column': [f[2] for f in flickers_by_flavor],
#     #     'dataset_name': [f[3] for f in flickers_by_flavor],
#     #     'ix': [f[4] for f in flickers_by_flavor],
#     # })
#
#     # histogram count
#     data = {}
#     data_normalized = {}
#     data_by_dataset_name = {}
#     data_normalized_by_dataset_name = {}
#     normix = {'WAKE': 0, 'NREM': 1, 'REM': 2}
#
#     for flicker_flavor, proportion_wake, column_name, dataset_name, flicker_ix, df, flicker_length in [f for f in flickers_by_flavor if not np.isnan(f[1])]:
#         if region_filter is None or column_name.split('-')[0] == region_filter.name:
#             data[flicker_flavor] = data.get(flicker_flavor, [0] * bins)  # initialize counts per each flicker flavor
#             for ds in DATASET_NAMES:
#                 data_by_dataset_name[(flicker_flavor, ds)] = data_by_dataset_name.get((flicker_flavor, ds), [0] * bins)
#
#             bin_ix = min(int(proportion_wake * bins), bins-1)  # min(..., bins-1) accounts for edge case of 1.0 values
#
#             data[flicker_flavor][bin_ix] += 1
#             data_by_dataset_name[(flicker_flavor, dataset_name)][bin_ix] += 1
#
#     time_per_wnr_state_sec = get_time_per_wnr_state_sec(flicker_dataframes, n_bins=bins)
#
#     for flicker_flavor in data.keys():
#         data_normalized[flicker_flavor] = data[flicker_flavor] / time_per_wnr_state_sec[normix[flicker_flavor.value[1]]] * 60
#         for dataset_name in DATASET_NAMES:
#             data_normalized_by_dataset_name[(flicker_flavor, dataset_name)] = data_by_dataset_name[(flicker_flavor, dataset_name)] / time_per_wnr_state_sec[normix[flicker_flavor.value[1]]] * 60
#
#     y_wake = data_normalized[FlickerFlavor.NREM_WITHIN_WAKE] + \
#              data_normalized[FlickerFlavor.REM_WITHIN_WAKE]
#     y_sleep = data_normalized[FlickerFlavor.WAKE_WITHIN_NREM] + \
#               data_normalized[FlickerFlavor.REM_WITHIN_NREM] + \
#               data_normalized[FlickerFlavor.WAKE_WITHIN_REM] + \
#               data_normalized[FlickerFlavor.NREM_WITHIN_REM]
#     y_wake_error = np.std(np.array([
#         data_normalized_by_dataset_name[(FlickerFlavor.NREM_WITHIN_WAKE, dn)] +
#         data_normalized_by_dataset_name[(FlickerFlavor.REM_WITHIN_WAKE, dn)]
#         for dn in DATASET_NAMES
#     ]), axis=0) / np.sqrt(len(DATASET_NAMES))
#     y_sleep_error = np.std(np.array([
#         data_normalized_by_dataset_name[(FlickerFlavor.WAKE_WITHIN_NREM, dn)] +
#         data_normalized_by_dataset_name[(FlickerFlavor.REM_WITHIN_NREM, dn)] +
#         data_normalized_by_dataset_name[(FlickerFlavor.WAKE_WITHIN_REM, dn)] +
#         data_normalized_by_dataset_name[(FlickerFlavor.NREM_WITHIN_REM, dn)]
#         for dn in DATASET_NAMES
#     ]), axis=0) / np.sqrt(len(DATASET_NAMES))
#
#     # bar plot - normalized by the total time spent in the surrounding state (e.g. NREM for WAKE-within-NREM)
#     spacing = np.linspace(0, 1, bins + 1)
#     x_labels = [f'{spacing[i]:0.1f} - {spacing[i + 1]:0.1f}' for i in range(0, bins)]
#     fig = go.Figure(
#         data=[
#             go.Bar(name='Sleep', x=x_labels, y=y_sleep, error_y=dict(type='data', array=y_sleep_error)),
#             go.Bar(name='Wake', x=x_labels, y=y_wake, error_y=dict(type='data', array=y_wake_error)),
#         ]
#     )
#
#     fig.update_xaxes(title='Proportion of past hour spent in wake')
#     fig.update_yaxes(title='Flickers per minute across all datasets')
#     fig.update_layout(barmode='group', title=f'Flickers rate for region: {"all" if region_filter is None else region_filter.value}')
#
#     # audit dataframe
#     df_audit = None
#     for flicker_flavor, dataset_name in data_normalized_by_dataset_name:
#         df_column = pd.DataFrame({
#             'flicker_rate': data_normalized_by_dataset_name[(flicker_flavor, dataset_name)],
#             'proportion_past_hr_from': np.arange(0, 1, 0.2),
#             'proportion_past_hr_to': np.arange(0, 1, 0.2) + 0.2,
#         })
#         df_column['dataset_name'] = dataset_name
#         df_column['flicker_flavor'] = flicker_flavor.name
#         df_audit = df_audit.append(df_column) if df_audit is not None else df_column
#
#     df_audit.to_csv(f'{BASEPATH_AUDIT}/flickers-by-proportion-wake-aggregated.csv')
#     _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers/flickers-by-proportion-wake-aggregated/flickers-by-proportion-wake-aggregated{"-" if region_filter is not None else ""}{region_filter or ""}')


def plot_flickers_by_time_of_day(flicker_dataframes: List[pd.DataFrame],
                                 interval_hrs: float = 2,
                                 include_flicker_flavors: List[FlickerFlavor] = None,
                                 include_regions: List[Regions] = None,
                                 name: str = None):

    include_regions_list = [r.name for r in (include_regions or Regions) if r != Regions.BADPROBE]
    map_wnr = {'WAKE': 0, 'NREM': 1, 'REM': 2}
    include_flicker_by_flicker = [map_wnr[f.value[0]] for f in (include_flicker_flavors or FlickerFlavor)]
    include_flicker_by_surrounding = [map_wnr[f.value[1]] for f in (include_flicker_flavors or FlickerFlavor)]

    # time of day in seconds split at 1:30am, 7:30am, 1:30pm, 7:30pm.
    time_of_day_splits = np.linspace(0, 86400, 48 + 1)[:-1]  # 0 to 24 hours in half hour increments

    time_split_counts = np.zeros(time_of_day_splits.shape, dtype=np.int64)
    total_time_of_day_sec = np.zeros(time_of_day_splits.shape, dtype=np.float64)

    for df in flicker_dataframes:
        for column in [c for c in df.columns if c.endswith('-flicker-state')]:
            if column.split('-')[0] in include_regions_list:  # region filter
                # normalizing constant calculation
                norm_time_of_day = np.searchsorted(time_of_day_splits, df['time_of_day_sec'].to_numpy(), side='left') - 1
                norm_time_split_ix, norm_time_split_count = np.unique(norm_time_of_day, return_counts=True)
                for tsi, tsc in zip(norm_time_split_ix, norm_time_split_count):
                    total_time_of_day_sec[tsi] += tsc * (1 / df.attrs['fps'])

                # Get first row from each flicker
                flickers = df[column].to_numpy()
                mask_first_row_one_indexed = (flickers[1:] != flickers[:-1]) & (flickers[1:] >= 0)
                index_first_row = mask_first_row_one_indexed.nonzero()[0] + 1

                # Filter by flicker flavor
                mask_included_flickers = np.isin(df.iloc[index_first_row][column], include_flicker_by_flicker)
                mask_included_surrounding_regions = np.isin(df.iloc[index_first_row][column.split('-flicker-state')[0] + '-surrounding-state'], include_flicker_by_surrounding)
                mask_included = np.logical_and(mask_included_flickers, mask_included_surrounding_regions)

                flicker_time_of_day_sec = df.iloc[index_first_row].iloc[mask_included]['time_of_day_sec'].to_numpy()
                flicker_time_of_day = np.searchsorted(time_of_day_splits, flicker_time_of_day_sec, side='left') - 1
                time_split_ix, time_split_count = np.unique(flicker_time_of_day, return_counts=True)

                for tsi, tsc in zip(time_split_ix, time_split_count):
                    time_split_counts[tsi] += tsc

    normalized_time_split_counts = time_split_counts / (total_time_of_day_sec / 60 / 48)

    x = np.linspace(interval_hrs / 2, 24 - interval_hrs / 2, 24 // interval_hrs)
    y = normalized_time_split_counts

    # Shift so the plot starts at 1:30am and centers on the light/dark cycle
    shift = -np.sum(x < 1.5)
    x_shifted = np.roll(x, shift=shift)
    if shift < 0:
        x_shifted[shift:] += 24
    y_shifted = np.roll(y, shift=shift)

    # aggregate by `interval_hrs`
    assert 48 % interval_hrs == 0
    y_aggregate = [np.mean(y_shifted[i:i+int(interval_hrs * 2)]) for i in range(0, y_shifted.shape[0], int(interval_hrs * 2))]

    # Calculate the point at 1:30am and plot it just off screen for continuity of the plot
    if x[0] == 1.5:
        edge_point = y_aggregate[0]
    else:
        percentage_early = (x_shifted[0] - 1.5) / (x_shifted[0] - 1.5 + 25.5 - x_shifted[-1])
        last_first_diff = y_aggregate[-1] - y_aggregate[0]
        edge_point = y_aggregate[0] + last_first_diff * percentage_early

    fig = go.Figure(
        data=[
            # Dark / light overlay
            go.Scatter(
                x=[1.0, 1.0, 7.5, 7.5, None, 19.5, 19.5, 26, 26],
                y=[0, 3000, 3000, 0, None, 0, 3000, 3000, 0],
                fill='toself',
                showlegend=False,
                fillcolor='rgba(0,0,0,0.1)',
                line=dict(width=0),
                marker=dict(size=3),
            ),
            # Data: Line plot
            go.Scatter(
                name='Flicker Rate',
                x=[0.85] + x_shifted.tolist() + [26.15],
                y=[edge_point] + y_aggregate + [edge_point]
            ),
        ]
    )

    fig.add_vline(x=13.5, line_width=1, line_dash="dash", line_color="gray")  # 1:30pm

    fig.update_xaxes(
        title='Time of day (lights on 7:30am, lights off 7:30pm)',
        tickvals=[1.5, 7.5, 13.5, 19.5, 25.5],
        ticktext=['1:30am', '7:30am LightsOn', '1:30pm', '7:30pm LightsOff', '1:30am'],
        range=[1.4, 25.6],
    )
    fig.update_yaxes(
        title='Flickers per minute rate',
        range=[np.min(normalized_time_split_counts) * 0.95, np.max(normalized_time_split_counts) * 1.05]
    )

    filt_flickers = 'allflavors' if include_flicker_flavors is None else '-'.join(['in'.join(fl.value) for fl in include_flicker_flavors])
    filt_regions = 'allregions' if include_regions is None else '-'.join([r.value for r in include_regions])

    fig.update_layout(title=f'Flicker rate by time of day - {filt_regions} - {name + " - " if name else ""}{interval_hrs}hr intervals - Lights on at 7:30am, lights off at 7:30pm')

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers/time-of-day/flickers-by-time-of-day-{name + "-" if name else ""}{interval_hrs}hrs-{filt_flickers}-{filt_regions}')


def plot_unity_line_accuracy_highpass_with_shuffle(shuffled_df: pd.DataFrame, unshuffled_df: pd.DataFrame):
    merged_df = pd.merge(shuffled_df, unshuffled_df, on=['dataset_name', 'channel', 'input_size', 'region'], how='outer', suffixes=['_shuffled', '_unshuffled']).dropna()

    fig = go.Figure()

    # plot data
    fig.add_trace(go.Scatter(
        name='Balanced Accuracy',
        x=merged_df['balanced_accuracy_unshuffled'],
        y=merged_df['balanced_accuracy_shuffled'],
        mode='markers',
        marker=dict(
            size=np.log(merged_df['input_size'].to_numpy()) + 4,
            color=merged_df['region'].apply(lambda x: COLOR_MAP[Regions(x)]),
        ),
    ))

    # add unity line
    fig.add_trace(go.Scatter(
        showlegend=False,
        x=[0, 1.0],
        y=[0, 1.0],
        mode='lines',
        line=dict(color='rgba(0,0,0,0.5)', dash='dash', width=3),
    ))

    fig.update_layout(
        title=f'750hz highpass models balanced accuracy within-sample-shuffled vs unshuffled',
        width=1000, height=1000,
    )
    fig.update_xaxes(range=[0.3, 1.0], title='Balanced Accuracy (750hz raw data unshuffled)')
    fig.update_yaxes(range=[0.3, 1.0], title='Balanced Accuracy (750hz within sample shuffled)')
    fig.update_layout(showlegend=False)

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-highpass-unity-line/figure-highpass-unity-line')


def data_compute_time_to_from_last_state_change(flicker_dataframes: List[pd.DataFrame]):
    # search for time since last state change per each flicker
    time_from_to_state_change = common_utils.map2(
        func=compute_time_to_from_last_state_change_sec,
        fixed_values=dict(mode_filter_window_sec=60),
        args=[(fdf['label_wnr_012'], fdf.attrs['fps']) for fdf in flicker_dataframes],
        parallelism=N_THREADS,
    )

    # append new column with time from and to last state change
    for df, (time_from, time_to) in zip(flicker_dataframes, time_from_to_state_change):
        df['time_since_last_state_change_sec'] = time_from
        df['time_to_next_state_change_sec'] = time_to
        assert np.isnan(time_from).sum() == 0 and np.isnan(time_to).sum() == 0

    return flicker_dataframes


def plot_flickers_scatter_by_time_since_last_state_change(flicker_flavor: FlickerFlavor, flickers_by_flavor: List[Tuple]):
    data_time = [fbf[5].iloc[fbf[4]]['time_since_last_state_change_sec'] for fbf in flickers_by_flavor if fbf[0] == flicker_flavor]
    data_percent = [
        fbf[5].iloc[fbf[4]]['time_since_last_state_change_sec'] /
        (fbf[5].iloc[fbf[4]]['time_since_last_state_change_sec'] + fbf[5].iloc[fbf[4]]['time_to_next_state_change_sec'])
        for fbf in flickers_by_flavor if fbf[0] == flicker_flavor
    ]

    fig = go.Figure(data=[go.Histogram(x=data_time, name='-within-'.join(flicker_flavor.value))])
    fig.update_layout(title=f'Time since last state change for {"-within-".join(flicker_flavor.value)}')
    fig.update_xaxes(title='Time since last state change seconds')
    fig.update_yaxes(title='Number of flickers')

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers/flicker-histogram-{"-within-".join(flicker_flavor.value)}')

    fig = go.Figure(data=[go.Scatter(
        name='-within-'.join(flicker_flavor.value),
        x=data_percent,
        y=data_time,
        mode='markers',
        marker=dict(size=12),
    )])
    fig.update_layout(title=f'Flicker scatter for {"-within-".join(flicker_flavor.value)}')
    fig.update_xaxes(title=f'Percentage of time into state change')
    fig.update_yaxes(title='Time since last state change')

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers/flicker-scatter-{"-within-".join(flicker_flavor.value)}')


@cache.memoize()
def compute_time_to_from_last_state_change_sec(labels: pd.Series, fps: int, mode_filter_window_sec: int):
    window_size = mode_filter_window_sec * fps

    labels_mode_filtered = labels.rolling(window=window_size, center=True).apply(lambda x: scipy.stats.mode(x)[0]).to_numpy()

    state_changes = np.insert(labels_mode_filtered[1:] != labels_mode_filtered[:-1], 0, True)
    frames_since_last_state_change = np.empty(shape=state_changes.shape, dtype=np.int32)
    frames_to_next_state_change = np.empty(shape=state_changes.shape, dtype=np.int32)

    c = 0
    for i in range(len(state_changes)):
        c = 0 if state_changes[i] else c + 1
        frames_since_last_state_change[i] = c
    for i in range(len(state_changes) - 1, -1, -1):
        c = 0 if state_changes[i] else c + 1
        frames_to_next_state_change[i] = c

    time_since_last_state_change_sec = frames_since_last_state_change.astype(np.float32) * 1 / fps
    time_to_next_state_change_sec = frames_to_next_state_change.astype(np.float32) * 1 / fps

    return time_since_last_state_change_sec, time_to_next_state_change_sec


def remove_prefixes(s: str, prefixes: Union[List[str], Tuple[str], str]) -> str:
    """ Removes one or more prefixes from the beginning of a string. """
    prefixes = [prefixes] if isinstance(prefixes, str) else prefixes
    for p in prefixes:
        if s.startswith(p):
            s = s[len(p):]
    return s


# noinspection PyUnusedLocal
@cache.memoize()
def load_optical_flow_dataframe(url: str, last_modified_timestamp: str) -> pd.DataFrame:
    """ Load a single optical flow dataframe, cacheable. """
    with smart_open.open(url, 'rb') as f:
        df = pd.read_csv(f)

    # if 60 fps then double the index numbering so it matches the labels file
    # get fps
    dataset_name = [d for d in DATASET_NAMES if d in url][0]
    fps = _get_fps(dataset_name)
    if fps == 30:
        df['frame_index'] = df['frame_index'].apply(lambda x: x * 2)

    # fix filename bug
    parsed_filename = remove_prefixes(os.path.basename(url).split('.')[0], [f'{d}_' for d in DATASET_NAMES]) + '.mp4'
    df['filename'] = parsed_filename

    # convert NaN to 0.0
    df = df.fillna(0.0)

    df.attrs['s3url'] = url

    return df


# noinspection PyUnusedLocal
@cache.memoize()
def load_predictions_file(url: str, last_modified_timestamp: str) -> pd.DataFrame:
    """ Load the predictions file from S3, cacheable, returned as DataFrame """
    with smart_open.open(url, 'rb') as f:
        with zipfile.ZipFile(f) as z:
            zn = z.namelist()
            assert len(zn) == 1
            buff = z.read(zn[0])
            df = pd.read_csv(io.BytesIO(buff))
            return df


def get_optical_flow_dataframe(dataset_names: List[str],
                               long_median_window: int,
                               short_median_window: int,
                               use_cached: bool = True) -> pd.DataFrame:
    """ List all optical flow dataframes. Merge with the relevant labels file. Return merged DataFrame. """

    all_optical_flow_objects = s3wrangler.describe_objects(f's3://hengenlab/optical_flow/results/')
    all_optical_flow_url_last_modified = [(s3url, str(s3dict['LastModified'])) for s3url, s3dict in all_optical_flow_objects.items()]
    # note: prediction files do not exactly match label files in all cases, hence label files are not used here
    all_prediction_objects = s3wrangler.describe_objects([
        f's3://hengenlab/{dataset_name}/Runs/wnr-v14-perregion-c64k-0-64-run1/Results/predictions_{dataset_name}.csv.zip'
        for dataset_name in dataset_names
    ])
    all_prediction_url_last_modified = [(s3url, str(s3dict['LastModified'])) for s3url, s3dict in all_prediction_objects.items()]

    # cache the cpu intensive merge operation
    cache_key = (
        '__get_optical_flow_dataframe__',
        dataset_names,
        long_median_window,
        short_median_window,
        all_optical_flow_url_last_modified,
        all_prediction_url_last_modified
    )
    merged_df = cache.get(cache_key) if use_cached else None

    if merged_df is None:

        # Optical flow data
        all_optical_flow_dfs = common_utils.map2(
            func=load_optical_flow_dataframe,
            args=all_optical_flow_url_last_modified,
            parallelism=N_THREADS,
        )
        # assert np.all([len(df) > 53000 for df in all_optical_flow_dfs]), \
        #     'Sanity check that optical flow dataframes are 1/15th of a sec.'

        # rename and remove extraneous columns, rename is needed for merge to work below
        all_optical_flow_dfs = [
            df.rename(columns={'frame_index': 'video_frame_offset', 'filename': 'video_filename'}).drop(columns=['Unnamed: 0'], errors='ignore')
            for df in all_optical_flow_dfs
        ]

        # predictions files
        all_predictions_dfs = common_utils.map2(
            func=load_predictions_file,
            args=all_prediction_url_last_modified,
            parallelism=N_THREADS,
        )
        all_predictions_dfs = [df.assign(dataset_name=dn) for df, dn in zip(all_predictions_dfs, dataset_names)]
        # add index column
        for df in all_predictions_dfs:
            assert len(df) > 53500, 'Sanity check that approx 1 hour of frames exist.'
            df['ix'] = range(0, len(df))

        optical_flow_df = pd.concat(all_optical_flow_dfs)
        pred_df = pd.concat(all_predictions_dfs)
        del all_optical_flow_dfs, all_predictions_dfs

        # merge labels and optical flow dataframes
        optical_flow_df.rename(columns={'video_frame_offset': 'video_frame_ix'}, inplace=True)
        merged_df = pd.merge(
            pred_df, optical_flow_df,
            on=['video_filename', 'video_frame_ix'],
            how='left',
        )

        # add column with normalized flow and median filter
        merged_df['percentile_normalized_flow'] = np.nan  # add the column for normalized flow
        merged_df['median_filter_percentile_normalized_flow'] = np.nan
        merged_df['local_median_filter_percentile_normalized_flow'] = np.nan
        for dataset_name in DATASET_NAMES:
            mask = (merged_df['dataset_name'] == dataset_name) & (~merged_df['normalized_flow'].isna())
            df = merged_df.loc[mask]

            # normalized flow
            percentile = df['normalized_flow'].quantile(0.75)
            percentile_normalized_flow = df['normalized_flow'] / percentile * 0.75
            merged_df.at[mask, 'percentile_normalized_flow'] = percentile_normalized_flow

            # median filter over normalized flow
            median_line = percentile_normalized_flow.clip(lower=0.0, upper=1.0).rolling(window=long_median_window, min_periods=1, center=True).median()
            merged_df.at[mask, 'median_filter_percentile_normalized_flow'] = median_line

            # short median filter over normalized flow
            local_median_line = percentile_normalized_flow.clip(lower=0.0, upper=1.0).rolling(window=short_median_window, min_periods=1, center=True).median()
            merged_df.at[mask, 'local_median_filter_percentile_normalized_flow'] = local_median_line

        cache[cache_key] = merged_df

    return merged_df


def _get_normalized_flow(flicker: Tuple, optical_flow_dataframes: Dict[str, pd.DataFrame]):
    """ Used by plot_flickers_against_motion to compute normalized flow """
    optical_flow_dataframe = optical_flow_dataframes[flicker[3]]
    flicker_loc = optical_flow_dataframe['ix'] == flicker[4]
    optical_flow_flicker = optical_flow_dataframe.loc[flicker_loc]
    assert len(optical_flow_flicker) == 1
    optical_flow_val = optical_flow_flicker['normalized_flow'].values[0]
    flicker += (optical_flow_val,)

    return flicker


def compute_flicker_rate_per_optical_flow(flow_per_flicker_valid_only: pd.Series, wake_state_optical_flow: pd.Series, bins: (int, np.ndarray)):
    flow_per_flicker_valid_only = flow_per_flicker_valid_only.to_numpy()
    wake_state_optical_flow = wake_state_optical_flow.to_numpy()

    bins = np.linspace(0, 1, bins) if isinstance(bins, int) else bins
    hist_f, _ = np.histogram(np.clip(flow_per_flicker_valid_only, 0.0, 1.0), bins=bins)
    hist_o, _ = np.histogram(np.clip(wake_state_optical_flow, 0.0, 1.0), bins=bins)
    y = hist_f / (hist_o / 60 / 15)

    return y, bins


def get_wake_state_optical_flow_per_dataset_rescaled(flicker_dataframes: List[pd.DataFrame], optical_flow_dataframe: pd.DataFrame):
    """ Split the optical flow dataframe into per dataset dataframes and rescales wake-state to 75th percentile """
    flickers_by_flavor = get_flickers_by_flavor(flicker_dataframes)

    optical_flow_dataframes_per_dataset_name_all = {
        dataset_name: optical_flow_dataframe.loc[optical_flow_dataframe['dataset_name'] == dataset_name]
        for dataset_name in DATASET_NAMES
    }
    for dsn, df in optical_flow_dataframes_per_dataset_name_all.items():
        df.set_index(keys=['ix'])
        # Rescale
        wake_state = df['label_wnr_012'] == 1
        quantile = df.loc[wake_state]['normalized_flow'].dropna().quantile(0.75)
        normalized_flow = df['normalized_flow'].to_numpy()
        normalized_flow_rescaled = normalized_flow / quantile * 0.75
        df['normalized_flow'] = normalized_flow_rescaled
        df.attrs['wake_state'] = wake_state

    wake_flickers = list(common_utils.map2(
        func=_get_normalized_flow,
        args=[fbf for fbf in flickers_by_flavor if fbf[0].value[1] == 'WAKE'],
        fixed_values=dict(optical_flow_dataframes=optical_flow_dataframes_per_dataset_name_all),
        parallelism=1,
    ))
    df_wake_flickers = pd.DataFrame({
        'flicker_flavor': [f[0].name for f in wake_flickers],
        'time_since_last_state_change': [f[1] for f in wake_flickers],
        'column': [f[2] for f in wake_flickers],
        'dataset_name': [f[3] for f in wake_flickers],
        'ix': [f[4] for f in wake_flickers],
        'flow': [f[6] for f in wake_flickers],
    })

    return optical_flow_dataframes_per_dataset_name_all, df_wake_flickers


def plot_flickers_against_motion(flicker_dataframes: List[pd.DataFrame], optical_flow_dataframe: pd.DataFrame, bins: int = 25):
    """ Plot histogram of movement of wake-state flickers. """
    # calculate movement per flicker
    # flickers_by_flavor = get_flickers_by_flavor(flicker_dataframes)
    optical_flow_dataframes_per_dataset_name_all, df_wake_flickers = \
        get_wake_state_optical_flow_per_dataset_rescaled(flicker_dataframes, optical_flow_dataframe)

    # Calculate rate per minute for flickers
    flow_per_flicker_valid_only = df_wake_flickers['flow'].dropna()
    wake_state_optical_flow = pd.concat([df.loc[df.attrs['wake_state']]['normalized_flow'].dropna() for df in optical_flow_dataframes_per_dataset_name_all.values()])
    y, bin_edges = compute_flicker_rate_per_optical_flow(flow_per_flicker_valid_only, wake_state_optical_flow, bins)

    # Compute y per each animal for std err
    y_per_dataset_name = []
    for dataset_name in DATASET_NAMES:
        ds_mask = df_wake_flickers['dataset_name'] == dataset_name
        flow_per_flicker_valid_only_per_ds = df_wake_flickers.loc[ds_mask]['flow'].dropna()
        wake_state_optical_flow_only_per_ds = pd.concat([
            df.loc[df.attrs['wake_state'] & (df['dataset_name'] == dataset_name)]['normalized_flow'].dropna()
            for df in optical_flow_dataframes_per_dataset_name_all.values()
        ])
        y_per_dataset_name.append(
            compute_flicker_rate_per_optical_flow(
                flow_per_flicker_valid_only_per_ds,
                wake_state_optical_flow_only_per_ds,
                bins
            )[0]
        )
    error_y = np.array(y_per_dataset_name).std(axis=0) / np.sqrt(len(DATASET_NAMES)).clip(0.0)

    fig = go.Figure(data=[go.Bar(
        x=bin_edges,
        y=y,
        error_y=dict(type='data', array=error_y),
        marker_color='red',
    )])

    ymax = (max(error_y) + max(y)) * 1.05
    fig.update_layout(title=f'Flickers by movement - wake flickers only - error per animal', showlegend=False)
    fig.update_xaxes(title=f'Movement measured by optical flow')
    fig.update_yaxes(title='Flicker rate per minute', range=[0, ymax])

    # Vertical line at quiescence vs active, line chosen by looking at videos for 5 datasets and rescaling from
    # video overlay to scaled values computed in this function. Datasets: CAF106|99|42|69 EAB40
    fig.add_trace(go.Scatter(x=[0.2223, 0.2223], y=[0, ymax * 0.95], mode='lines', showlegend=False))
    fig.add_annotation(text='Quiescence', xref='x', yref='y', x=0.2223-0.01, y=ymax * 0.65, textangle=-90)
    fig.add_annotation(text='Active Wake', xref='x', yref='y', x=0.2223+0.04, y=ymax * 0.65, textangle=90)

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers/flickers-against-motion')

    fig_box = go.Figure()
    for dataset_name in DATASET_NAMES:
        y = optical_flow_dataframes_per_dataset_name_all[dataset_name]
        y = y.loc[y.attrs['wake_state']]['normalized_flow'].dropna()
        fig_box.add_trace(go.Box(y=y, name=dataset_name, boxpoints='outliers'))

    fig_box.update_layout(title=f'Wake state optical flow.')
    fig_box.update_xaxes(title=f'Distribution of wake state optical flow per animal.')
    fig_box.update_yaxes(title='Optical flow values in [0, 1] range.')

    _save_figure(fig_box, f'{BASEPATH_FIGURES}/figure-flickers/optical_flow_box_whiskers')


def plot_flickers_by_quiescence_and_active(flicker_dataframes: List[pd.DataFrame], optical_flow_dataframe: pd.DataFrame):
    active_quiescence_threshold = 0.2223

    optical_flow_dataframes_per_dataset_name, df_wake_flickers = \
        get_wake_state_optical_flow_per_dataset_rescaled(flicker_dataframes, optical_flow_dataframe)

    active_mask = optical_flow_dataframe['normalized_flow'] >= active_quiescence_threshold
    quiescence_mask = optical_flow_dataframe['normalized_flow'] < active_quiescence_threshold

    y = []
    y_error = []
    p_val = []

    for dataset_name in DATASET_NAMES:
        # Calculate rate per minute for flickers
        flow_per_flicker = df_wake_flickers['flow'].dropna()
        wake_state_optical_flow = pd.concat([
            df.loc[df.attrs['wake_state']]['normalized_flow'].dropna()
            for df in optical_flow_dataframes_per_dataset_name.values()
        ])
        y, bin_edges = compute_flicker_rate_per_optical_flow(flow_per_flicker, wake_state_optical_flow, bins=[0.0, active_quiescence_threshold, 1.0])


    fig = go.Figure()


def merge_flicker_optical_flow(flicker_dataframes: List[pd.DataFrame], optical_flow_dataframe: pd.DataFrame, use_cache: bool = True):
    """ merge into one dataframe, downsample to 15fps, clean up nans. """
    cache_key = (
        '__merge_flicker_optical_flow__',
        [pd.util.hash_pandas_object(fd).sum() for fd in flicker_dataframes],
        pd.util.hash_pandas_object(optical_flow_dataframe).sum(),
    )
    df = cache.get(cache_key) if use_cache else None

    if df is None:
        # add column that identifies each flicker,
        #   `-1` indicates no flicker,
        #   `N` (>0) indicates the start of a flicker of length N
        #   `0` indicates a segment in which a flicker is active
        optical_flow_dataframe['flicker'] = -1
        optical_flow_dataframe.attrs['docs_flicker_column'] = \
            'Flicker column usage: ' \
            '(-1) is no flicker; ' \
            '(N) >0 indicates the start of a flicker of length N' \
            '(0) indicates a segment in which a flicker is active.'

        # add column that identifies flicker type
        optical_flow_dataframe['flicker_flavor'] = None
        optical_flow_dataframe['flicker_region'] = None

        # add columns to optical flow for flicker start and length
        flickers_by_flavor = get_flickers_by_flavor(flicker_dataframes=flicker_dataframes, as_dataframe=True)
        ix_by_dataset_name = {dn: np.where((optical_flow_dataframe['dataset_name'] == dn).to_numpy())[0] for dn in DATASET_NAMES}
        df_list = []
        for _, flicker in flickers_by_flavor.iterrows():
            fl = flicker['flicker_length']
            ix = ix_by_dataset_name[flicker['dataset_name']][flicker['ix']]
            region = Regions[flicker['column_name_flicker_calling'].split('-')[0]].value

            # note flicker length is not set here, it's set lower after downsampling has occurred for 30fps
            optical_flow_dataframe.iloc[ix:ix + fl]['flicker'] = 0
            optical_flow_dataframe.iloc[ix:ix + fl]['flicker_flavor'] = flicker['flicker_flavor']
            optical_flow_dataframe.iloc[ix:ix + fl]['flicker_region'] = region

        # interpolate optical flow values & downsample to 15fps by iterating through each dataset
        # excluding datasets with no optical flow currently computed
        for dn, flicker_dataframe in zip(DATASET_NAMES, flicker_dataframes):
            dn_ixs = ix_by_dataset_name[dn]
            dn_df = optical_flow_dataframe.iloc[dn_ixs]

            assert flicker_dataframe.attrs['dataset_name'] == dn and len(dn_df) == len(flicker_dataframe)
            dn_df['time_since_last_state_change_sec'] = pd.Series(flicker_dataframe['time_since_last_state_change_sec'].to_numpy(), index=dn_df.index)
            dn_df['time_to_next_state_change_sec'] = pd.Series(flicker_dataframe['time_to_next_state_change_sec'].to_numpy(), index=dn_df.index)
            assert dn_df['time_since_last_state_change_sec'].isna().sum() == 0 and dn_df['time_to_next_state_change_sec'].isna().sum() == 0

            dn_df['proportion_wake_3600s_window'] = pd.Series(flicker_dataframe['proportion_wake_3600s_window'].to_numpy(), index=dn_df.index)

            # interpolate away NaNs, NaNs exist every other sample for 30fps video because optical flow
            # was calculated at 15fps, and there are a few NaNs at the end of the file in some cases,
            # limit=10 accounts for the end cases, no more than 10 long, which should account for all the
            # edge cases.
            dn_df['local_median_filter_percentile_normalized_flow'] = dn_df['local_median_filter_percentile_normalized_flow'].interpolate(method='linear', limit=10, limit_direction='both')
            dn_df['median_filter_percentile_normalized_flow'] = dn_df['median_filter_percentile_normalized_flow'].interpolate(method='linear', limit=10, limit_direction='both')
            dn_df['percentile_normalized_flow'] = dn_df['percentile_normalized_flow'].interpolate(method='linear', limit=10, limit_direction='both')
            assert dn_df['local_median_filter_percentile_normalized_flow'].isna().sum() == 0
            assert dn_df['median_filter_percentile_normalized_flow'].isna().sum() == 0
            assert dn_df['percentile_normalized_flow'].isna().sum() == 0

            # downsample 30fps datasets
            if _get_fps(dn) == 30:
                dn_df = dn_df.iloc[::2]
                dn_df['ix'] = np.arange(len(dn_df))

            # compute flicker begin and length based on 15fps downsampled data
            is_flicker = np.pad(dn_df['flicker'].fillna(-1).to_numpy(), pad_width=1, mode='constant', constant_values=-1)
            ix_pairs = np.where(is_flicker[1:] != is_flicker[:-1])[0]  # produces a list of [start0, end0, start1, end1, ...] pairs indexing into dn_df['flickers']
            assert len(ix_pairs) % 2 == 0
            for i in range(0, len(ix_pairs), 2):
                dn_df.iloc[ix_pairs[i]:ix_pairs[i] + 1]['flicker'] = ix_pairs[i+1] - ix_pairs[i]  # set flicker length on start of flicker

            # save results back to larger df
            df_list.append(dn_df)

        df = pd.concat(df_list)

        # Add columns that identify whether each data point is segmented into [wide and short] band high activity (dark green and dark blue segments in flicker raster plot)
        df['is_in_wide_band_high_activity'] = df['median_filter_percentile_normalized_flow'] > LONG_MEDIAN_THRESHOLD
        df['time_in_wide_band_high_activity_forward'] = df['is_in_wide_band_high_activity'].cumsum() - df['is_in_wide_band_high_activity'].cumsum().where(~df['is_in_wide_band_high_activity']).ffill().fillna(0).astype(int)
        df['time_in_wide_band_high_activity_reverse'] = (df['is_in_wide_band_high_activity'][::-1].cumsum() - df['is_in_wide_band_high_activity'][::-1].cumsum().where(~df['is_in_wide_band_high_activity'][::-1]).ffill().fillna(0).astype(int))[::-1]
        df['length_segment_wide_band_high_activity'] = df['time_in_wide_band_high_activity_forward'] + df['time_in_wide_band_high_activity_reverse'] - 1

        df['is_in_wide_band_low_activity'] = df['median_filter_percentile_normalized_flow'] <= LONG_MEDIAN_THRESHOLD
        df['time_in_wide_band_low_activity_forward'] = df['is_in_wide_band_high_activity'].cumsum() - df['is_in_wide_band_high_activity'].cumsum().where(~df['is_in_wide_band_high_activity']).ffill().fillna(0).astype(int)
        df['time_in_wide_band_low_activity_reverse'] = (df['is_in_wide_band_high_activity'][::-1].cumsum() - df['is_in_wide_band_high_activity'][::-1].cumsum().where(~df['is_in_wide_band_high_activity'][::-1]).ffill().fillna(0).astype(int))[::-1]
        df['length_segment_wide_band_low_activity'] = df['time_in_wide_band_low_activity_forward'] + df['time_in_wide_band_low_activity_reverse'] - 1

        df['is_in_short_band_high_activity'] = df['local_median_filter_percentile_normalized_flow'] > SHORT_MEDIAN_THRESHOLD
        df['time_in_short_band_high_activity_forward'] = df['is_in_short_band_high_activity'].cumsum() - df['is_in_short_band_high_activity'].cumsum().where(~df['is_in_short_band_high_activity']).ffill().fillna(0).astype(int)
        df['time_in_short_band_high_activity_reverse'] = (df['is_in_short_band_high_activity'][::-1].cumsum() - df['is_in_short_band_high_activity'][::-1].cumsum().where(~df['is_in_short_band_high_activity'][::-1]).ffill().fillna(0).astype(int))[::-1]
        df['percent_time_short_band_high_activity'] = df['time_in_short_band_high_activity_forward'] / (df['time_in_short_band_high_activity_forward'] + df['time_in_short_band_high_activity_reverse'])
        df['length_segment_short_band_high_activity'] = df['time_in_short_band_high_activity_forward'] + df['time_in_short_band_high_activity_reverse'] - 1

        df['is_in_short_band_low_activity'] = df['local_median_filter_percentile_normalized_flow'] <= SHORT_MEDIAN_THRESHOLD
        df['time_in_short_band_low_activity_forward'] = df['is_in_short_band_low_activity'].cumsum() - df['is_in_short_band_low_activity'].cumsum().where(~df['is_in_short_band_low_activity']).ffill().fillna(0).astype(int)
        df['time_in_short_band_low_activity_reverse'] = (df['is_in_short_band_low_activity'][::-1].cumsum() - df['is_in_short_band_low_activity'][::-1].cumsum().where(~df['is_in_short_band_low_activity'][::-1]).ffill().fillna(0).astype(int))[::-1]
        df['percent_time_short_band_low_activity'] = df['time_in_short_band_low_activity_forward'] / (df['time_in_short_band_low_activity_forward'] + df['time_in_short_band_low_activity_reverse'])
        df['length_segment_short_band_low_activity'] = df['time_in_short_band_low_activity_forward'] + df['time_in_short_band_low_activity_reverse'] - 1

        # Add column for number of probes in each dataset
        map_n_probes = {dn: len([p for p in PROBE_REGION_CHANNEL_LIST if p.region != Regions.BADPROBE and p.dataset_name == dn]) for dn in DATASET_NAMES}
        df['n_probes'] = df['dataset_name'].map(map_n_probes)

        cache[cache_key] = df
        df.to_csv(f'{BASEPATH_AUDIT}/optical-flow-flickers-df.csv')

    return df


def plot_flow_after_flicker(optical_flow_flickers_df: pd.DataFrame, movement_threshold: float, n_frames: int = 75):
    """ Bar plot of optical flow values following onset of an marked flicker """
    marked_flickers_mask = \
        (optical_flow_flickers_df['flicker_flavor'] == FlickerFlavor.NREM_WITHIN_WAKE.name) & \
        (optical_flow_flickers_df['flicker'] > 0) & \
        (optical_flow_flickers_df['median_filter_percentile_normalized_flow'] > movement_threshold)

    marked_flickers_ixs = np.where(marked_flickers_mask)[0]
    assert len(marked_flickers_ixs) > 0

    xy = [
        list(zip(
            np.arange(-75, 76),
            optical_flow_flickers_df.iloc[ix - n_frames:ix+n_frames + 1]['percentile_normalized_flow'],
        ))
        for ix in marked_flickers_ixs
    ]
    x, y = zip(*itertools.chain(*xy))
    x = np.array(x)
    y = np.clip(np.array(y), 0, 1)
    mean_line = np.mean(np.clip(np.array(xy)[:,:,1], 0, 1), axis=1)
    assert not np.any(np.isnan(mean_line))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        name='OpticalFlow',
        x=(x + (np.random.rand(x.shape[0]) - 0.5) * 0.0) / 15,  # -0.5 for random [-0.5, +0.5], *0.8 to shrink it to +/-0.4, /15 to convert to seconds on x axis
        y=y,
        mode='markers',
        marker=dict(color='rgba(128,128,0,0.1)', size=10, line=dict(width=0), symbol='square'),
    ))
    fig.add_trace(go.Scatter(
        name='Mean',
        x=np.arange(-75, 76) / 15,
        y=mean_line,
        mode='lines',
    ))

    fig.update_layout(
        title=f'Optical flow surrounding flicker onset for flickers in regions of high activity, NREM_WITHIN_WAKE, 0.75 movement threshold',
        width=1800, height=800,
    )
    fig.update_xaxes(
        title=f'Seconds before (negative) or after (positive) flicker onset with random jitter for visualization',
        showgrid=False,
        tickformat='%d',
    )
    fig.update_yaxes(
        title='Optical flow values in [0, 1] range below, flicker raster above, wake|nrem|rem overlay.',
        showgrid=False,
    )

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flicker-flow-surrounding/flicker-flow-surrounding')


def plot_flicker_raster_with_motion(optical_flow_flickers_df: pd.DataFrame, long_median_threshold: float, short_median_threshold: float, plot_range: Tuple[int] = None):
    """ Plot optical flow value with a flicker raster per region above. """

    # extract dataset_name and validate that only one dataset is present
    dataset_name = optical_flow_flickers_df['dataset_name'].unique()
    assert len(dataset_name) == 1
    dataset_name = dataset_name[0]

    fig = go.Figure()
    shapes = []

    # Plot sleep wake overlay
    sleep_state = np.pad(optical_flow_flickers_df['label_wnr_012'].to_numpy(), pad_width=1, mode='constant', constant_values=-1)
    long_median = np.pad(optical_flow_flickers_df['median_filter_percentile_normalized_flow'].to_numpy(), pad_width=1, mode='constant', constant_values=-1)
    short_median = np.pad(optical_flow_flickers_df['local_median_filter_percentile_normalized_flow'].to_numpy(), pad_width=1, mode='constant', constant_values=-1)

    is_wake_mask = sleep_state == 0
    is_nrem_mask = sleep_state == 1
    is_rem_mask = sleep_state == 2
    is_high_activity_long_mask = long_median > long_median_threshold
    is_high_activity_short_mask = short_median > short_median_threshold

    is_wake_ixs = np.where(is_wake_mask[1:] != is_wake_mask[:-1])[0]
    is_nrem_ixs = np.where(is_nrem_mask[1:] != is_nrem_mask[:-1])[0]
    is_rem_ixs = np.where(is_rem_mask[1:] != is_rem_mask[:-1])[0]
    is_high_activity_long_ixs = np.where(is_high_activity_long_mask[1:] != is_high_activity_long_mask[:-1])[0]
    is_high_activity_short_ixs = np.where(is_high_activity_short_mask[1:] != is_high_activity_short_mask[:-1])[0]

    assert not optical_flow_flickers_df['percentile_normalized_flow'].hasnans
    x_range = (np.arange(0, len(optical_flow_flickers_df), dtype=np.float32) / 15)

    # Plot median filters
    fig.add_trace(go.Scatter(
        x=x_range,
        y=optical_flow_flickers_df['median_filter_percentile_normalized_flow'],
        mode='lines',
        name='LongMedianFlow',
        line=dict(color='rgba(200,200,200,0.65)', width=4),
        visible="legendonly"
    ))
    fig.add_trace(go.Scatter(
        x=x_range,
        y=optical_flow_flickers_df['local_median_filter_percentile_normalized_flow'],
        mode='lines',
        name='ShortMedianFlow',
        line=dict(color='rgba(200,200,255,0.65)', width=4),
        visible="legendonly"
    ))

    # Plot motion
    fig.add_trace(go.Scatter(
        x=x_range,
        y=optical_flow_flickers_df['percentile_normalized_flow'].clip(lower=0.0, upper=1.0),
        mode='lines',
        name='OpticalFlow',
        line=dict(color='rgba(0,0,0,0.7)', width=0.1),
        fill='tozeroy',
        fillcolor='rgba(0,0,0,0.5)',  # pick a nice color
    ))

    # Plot flicker raster
    flicker_flavors_sorted = [
        FlickerFlavor.NREM_WITHIN_WAKE, FlickerFlavor.REM_WITHIN_WAKE,
        FlickerFlavor.REM_WITHIN_NREM, FlickerFlavor.WAKE_WITHIN_NREM,
        FlickerFlavor.NREM_WITHIN_REM, FlickerFlavor.WAKE_WITHIN_REM,
    ]
    for i, flicker_flavor in enumerate(flicker_flavors_sorted):
        mask_flickers = (optical_flow_flickers_df['flicker'] > 0).to_numpy()
        mask_flavor = (optical_flow_flickers_df['flicker_flavor'] == flicker_flavor.name).to_numpy()
        mask_high_activity = (optical_flow_flickers_df['median_filter_percentile_normalized_flow'] > LONG_MEDIAN_THRESHOLD).to_numpy()

        # flicker_x = optical_flow_flickers_df[mask_flickers & mask_flavor]['ix']
        flicker_x_unmarked = optical_flow_flickers_df.iloc[mask_flickers & mask_flavor & ~mask_high_activity]['ix']
        flicker_x_marked = optical_flow_flickers_df.iloc[mask_flickers & mask_flavor & mask_high_activity]['ix']
        flicker_x1_unmarked = optical_flow_flickers_df.iloc[mask_flickers & mask_flavor & ~mask_high_activity]['ix'] + optical_flow_flickers_df.iloc[mask_flickers & mask_flavor & ~mask_high_activity]['flicker']
        flicker_x1_marked = optical_flow_flickers_df.iloc[mask_flickers & mask_flavor & mask_high_activity]['ix'] + optical_flow_flickers_df.iloc[mask_flickers & mask_flavor & mask_high_activity]['flicker']
        
        y_unmarked = np.ones(len(flicker_x_unmarked), dtype=np.int32) * (1.09 + 0.03 * i)
        y_marked = np.ones(len(flicker_x_marked), dtype=np.int32) * (1.09 + 0.03 * i)
        # fig.add_trace(go.Scatter(
        #     x=flicker_x / 15,
        #     y=np.ones(len(flicker_x), dtype=np.int32) * (1.09 + 0.03 * i),
        #     mode='markers',
        #     marker=dict(symbol='diamond-tall', size=13),
        #     name=f'{flicker_flavor.value[0]}-IN-{flicker_flavor.value[1]}',
        # ))
        fig.add_trace(go.Scatter(
            x=flicker_x_unmarked / 15,
            y=y_unmarked,
            mode='markers',
            marker=dict(symbol='diamond-tall', size=13),
            name=f'{flicker_flavor.value[0]}-IN-{flicker_flavor.value[1]}',
        ))
        fig.add_trace(go.Scatter(
            x=flicker_x_marked / 15,
            y=y_marked,
            mode='markers',
            marker=dict(symbol='x', size=13),
            name=f'{flicker_flavor.value[0]}-IN-{flicker_flavor.value[1]}',
        ))
        shapes.extend([
            go.layout.Shape(
                type='line',
                x0=x0, x1=x1, y0=y, y1=y,
                line=dict(color='black', width=1),
                xref='x', yref='y',
                layer='below',
            )
            for x0, x1, y in zip(flicker_x_unmarked / 15, flicker_x1_unmarked / 15, y_unmarked)
        ])
        shapes.extend([
            go.layout.Shape(
                type='line',
                x0=x0, x1=x1, y0=y, y1=y,
                line=dict(color='black', width=1),
                xref='x', yref='y',
                layer='below',
            )
            for x0, x1, y in zip(flicker_x_marked / 15, flicker_x1_marked / 15, y_marked)
        ])

    # Plot background label colors
    for color, ixs in zip(['rgba(50,168,82,1)', 'rgba(35,91,186,1)', 'rgba(194,29,40,1)'], [is_wake_ixs, is_nrem_ixs, is_rem_ixs]):
        assert len(ixs) % 2 == 0
        ixs_pairs = np.array_split(ixs, len(ixs)//2)
        for x0, x1 in ixs_pairs:
            # fig.add_shape(
            shapes.append(go.layout.Shape(
                type='rect',
                x0=x0 / 15, x1=x1 / 15,
                y0=-0.075, y1=1,
                fillcolor=color,
                line=dict(color='rgba(0,0,0,0)', width=0),
                xref='x', yref='y',
                layer='below',
            ))

    # Plot binarized segmentation
    vertoffset_color_ixs = zip(
        [0, 0, 1, 1],
        ['darkgreen', 'lightgreen', 'darkblue', 'lightblue'],
        [is_high_activity_long_ixs, is_high_activity_long_ixs[1:-1],
         is_high_activity_short_ixs, is_high_activity_short_ixs[1:-1]]
    )
    for vert_offset, color, ixs in vertoffset_color_ixs:
        assert len(ixs) % 2 == 0
        ixs_pairs = np.array_split(ixs, len(ixs)//2)
        for j, (x0, x1) in enumerate(ixs_pairs):
            shapes.append(go.layout.Shape(
                type='rect',
                x0=x0 / 15,
                x1=x1 / 15,
                y0=1.01 + vert_offset * 0.03,
                y1=1.03 + vert_offset * 0.03,
                fillcolor=color,
                line=dict(color='rgba(0,0,0,0)', width=0),
                xref='x', yref='y',
                layer='below',
            ))

    fig.update_layout(
        title=f'Flicker raster {dataset_name}',
        width=1800, height=800,
        shapes=shapes,
    )
    fig.update_xaxes(
        title=f'Time in seconds, 24hrs = 86,400s',
        showgrid=False,
        tickformat='%d',
        range=plot_range,
    )
    fig.update_yaxes(
        title='Optical flow values in [0, 1] range below, flicker raster above, wake|nrem|rem overlay.',
        range=[-0.075, 1.27],
        showgrid=False,
    )

    plot_range_text = f'-range-{plot_range[0]}-{plot_range[1]}' if plot_range is not None else ''
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flicker-raster/flicker-raster-{dataset_name}{plot_range_text}', save_formats=('svg', 'html', 'pdf'))


def compute_flicker_rate_distribution(optical_flow_flickers_df: pd.DataFrame,
                                      dataset_names: List[str],
                                      flicker_flavors: List[FlickerFlavor],
                                      regions: List[Regions],
                                      n_bins: int = 20,
                                      filter_high_activity_edges: float = 0.25,
                                      min_short_band_segment_size: int = 10 * 15,
                                      normalize_by_total_flicker_rate: bool = True):
    df = optical_flow_flickers_df  # shorthand for readability

    df['weighted_flicker'] = 1 / df['n_probes']  # weighted by number of probes, for counting flickers
    mask_flickers = df['dataset_name'].isin(dataset_names) & \
                    (df['flicker_flavor'].isin([f.name for f in flicker_flavors])) & \
                    (df['flicker_region'].isin([r.value for r in regions])) & \
                    (df['flicker'] > 0)
    mask_is_not_edge_short_high = ((1 - filter_high_activity_edges) >= df['percent_time_short_band_high_activity']) & (df['percent_time_short_band_high_activity'] >= filter_high_activity_edges)  # todo testing
    mask_is_not_small_segment_short_band_high = df['length_segment_short_band_high_activity'] >= min_short_band_segment_size
    mask_is_wake = (df['label_wnr_012'] == 0) & df['dataset_name'].isin(dataset_names)
    mask_high_activity = mask_flickers & mask_is_wake & df['is_in_wide_band_high_activity'] & df['is_in_short_band_high_activity'] & mask_is_not_edge_short_high & mask_is_not_small_segment_short_band_high  # todo testing
    mask_low_activity = mask_flickers & mask_is_wake & df['is_in_wide_band_high_activity'] & df['is_in_short_band_low_activity']

    mask_is_quiescence_period = ~df['is_in_wide_band_high_activity'] & mask_is_wake & df['dataset_name'].isin(dataset_names)
    mask_is_active_wake_period = df['is_in_wide_band_high_activity'] & df['is_in_short_band_high_activity'] & mask_is_wake & df['dataset_name'].isin(dataset_names) & mask_is_not_edge_short_high & mask_is_not_small_segment_short_band_high
    mask_is_pause_period = df['is_in_wide_band_high_activity'] & df['is_in_short_band_low_activity'] & mask_is_wake & df['dataset_name'].isin(dataset_names)

    wake_sum = mask_is_quiescence_period.sum() + mask_is_active_wake_period.sum() + mask_is_pause_period.sum()
    fraction_time_in_quiescence_period = mask_is_quiescence_period.sum() / wake_sum
    fraction_time_in_active_wake_period = mask_is_active_wake_period.sum() / wake_sum
    fraction_time_in_pause_period = mask_is_pause_period.sum() / wake_sum
    np.testing.assert_almost_equal(fraction_time_in_quiescence_period + fraction_time_in_active_wake_period + fraction_time_in_pause_period, 1.0)

    # normalize by total flicker rate, notice that there is an order of magnitude difference between the flicker rate of EAB50 and CAF42
    overall_flicker_rate = mask_flickers.sum() if normalize_by_total_flicker_rate else 1
    flicker_rate_in_quiescence = (mask_is_quiescence_period & mask_flickers).sum() / fraction_time_in_quiescence_period / overall_flicker_rate
    flicker_rate_in_active_wake = mask_high_activity.sum() / fraction_time_in_active_wake_period / overall_flicker_rate
    flicker_rate_in_pause = mask_low_activity.sum() / fraction_time_in_pause_period / overall_flicker_rate

    hist_high_activity, _ = np.histogram(df[mask_high_activity]['percent_time_short_band_high_activity'], bins=n_bins)
    hist_low_activity, _ = np.histogram(df[mask_low_activity]['percent_time_short_band_low_activity'], bins=n_bins)

    flicker_rate_high_high_per_bin = hist_high_activity / fraction_time_in_active_wake_period
    flicker_rate_high_low_per_bin = hist_low_activity / fraction_time_in_pause_period

    flicker_rate_distribution = np.concatenate((
        flicker_rate_high_high_per_bin[n_bins//2:],     # left dark blue
        flicker_rate_high_low_per_bin,                  # center light blue
        flicker_rate_high_high_per_bin[:n_bins//2]      # right dark blue
    ))

    return flicker_rate_distribution, flicker_rate_in_quiescence, flicker_rate_in_active_wake, flicker_rate_in_pause


def plot_flicker_alignment_distribution(optical_flow_flickers_df: pd.DataFrame, flicker_flavors: List[FlickerFlavor], regions: List[Regions], n_bins: int = 20):
    _, flicker_rate_in_quiescence, flicker_rate_in_active_wake, flicker_rate_in_pause = \
        compute_flicker_rate_distribution(
            optical_flow_flickers_df, DATASET_NAMES, flicker_flavors, regions, n_bins,
            filter_high_activity_edges=0.0, min_short_band_segment_size=0, normalize_by_total_flicker_rate=False
        )
    n_probes = len([p for p in PROBE_REGION_CHANNEL_LIST if p.region != Regions.BADPROBE])
    flicker_rate_in_quiescence_norm_by_probes = flicker_rate_in_quiescence / n_probes
    flicker_rate_in_active_wake_norm_by_probes = flicker_rate_in_active_wake / n_probes
    flicker_rate_in_pause_norm_by_probes = flicker_rate_in_pause / n_probes

    flicker_rate_per_dataset_name_unnormalized, flicker_rate_per_dataset_name_in_quiescence_unnormalized, flicker_rate_per_dataset_name_in_active_wake_unnormalized, flicker_rate_per_dataset_name_in_pause_unnormalized = zip(*[
        compute_flicker_rate_distribution(
            optical_flow_flickers_df, [dn], flicker_flavors, regions, n_bins,
            filter_high_activity_edges=0.0, min_short_band_segment_size=0, normalize_by_total_flicker_rate=False
        )
        for dn in DATASET_NAMES
    ])

    flicker_rate_per_dataset_name_normalized_by_probe_count = np.array([
        flicker_rate / len([p for p in PROBE_REGION_CHANNEL_LIST if p.dataset_name == dn and p.region != Regions.BADPROBE])  # normalize by probe count per animal
        for dn, flicker_rate in zip(DATASET_NAMES, flicker_rate_per_dataset_name_unnormalized)
    ])
    flicker_rate_per_dataset_name_in_quiescence_normalized_by_probe_count = np.array([
        flicker_rate / len([p for p in PROBE_REGION_CHANNEL_LIST if p.dataset_name == dn and p.region != Regions.BADPROBE])
        for dn, flicker_rate in zip(DATASET_NAMES, flicker_rate_per_dataset_name_in_quiescence_unnormalized)
    ])

    zero_y = np.mean(flicker_rate_per_dataset_name_in_quiescence_normalized_by_probe_count / n_bins, axis=0)
    zero_y_err = np.std(flicker_rate_per_dataset_name_in_quiescence_normalized_by_probe_count / n_bins, axis=0) / np.sqrt(flicker_rate_per_dataset_name_in_quiescence_normalized_by_probe_count.shape[0])

    flicker_rate_per_dn_norm_probe_zero_offset = flicker_rate_per_dataset_name_normalized_by_probe_count - zero_y

    x = np.linspace(-1.0 + 1/n_bins/2, 1.0 - 1/n_bins/2, num=n_bins * 2).tolist()
    x = [-1] + x + [1]  # makes line continuous to edge
    y = np.mean(flicker_rate_per_dn_norm_probe_zero_offset, axis=0).tolist()
    y = np.array([(y[-1] + y[0]) / 2] + y + [(y[-1] + y[0]) / 2])  # makes line continuous to edge
    y_err = (np.std(flicker_rate_per_dn_norm_probe_zero_offset, axis=0) / np.sqrt(flicker_rate_per_dn_norm_probe_zero_offset.shape[0])).tolist()
    y_err = np.array([(y_err[-1] + y_err[0]) / 2] + y_err + [(y_err[-1] + y_err[0]) / 2])  # makes line continuous to edge

    y_ceiling = np.max(np.concatenate((y, y + y_err))) + 0.1
    y_floor = np.min(np.concatenate((y, y - y_err))) - 0.1

    fig = go.Figure(data=[
        # Filled std err
        go.Scatter(
            name='SEM Upper Bound',
            x=x,
            y=y + y_err,
            mode='lines',
            marker=dict(color='rgba(100, 100, 100, 0.5)'),
            line=dict(width=0),
            showlegend=False,
            line_shape='spline',
        ),
        go.Scatter(
            name='SEM Lower Bound',
            x=x,
            y=y - y_err,
            mode='lines',
            marker=dict(color='rgba(100, 100, 100, 0.5)'),
            line=dict(width=0),
            showlegend=False,
            fillcolor='rgba(100, 100, 100, 0.5)',
            fill='tonexty',
            line_shape='spline',
        ),
        go.Scatter(
            name='Flicker Rate with SEM',
            x=x,
            y=y,
            line=dict(width=4, color='black'),
            marker=dict(size=7),
            mode='lines+markers',
            line_shape='spline',
        ),
        # Zero line std err
        go.Scatter(
            name='Zero Line SEM',
            x=[-1, 1],
            y=[zero_y_err, zero_y_err],
            mode='lines',
            line=dict(dash='dash', color='rgba(50, 50, 50, 0.5)'),
        ),
        go.Scatter(
            x=[-1, 1],
            y=[-zero_y_err, -zero_y_err],
            showlegend=False,
            mode='lines',
            line=dict(dash='dash', color='rgba(50, 50, 50, 0.5)'),
        )
    ])

    # background color to differentiate the two distributions
    shapes = [
        go.layout.Shape(
            type='rect',
            x0=-1, x1=-0.5,
            y0=y_floor, y1=y_ceiling,
            fillcolor='#04D9FF',
            line=dict(color='rgba(0,0,0,0)', width=0),
            xref='x', yref='y',
            layer='below',
        ),
        go.layout.Shape(
            type='rect',
            x0=-0.5, x1=0.5,
            y0=y_floor, y1=y_ceiling,
            fillcolor='#FEC615',
            line=dict(color='rgba(0,0,0,0)', width=0),
            xref='x', yref='y',
            layer='below',
        ),
        go.layout.Shape(
            type='rect',
            x0=0.5, x1=1,
            y0=y_floor, y1=y_ceiling,
            fillcolor='#04D9FF',
            line=dict(color='rgba(0,0,0,0)', width=0),
            xref='x', yref='y',
            layer='below',
        ),
    ]

    fig.update_layout(
        bargap=0.15,
        shapes=shapes,
        title=f'flickers/day/probe quiescence: {flicker_rate_in_quiescence_norm_by_probes:.2f}, '
              f'flickers/day/probe active_wake: {flicker_rate_in_active_wake_norm_by_probes:.2f}, '
              f'flickers/day/probe pause: {flicker_rate_in_pause_norm_by_probes:.2f}, ',
        width=1200,
        height=900,
    )
    fig.update_yaxes(
        showgrid=False,
        range=[y_floor, y_ceiling],
        zeroline=True,
        title=f'Flickers rate per day per probe deviation from quiescence mean flicker rate',
    )
    fig.update_xaxes(
        tickmode='array',
        zeroline=False,
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=['50%', '100% | 0%', '50%', '100% | 0%', '50%'],
        range=[-1, 1],
    )

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flicker-distribution/flicker-distribution-{n_bins}')


def plot_quiet_active_passive_bar(optical_flow_flickers_df):
    """ Plot Quiescence, Active-wake, Passive-wake. """
    # df = optical_flow_flickers_df  # shorthand for readability
    # rate_unit = 15 * 60 * 60  # per hour rate (15 frames/sec * 60 sec/min * 60 min/hr) - default rate is calculated per 1/15th of a sec
    #
    # mask_flickers = (df['flicker'] > 0) & (df['flicker_flavor'] == FlickerFlavor.NREM_WITHIN_WAKE.name)
    # mask_is_wake = (df['label_wnr_012'] == 0)
    # mask_per_dataset = [df['dataset_name'] == dataset_name for dataset_name in DATASET_NAMES]
    #
    # # masks for flickers and periods
    # # DEFINITIONS: quiescence = low movement over long period; active_wake = high movement not pause; pause = high movement with short break in movement
    # mask_quiescent_flickers = mask_flickers & (~df['is_in_wide_band_high_activity'])
    # mask_active_wake_flickers = mask_flickers & df['is_in_wide_band_high_activity'] & df['is_in_short_band_high_activity']
    # mask_pause_flickers = mask_flickers & df['is_in_wide_band_high_activity'] & df['is_in_short_band_low_activity']
    #
    # mask_is_quiescence_period = ~df['is_in_wide_band_high_activity'] & mask_is_wake
    # mask_is_active_wake_period = df['is_in_wide_band_high_activity'] & df['is_in_short_band_high_activity'] & mask_is_wake
    # mask_is_pause_period = df['is_in_wide_band_high_activity'] & df['is_in_short_band_low_activity'] & mask_is_wake
    #
    # # flicker_rate_in_quiescence = mask_quiescent_flickers.sum() / mask_is_quiescence_period.sum() * rate_unit
    # # flicker_rate_in_active_wake = mask_active_wake_flickers.sum() / mask_is_active_wake_period.sum() * rate_unit
    # # flicker_rate_in_pause = mask_pause_flickers.sum() / mask_is_pause_period.sum() * rate_unit
    #
    # n_probes_per_dataset = [
    #     len([p for p in PROBE_REGION_CHANNEL_LIST if p.region != Regions.BADPROBE and p.dataset_name == dataset_name])
    #     for dataset_name in DATASET_NAMES
    # ]
    #
    # flicker_rate_in_quiescence_per_dataset = [
    #     (mask_quiescent_flickers & mask_dn).sum() / (mask_is_quiescence_period & mask_dn).sum() / n_probes * rate_unit
    #     for n_probes, mask_dn in zip(n_probes_per_dataset, mask_per_dataset)
    # ]
    # flicker_rate_in_active_wake_per_dataset = [
    #     (mask_active_wake_flickers & mask_dn).sum() / (mask_is_active_wake_period & mask_dn).sum() / n_probes * rate_unit
    #     for n_probes, mask_dn in zip(n_probes_per_dataset, mask_per_dataset)
    # ]
    # flicker_rate_in_pause_per_dataset = [
    #     (mask_pause_flickers & mask_dn).sum() / (mask_is_pause_period & mask_dn).sum() / n_probes * rate_unit
    #     for n_probes, mask_dn in zip(n_probes_per_dataset, mask_per_dataset)
    # ]
    #
    # # anova - dataframe ['dataset_name', 'Q|A|P', 'flicker_rate_per_dataset']
    # df_anova = pd.DataFrame({
    #     'dataset_name': DATASET_NAMES * 3,
    #     'QAP': ['quiescence'] * 9 + ['active-wake'] * 9 + ['pause'] * 9,
    #     'flicker_rate': flicker_rate_in_quiescence_per_dataset + flicker_rate_in_active_wake_per_dataset + flicker_rate_in_pause_per_dataset,
    # })
    # df_anova.to_csv(f'{BASEPATH_AUDIT}/quiescence_active_pause.csv')
    # print(df_anova)
    #
    # anova_result_qap = anova.AnovaRM(data=df_anova, depvar='flicker_rate', subject='dataset_name', within=['QAP']).fit()
    # p_val_qap = anova_result_qap.anova_table['Pr > F']['QAP']
    # n_stars_qap = _get_n_stars(p_val_qap)
    # print(anova_result_qap)
    #
    # mask_qa = df_anova['QAP'].isin(['quiescence', 'active-wake'])
    # anova_result_qa = anova.AnovaRM(data=df_anova[mask_qa], depvar='flicker_rate', subject='dataset_name', within=['QAP']).fit()
    # p_val_qa = anova_result_qa.anova_table['Pr > F']['QAP']
    # n_stars_qa = _get_n_stars(p_val_qa)
    # print(anova_result_qa)
    #
    # mask_qp = df_anova['QAP'].isin(['quiescence', 'pause'])
    # anova_result_qp = anova.AnovaRM(data=df_anova[mask_qp], depvar='flicker_rate', subject='dataset_name', within=['QAP']).fit()
    # p_val_qp = anova_result_qp.anova_table['Pr > F']['QAP']
    # n_stars_qp = _get_n_stars(p_val_qp)
    # print(anova_result_qp)
    #
    # mask_ap = df_anova['QAP'].isin(['active-wake', 'pause'])
    # anova_result_ap = anova.AnovaRM(data=df_anova[mask_ap], depvar='flicker_rate', subject='dataset_name', within=['QAP']).fit()
    # p_val_ap = anova_result_ap.anova_table['Pr > F']['QAP']
    # n_stars_ap = _get_n_stars(p_val_ap)
    # print(anova_result_ap)
    #
    # # following https://www.statology.org/tukey-test-python/
    # one_way_anova_result = scipy.stats.f_oneway(flicker_rate_in_quiescence_per_dataset, flicker_rate_in_active_wake_per_dataset, flicker_rate_in_pause_per_dataset)
    # print(one_way_anova_result)
    # import statsmodels.stats.multicomp as multi
    # tukey_test = multi.pairwise_tukeyhsd(endog=df_anova['flicker_rate'], groups=df_anova['QAP'], alpha=0.05)
    # print(tukey_test)
    #
    # # y = [
    # #     flicker_rate_in_quiescence,
    # #     flicker_rate_in_active_wake,
    # #     flicker_rate_in_pause
    # # ]
    # y = [
    #     np.mean(flicker_rate_in_quiescence_per_dataset),
    #     np.mean(flicker_rate_in_active_wake_per_dataset),
    #     np.mean(flicker_rate_in_pause_per_dataset),
    # ]
    # error_y = [
    #     np.std(flicker_rate_in_quiescence_per_dataset, axis=0) / np.sqrt(len(DATASET_NAMES)),
    #     np.std(flicker_rate_in_active_wake_per_dataset, axis=0) / np.sqrt(len(DATASET_NAMES)),
    #     np.std(flicker_rate_in_pause_per_dataset, axis=0) / np.sqrt(len(DATASET_NAMES)),
    # ]
    # y_sig = np.max(y) + np.max(error_y) + 0.3

    flicker_flavors = [FlickerFlavor.NREM_WITHIN_WAKE]
    regions = set(Regions) - {Regions.BADPROBE}
    probes = [p for p in PROBE_REGION_CHANNEL_LIST if p.region != Regions.BADPROBE]
    n_bins = 10

    _, flicker_rate_in_quiescence, flicker_rate_in_active_wake, flicker_rate_in_pause = \
        compute_flicker_rate_distribution(optical_flow_flickers_df, DATASET_NAMES, flicker_flavors, regions, n_bins)

    # flicker_rate_per_dataset_name_unnormalized, flicker_rate_per_dataset_name_in_quiescence_unnormalized, flicker_rate_per_dataset_name_in_active_wake_unnormalized, flicker_rate_per_dataset_name_in_pause_unnormalized = zip(*[
    #     compute_flicker_rate_distribution(optical_flow_flickers_df, [dn], flicker_flavors, regions, n_bins)
    #     for dn in DATASET_NAMES
    # ])

    flicker_rate_per_probe, flicker_rate_per_probe_in_quiescence, flicker_rate_per_probe_in_active_wake, flicker_rate_per_probe_in_pause = zip(*[
        compute_flicker_rate_distribution(
            optical_flow_flickers_df, [probe.dataset_name], flicker_flavors, [probe.region], n_bins,
            filter_high_activity_edges=0.1, min_short_band_segment_size=2 * 15, normalize_by_total_flicker_rate=True,
        )
        for probe in probes
    ])

    # flicker_rate_per_dataset_name_in_quiescence_normalized_by_probe_count = np.array([
    #     flicker_rate / len([p for p in PROBE_REGION_CHANNEL_LIST if p.dataset_name == dn and p.region != Regions.BADPROBE])
    #     for dn, flicker_rate in zip(DATASET_NAMES, flicker_rate_per_dataset_name_in_quiescence_unnormalized)
    # ])
    # flicker_rate_per_dataset_name_in_active_wake_normalized_by_probe_count = np.array([
    #     flicker_rate / len([p for p in PROBE_REGION_CHANNEL_LIST if p.dataset_name == dn and p.region != Regions.BADPROBE])
    #     for dn, flicker_rate in zip(DATASET_NAMES, flicker_rate_per_dataset_name_in_active_wake_unnormalized)
    # ])
    # flicker_rate_per_dataset_name_in_pause_normalized_by_probe_count = np.array([
    #     flicker_rate / len([p for p in PROBE_REGION_CHANNEL_LIST if p.dataset_name == dn and p.region != Regions.BADPROBE])
    #     for dn, flicker_rate in zip(DATASET_NAMES, flicker_rate_per_dataset_name_in_pause_unnormalized)
    # ])

    # Create dataset for Lmer model
    df_anova = pd.DataFrame({
        'flicker_rate': flicker_rate_per_probe_in_quiescence +
                        flicker_rate_per_probe_in_active_wake +
                        flicker_rate_per_probe_in_pause,
        'QAP': ['quiescence'] * len(flicker_rate_per_probe_in_quiescence) +
               ['active'] * len(flicker_rate_per_probe_in_active_wake) +
               ['pause'] * len(flicker_rate_per_probe_in_pause),
        'dataset_name': [probe.dataset_name for probe in probes] * 3,
        'region': [probe.region.value for probe in probes] * 3,
        'probe': [f'{probe.dataset_name}_{probe.region}' for probe in probes] * 3,
    })
    # Filter rows with rate of 0 (occurs many times)
    df_anova = df_anova[df_anova['flicker_rate'] > 0.0]
    # Apply log normal
    df_anova_log_norm = df_anova.copy()
    df_anova_log_norm['flicker_rate'] = np.log(df_anova['flicker_rate'])

    print('\n', df_anova_log_norm)
    print('\n', df_anova_log_norm.groupby(by='QAP').mean())

    # Perform post hoc ANOVA
    model = Lmer('flicker_rate ~ QAP + (1|dataset_name) + (1|region)', data=df_anova_log_norm)
    model.fit(
        factors={
            'QAP': df_anova_log_norm['QAP'].unique().tolist(),
            'dataset_name': df_anova_log_norm['dataset_name'].unique().tolist(),
            'region': df_anova_log_norm['region'].unique().tolist(),
        },
        order=True,
        summarize=True,
    )
    anova_value = model.anova(force_orthogonal=True)
    print('Anova_value> \n', anova_value)

    marginal_estimates_state, comparisons_state = model.post_hoc(marginal_vars='QAP', p_adjust='bonf')
    print('Marginal estimates by QAP: \n', marginal_estimates_state)
    print('Comparisons state> \n', comparisons_state)

    jitter = 0.5
    pointpos = 0
    marker_size = 7
    boxpoints = False

    fig = go.Figure()

    fig.add_trace(go.Box(
        name='Pause',
        x=[1],
        y=df_anova[df_anova['QAP'] == 'pause']['flicker_rate'],
        marker=dict(color='cornflowerblue', size=marker_size),
        # jitter=jitter,
        # pointpos=pointpos,
        boxpoints=boxpoints,
    ))
    fig.add_trace(go.Box(
        name='Quiescence',
        x=[2],
        y=df_anova[df_anova['QAP'] == 'quiescence']['flicker_rate'],
        marker=dict(color='indianred', size=marker_size),
        line=dict(width=1.5),
        # jitter=jitter,
        # pointpos=pointpos,
        boxpoints=boxpoints,
    ))
    fig.add_trace(go.Box(
        name='Active Wake',
        x=[3],
        y=df_anova[df_anova['QAP'] == 'active']['flicker_rate'],
        marker=dict(color='forestgreen', size=marker_size),
        # jitter=jitter,
        # pointpos=pointpos,
        boxpoints=boxpoints,
    ))

    # Add scatter plot with colors per region
    for region in set(Regions) - {Regions.BADPROBE}:
        mask = df_anova['region'] == region.value
        # x = df_anova[mask]['QAP'].map({'pause': 'Pause', 'quiescence': 'Quiescence', 'active': 'Active Wake'}).to_list()
        x = df_anova[mask]['QAP'].map({'pause': 1, 'quiescence': 2, 'active': 3}).to_list()
        y = df_anova[mask]['flicker_rate'].to_list()
        assert len(x) == len(y)
        fig.add_trace(go.Scatter(
            name=f'{region.value}',
            x=np.array(x) + (np.random.rand(len(x))/2 - 0.25),
            y=y,
            marker=dict(color=COLOR_MAP[region]),
            mode='markers',
        ))

    fig.update_layout(
        # title=f'p-val 3 / 2: {p_val_qap:.5f} / {p_val_ap:.5f}',
        width=450,
        height=700,
        xaxis=go.layout.XAxis(
            title='Quiescence (low activity wake)<br>'
                  'Active-wake (high activity wake)<br>'
                  'Pause (brief break in high activity region)<br>',
        ),
        yaxis=go.layout.YAxis(
            title='Flicker rate per probe normalized to overall flicker rate, log scale',
            type='log',
        ),
        bargap=0.20,
        margin=dict(b=0.5),
        # shapes=[
        #     # top bar across all three distributions
        #     go.layout.Shape(
        #         type='line',
        #         x0=-0.3, x1=2.3,
        #         y0=y_sig + 0.7, y1=y_sig + 0.7,
        #         line=go.layout.shape.Line(color='black', width=1.5),
        #         xref='x', yref='y'
        #     ),
        #     # second bar across last two distributions
        #     go.layout.Shape(
        #         type='line',
        #         x0=0.7, x1=2.3,
        #         y0=y_sig, y1=y_sig,
        #         line=go.layout.shape.Line(color='black', width=1.5),
        #         xref='x', yref='y'
        #     )
        # ],
        # annotations=[
        #     # top bar across all three distributions
        #     go.layout.Annotation(
        #         x=1,
        #         y=y_sig + 0.8,
        #         text='*' * n_stars_qap if n_stars_qap > 0 else 'ns',
        #         showarrow=False,
        #         font=go.layout.annotation.Font(size=30),
        #     ),
        #     go.layout.Annotation(
        #         x=1 + 0.45,
        #         y=y_sig + 0.8 + 0.22,
        #         text=f'p={p_val_qap:.4f}',
        #         showarrow=False,
        #         font=go.layout.annotation.Font(size=12),
        #     ),
        #     # second bar across last two distributions
        #     go.layout.Annotation(
        #         x=1.5,
        #         y=y_sig + 0.1,
        #         text='*' * n_stars_ap if n_stars_ap > 0 else 'ns',
        #         showarrow=False,
        #         font=go.layout.annotation.Font(size=30),
        #     ),
        #     go.layout.Annotation(
        #         x=1.5 + 0.45,
        #         y=y_sig + 0.1 + 0.22,
        #         text=f'p={p_val_ap:.4f}',
        #         showarrow=False,
        #         font=go.layout.annotation.Font(size=12),
        #     ),
        # ],
    )
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flicker-quiet-active-passive/flicker-quiet-active-passive-bar')
    df_anova.to_csv(f'{BASEPATH_AUDIT}/qap_plot_anova.csv')


def plot_human_labels_vs_cnn(dataset_name: str, versions: List[str], offset: int, length: int, testset_only: bool = True):
    predictions_df = pd.DataFrame(_pull_predictions(dataset_name, versions[0], testset_only))
    df = predictions_df[offset:offset+length]

    def find_slices(a):
        result = scipy.ndimage.find_objects(scipy.ndimage.label(a)[0])
        result = [r[0] for r in result]  # remove inter tuple to only support 1D
        return result

    fig = make_subplots(rows=1 + len(versions), cols=1, vertical_spacing=0.005)

    # HUMAN
    for label in [0, 1, 2]:
        slices = find_slices(df['label_wnr_012'] == label)

        color = COLOR_MAP[WNR_MAP[label]]
        # color = 'yellow' if label == 0 else 'blue' if label == 1 else 'red' if label == 2 else 'black'

        for s in slices:
            fig.add_shape(
                type='rect',
                x0=s.start, y0=0,
                x1=s.stop, y1=1,
                line=dict(width=0),
                fillcolor=color,
                row=1, col=1,
            )
        fig.update_xaxes(tickvals=[], row=1, col=1)

    for i, version in enumerate(versions):
        predictions_df = pd.DataFrame(_pull_predictions(dataset_name, version, testset_only))
        df = predictions_df[offset:offset+length]

        # CNN
        fig.add_trace(go.Scatter(
            x=np.arange(df.shape[0]),
            y=df['probability_rem'],
            mode='lines',
            line=dict(width=0.1, color=COLOR_MAP['rem']),
            fillcolor=COLOR_MAP['rem'],
            stackgroup='one',
            name='rem',
            showlegend=i == 0,
        ), row=2 + i, col=1)

        fig.add_trace(go.Scatter(
            x=np.arange(df.shape[0]),
            y=df['probability_nrem'],
            mode='lines',
            line=dict(width=0.1, color=COLOR_MAP['nrem']),
            fillcolor=COLOR_MAP['nrem'],
            stackgroup='one',
            name='nrem',
            showlegend=i == 0,
        ), row=2 + i, col=1)

        fig.add_trace(go.Scatter(
            x=np.arange(df.shape[0]),
            y=df['probability_wake'],
            mode='lines',
            line=dict(width=0.1, color=COLOR_MAP['wake']),
            fillcolor=COLOR_MAP['wake'],
            stackgroup='one',
            name='wake',
            showlegend=i == 0,
        ), row=2 + i, col=1)

        fig.update_xaxes(tickvals=[], row=2 + i, col=1)
        fig.update_yaxes(title=_get_region(dataset_name, _get_channel(version, RegexDef.PERREGION)).value, row=2 + i, col=1)

    last_row = 1 + len(versions)
    fig.update_xaxes(range=[0, 54000])
    fig.update_xaxes(
        tickvals=[0, 9000, 18000, 27000, 36000, 45000, 53999],
        ticktext=['0 min', '10 min', '20 min', '30 min', '40 min', '50 min', '60 min'],
        row=last_row, col=1,
    )
    for i in range(last_row):
        fig.update_xaxes(range=[0, len(df)], row=i + 1)

    fig.update_yaxes(range=[0, 1], tickvals=[])
    fig.update_yaxes(title='Human', row=1, col=1)

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-human-cnn-labels/human-cnn-labels-{dataset_name}-{offset}-{length}', save_formats=('svg', 'pdf', 'png', 'html'))


def compute_raw_flicker_rate(optical_flow_flickers_df: pd.DataFrame, dataset_name: str, region: Regions) -> Tuple[float]:
    """ Compute flicker rates: 1) overall rate, 2) wake rate, 3) nrem rate, 4) rem rate. """
    per_unit = 60 * 60 * 15  # per hour
    df = optical_flow_flickers_df[optical_flow_flickers_df.dataset_name == dataset_name]
    mask_flickers_by_probe = optical_flow_flickers_df.flicker_region == region.value
    result = []

    filters_by_flavor_label = [
        (['_WITHIN_'.join(f.value) for f in FlickerFlavor], [0, 1, 2]),
        (['_WITHIN_'.join(f.value) for f in FlickerFlavor if f.value[1] == 'WAKE'], [0]),
        (['_WITHIN_'.join(f.value) for f in FlickerFlavor if f.value[1] == 'NREM' or f.value[1] == 'REM'], [1, 2]),
        (['_WITHIN_'.join(f.value) for f in FlickerFlavor if f.value[1] == 'NREM'], [1]),
        (['_WITHIN_'.join(f.value) for f in FlickerFlavor if f.value[1] == 'REM'], [2]),
    ]

    for filter_by_flavor, filter_by_label in filters_by_flavor_label:
        mask_flicker_flavor = df['flicker_flavor'].isin(filter_by_flavor)
        mask_label = df['label_wnr_012'].isin(filter_by_label)
        flicker_count = (df[mask_flicker_flavor & mask_flickers_by_probe]['flicker'] > 0).sum()
        unit_time = mask_label.sum() / per_unit
        result.append(flicker_count / unit_time)

    return result


def plot_flicker_rates(optical_flow_flickers_df: pd.DataFrame):
    # Compute flicker rate per animal, flicker rates: overall|wake|nrem|rem
    cache_key = ('__plot_flicker_rates__', pd.util.hash_pandas_object(optical_flow_flickers_df).sum(), PROBE_REGION_CHANNEL_LIST)
    per_probe_flicker_rates = cache.get(cache_key)
    if per_probe_flicker_rates is None:
        per_probe_flicker_rates = pd.DataFrame([
            compute_raw_flicker_rate(optical_flow_flickers_df, probe.dataset_name, probe.region)
            for probe in PROBE_REGION_CHANNEL_LIST if probe.region != Regions.BADPROBE
        ], columns=['Overall', 'Wake', 'Sleep', 'NREM', 'REM'])
        cache[cache_key] = per_probe_flicker_rates

    df = pd.DataFrame({
        'flicker_rate': pd.concat((
            per_probe_flicker_rates['Overall'],
            per_probe_flicker_rates['Wake'],
            per_probe_flicker_rates['Sleep'],
            per_probe_flicker_rates['NREM'],
            per_probe_flicker_rates['REM'],
        )),
        'group': ['Overall'] * len(per_probe_flicker_rates['Overall']) +
                ['Wake'] * len(per_probe_flicker_rates['Wake']) +
                ['Sleep'] * len(per_probe_flicker_rates['Sleep']) +
                ['NREM'] * len(per_probe_flicker_rates['NREM']) +
                ['REM'] * len(per_probe_flicker_rates['REM']),
        'dataset_name': [probe.dataset_name for probe in PROBE_REGION_CHANNEL_LIST if probe.region != Regions.BADPROBE] * 5,
        'region': [probe.region.value for probe in PROBE_REGION_CHANNEL_LIST if probe.region != Regions.BADPROBE] * 5,
    })
    df = df[df['flicker_rate'] > 0.0]
    df_anova = df[df.group.isin(['Wake', 'NREM', 'REM'])]
    df_anova['flicker_rate'] = np.log(df_anova['flicker_rate'].to_numpy())

    print(df_anova)

    # Perform 2 tail t test
    mask_group_sleep = df_anova['group'].isin(['NREM', 'REM'])
    mask_group_wake = df_anova['group'] == 'Wake'
    ttest_result = scipy.stats.ttest_ind(df_anova[mask_group_wake]['flicker_rate'], df_anova[mask_group_sleep]['flicker_rate'])
    print('ttest_result\n', ttest_result)

    # Perform post hoc ANOVA
    model = Lmer('flicker_rate ~ group + (1|dataset_name)', data=df_anova)
    model.fit(
        factors={
            'group': ['Wake', 'NREM', 'REM'],
            'dataset_name': df_anova['dataset_name'].unique().tolist(),
            # 'region': df_anova['region'].unique().tolist(),
        },
        order=True,
        summarize=True,
    )
    anova_value = model.anova(force_orthogonal=True)
    print('Anova_value> \n', anova_value)

    marginal_estimates_state, comparisons_state = model.post_hoc(marginal_vars='group', p_adjust='bonf')
    print('Marginal estimates by group: \n', marginal_estimates_state)
    print('Comparisons state> \n', comparisons_state)

    fig = go.Figure()
    clrs = ['cornflowerblue', COLOR_MAP['wake'], 'coral', COLOR_MAP['nrem'], COLOR_MAP['rem']]
    for group, clr in zip(df.group.unique().tolist(), clrs):
        mask = df['group'] == group
        fig.add_trace(go.Box(
            y=df[mask]['flicker_rate'],
            name=group,
            jitter=0.3,
            boxpoints='all',
            pointpos=0,
            marker_color=clr,
        ))
    fig.update_layout(
        width=400,
        height=600,
        boxgap=0.1,
    )
    fig.update_yaxes(
        title='Per hour per probe flicker rate',
        type='log',
    )
    fig.update_xaxes(
        title='Overall|wake|NREM|REM rate per animal',
    )

    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flicker-rate/flicker-rate', save_formats=('svg', 'pdf', 'png', 'html'))


def plot_human_cnn_lfp(optical_flow_region_df: pd.DataFrame):

    dataset_name = optical_flow_region_df.dataset_name.unique()
    assert len(dataset_name) == 1
    dataset_name = dataset_name[0]

    dataset_regions = [probe.region.value for probe in PROBE_REGION_CHANNEL_LIST if probe.region != Regions.BADPROBE and probe.dataset_name == dataset_name]

    # get flicker closest to the center, that's the target flicker. In some cases there are multiple flickers in the window.
    flicker = optical_flow_region_df.iloc[:(len(optical_flow_region_df)+3)//2][optical_flow_region_df.flicker > 0].iloc[-1]

    import collections
    probes_dataset = [probe for probe in PROBE_REGION_CHANNEL_LIST if probe.dataset_name == dataset_name]
    predictions_per_region = collections.OrderedDict([
        (
            probe.region.value,
            _pull_predictions(
                dataset_name,
                f'wnr-v14-perregion-c64k-{probe.channel_from}-{probe.channel_to}-run1',
                testset_only=False,
            )[optical_flow_region_df.iloc[0].ix:optical_flow_region_df.iloc[-1].ix + 1]
        )
        for probe in probes_dataset
        ])

    neural_filename = optical_flow_region_df.iloc[0]['neural_filename']
    confidence_per_fps = predictions_per_region[flicker.flicker_region]['confidence_wnr_01']

    human_wake = (optical_flow_region_df['label_wnr_012'] == 0).astype(np.int32)
    human_nrem = (optical_flow_region_df['label_wnr_012'] == 1).astype(np.int32)
    human_rem = (optical_flow_region_df['label_wnr_012'] == 2).astype(np.int32)

    # get raw data sample
    fname = f'{BASEPATH_DATA}/{dataset_name}/Neural_Data/{neural_filename}'
    n_channels = 64 * len([probe for probe in PROBE_REGION_CHANNEL_LIST if probe.dataset_name == dataset_name])
    offset = optical_flow_region_df.iloc[0].neural_offset
    data_len = optical_flow_region_df.iloc[-1].neural_offset - offset
    neural_data_raw = _get_raw_data_sample(fname, n_channels, offset, data_len)

    x_raw = np.linspace(0, len(confidence_per_fps) / 15, num=neural_data_raw.shape[1])
    x_fps = np.linspace(0, len(human_wake) / 15, num=len(optical_flow_region_df))

    n_subplots = 2 + 2 * len(dataset_regions)
    row = 1
    fig = make_subplots(rows=n_subplots, cols=1, vertical_spacing=0)

    # Confidence, top row
    fig.add_trace(go.Scatter(
        x=x_fps,
        y=confidence_per_fps,
        name='Confidence',
        fill='tozeroy',
        line=dict(width=0, color='goldenrod')
    ), row=row, col=1)
    row += 1

    # Human labels, second row
    fig.add_trace(go.Scatter(
        x=x_fps, y=human_wake,
        stackgroup='human', fillcolor='#D5E59E', name='Human WAKE', showlegend=True,
        line=dict(width=0, color='#D5E59E')
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=x_fps, y=human_nrem,
        stackgroup='human', fillcolor='#08A24A', name='Human NREM', showlegend=True,
        line=dict(width=0, color='#08A24A')
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=x_fps, y=human_rem,
        stackgroup='human', fillcolor='#265C2E', name='Human REM', showlegend=True,
        line=dict(width=0, color='#265C2E')
    ), row=row, col=1)
    fig['layout'][f'xaxis{row}'].update(showticklabels=False)
    fig['layout'][f'yaxis{row}'].update(title='Human')
    row += 1

    # LFP, per region
    for brain_region in dataset_regions:
        probe = [probe for probe in probes_dataset if probe.region.value == brain_region][0]
        plot_channel = [c for c in SPIKING_CHANNELS[dataset_name] if probe.channel_from <= c < probe.channel_to][0]

        fig.add_trace(go.Scatter(
            x=x_raw,
            y=neural_data_raw[plot_channel, :],
            name=f'LFP {brain_region}',
            line=dict(color='rgb(8, 162, 74)')
        ), row=row, col=1)
        fig['layout'][f'yaxis{row}'].update(title=f'{brain_region}')
        fig['layout'][f'xaxis{row}'].update(showticklabels=False)
        row += 1

    # Per brain region predictions
    for brain_region, predictions in zip(dataset_regions, predictions_per_region.values()):
        # region_optical_flow_region_df = all_predictions[brain_region][csv_row:csv_row + plot_len_fps]
        fig.add_trace(go.Scatter(x=x_fps, y=predictions['probability_wake'], stackgroup='p',
                                 fillcolor='#D5E59E', name='Probability WAKE', showlegend=False,
                                 line=dict(width=0, color='#D5E59E')), row=row, col=1)
        fig.add_trace(go.Scatter(x=x_fps, y=predictions['probability_nrem'], stackgroup='p',
                                 fillcolor='#08A24A', name='Probability NREM', showlegend=False,
                                 line=dict(width=0, color='#08A24A')), row=row, col=1)
        fig.add_trace(go.Scatter(x=x_fps, y=predictions['probability_rem'], stackgroup='p',
                                 fillcolor='#265C2E', name='Probability REM', showlegend=False,
                                 line=dict(width=0, color='#265C2E')), row=row, col=1)
        fig['layout'][f'yaxis{row}'].update(title=f'{brain_region}')
        fig['layout'][f'xaxis{row}'].update(showticklabels=False)
        row += 1

    fig.update_layout(
        height=800, width=900,
        title=f'{flicker.dataset_name} | Flicker region: {flicker.flicker_region} | Flicker ix: {flicker.ix}'
    )
    fig['layout']['yaxis'].update(
        title=f'Confidence<br>{flicker.flicker_region}',
        range=[0, 1],
    )
    fig['layout']['xaxis'].update(visible=False)
    tickvals = np.linspace(0, optical_flow_region_df.shape[0] / 15, 5)
    ticktext = [f'{v:.1f}s' for v in tickvals]
    fig['layout'][f'xaxis{n_subplots}'].update(tickvals=tickvals, ticktext=ticktext)

    # fig.show()
    # fig.write_image(f'../figures/manual/{outname}.eps')
    # fig.write_image(f'../figures/manual/{outname}.svg')
    # fig.write_image(f'../figures/manual/{outname}.png')
    # fig.write_image(f'../figures/manual/{outname}.pdf')

    _save_figure(
        fig,
        f'{BASEPATH_FIGURES}/figure-human-readable-flickers/human-readable-flickers-{dataset_name}-{flicker.ix}',
        save_formats=('svg', 'pdf', 'html')
    )


def plot_human_cnn_lfp_all_flickers(optical_flow_flickers_df: pd.DataFrame):
    df = optical_flow_flickers_df
    seconds = 15
    plot_len = 10 * seconds

    job_list = []
    print('')

    with multiprocessing.Pool(8) as pool:
        for index in df[df.flicker > 0].index.to_numpy():
            if len(job_list) > 8 * 2:
                print(f'   ... awaiting {job_list[0][0]}')
                job_list.pop(0)[1].get()
            flicker_center = index + df.loc[index].flicker // 2
            df_flicker = df[flicker_center - plot_len:flicker_center + plot_len]
            # plot_human_cnn_lfp(df_flicker)
            job_name = f'{df.loc[index].dataset_name} - {flicker_center}'
            print(f'   ... submitting: {job_name}')
            job_list.append((job_name, pool.apply_async(plot_human_cnn_lfp, args=(df_flicker,))))

    # wait for jobs
    for j in job_list:
        print(f'   ... draining: {j[0]}')
        j[1].get()


def plot_flickers_by_proportion_wake(optical_flow_flickers_df: pd.DataFrame, bins=5):
    """ Plot flicker rate as a function of sleep pressure (% time asleep in past hour) """
    df = optical_flow_flickers_df  # shorthand
    bin_ranges = list(zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]))  # [(.0, .2), (.2, .4), ...]

    mask_flicker_start = df['flicker'] > 0
    mask_flicker_flavors = [df['flicker_flavor'] == f.name for f in FlickerFlavor if not f.name.endswith('ANOMALY')]
    mask_bins = [((a if a > 0 else -np.inf) < df['proportion_wake_3600s_window']) & (df['proportion_wake_3600s_window'] <= b) for a, b in bin_ranges]
    mask_dataset_names = [df.dataset_name == dn for dn in DATASET_NAMES]
    mask_is_wake = df['label_wnr_012'] == 0
    mask_is_sleep = (df['label_wnr_012'] == 1) | (df['label_wnr_012'] == 2)

    # todo change from per-animal to per-probe
    # todo change sleep pressure calculation to 30 min
    flicker_count_per_flavor_per_bin = pd.DataFrame(data=[
        (
            dataset_name,
            f.name,
            f'{a:.1f}-{b:.1f}',
            (mask_flicker_start & mask_dataset_name & mask_flicker_flavor & mask_bin).sum(),
        )
        for mask_bin, (a, b) in zip(mask_bins, bin_ranges)
        for dataset_name, mask_dataset_name in zip(DATASET_NAMES, mask_dataset_names)
        for f, mask_flicker_flavor in zip(FlickerFlavor, mask_flicker_flavors)
    ], columns=[
        'dataset_name',
        'flicker_flavor',
        'bin_range',
        'flicker_count',
    ])

    seconds_per_bin_wake = pd.Series([(mask_bin & mask_is_wake).sum() / 15 for mask_bin in mask_bins])
    seconds_per_bin_sleep = pd.Series([(mask_bin & mask_is_sleep).sum() / 15 for mask_bin in mask_bins])

    # print('\n', flicker_count_per_flavor_per_bin)
    # print('\nseconds_per_bin_wake\n', seconds_per_bin_wake)
    # print('\nseconds_per_bin_sleep\n', seconds_per_bin_sleep)

    unit = 60 * 60  # adjusts seconds to desired unit, this adjusts seconds
    x = [f'{int(a*100):d}% - {int(b*100):d}%' for a, b in bin_ranges]
    mask_wake = flicker_count_per_flavor_per_bin.flicker_flavor.str.endswith('WAKE')
    y_wake = flicker_count_per_flavor_per_bin[mask_wake].groupby(by='bin_range').flicker_count.sum().to_numpy() / (seconds_per_bin_wake.to_numpy() / unit)
    y_sleep = flicker_count_per_flavor_per_bin[~mask_wake].groupby(by='bin_range').flicker_count.sum().to_numpy() / (seconds_per_bin_sleep.to_numpy() / unit)

    # plot
    fig = go.Figure(data=[
        go.Bar(
            name='WAKE',
            x=x,
            y=y_wake,
            marker_color='rgb(200, 200, 200)',
        ),
        go.Bar(
            name='SLEEP',
            x=x,
            y=y_sleep,
            marker_color='rgb(55, 55, 55)',
        ),
    ])
    fig.update_layout(
        barmode='group',
    )
    fig.update_xaxes(
        title='← Less sleep pressure  |  Percent of last hour in wake  |  More sleep pressure →',
    )
    fig.update_yaxes(
        title='Flicker rate per hour per bin',
    )

    # todo save dataframe
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-sleep-pressure/flickers-sleep-pressure', save_formats=('svg', 'pdf', 'html'))


@cache.memoize()
def compute_flicker_segments(ixs_flickers, mask):
    flicker_segments = []  # [(segment_start_ix, segment_end_ix), (...), ...] uses python standard [inclusive, exclusive)
    for ix in ixs_flickers:
        segment_start_ix = np.where(~mask[:ix + 1][::-1])[0]
        segment_start_ix = ix - (segment_start_ix[0] if len(segment_start_ix) > 0 else 0) + 1
        segment_end_ix = np.where(~mask[ix:])[0]
        segment_end_ix = ix + (segment_end_ix[0] if len(segment_end_ix > 0) else len(mask[ix:]))
        if segment_end_ix - segment_start_ix > 0:
            flicker_segments.append((ix, segment_start_ix, segment_end_ix))
    return flicker_segments


def plot_flickers_by_percent_time_in_wake_sleep(optical_flow_flickers_df: pd.DataFrame):
    # get distribution of lengths of all waking periods
    s = optical_flow_flickers_df.label_wnr_012 == 0
    v = s.groupby(s.ne(s.shift()).cumsum()).sum()
    wake_lengths = v[v > 0].tolist()

    s = optical_flow_flickers_df.label_wnr_012.isin([1, 2])
    v = s.groupby(s.ne(s.shift()).cumsum()).sum()
    sleep_lengths = v[v > 0].tolist()

    # Plot/save histogram of segment distribution
    fig = px.histogram((np.array(wake_lengths) / 15 / 60).tolist(), title='Wake segments distribution')
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-percent-time/wake-distribution')
    fig = px.histogram((np.array(sleep_lengths) / 15 / 60).tolist(), title='Sleep segments distribution')
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-percent-time/sleep-distribution')

    # WAKE
    # mark only segments that have a flicker in them
    ixs_wake_flickers = np.where((optical_flow_flickers_df.flicker > 0).to_numpy())[0].tolist()
    mask_is_wake = (optical_flow_flickers_df.label_wnr_012 == 0).to_numpy()
    mask_flicker_wake_segments = np.zeros(len(optical_flow_flickers_df), dtype=bool)  # start as 000s, fill in with while loops below

    # update boolean array with 111s at each segment with a flicker in it
    wake_flicker_segments = compute_flicker_segments(ixs_wake_flickers, mask_is_wake)

    # Get distribution of lengths of flicker segments
    s = pd.Series(mask_flicker_wake_segments)
    v = s.groupby(s.ne(s.shift()).cumsum()).sum()
    wake_flicker_segment_lengths = v[v > 0].tolist()

    # SLEEP
    # mark only segments that have a flicker in them
    ixs_sleep_flickers = np.where((optical_flow_flickers_df.flicker > 0).to_numpy())[0].tolist()
    mask_is_sleep = (optical_flow_flickers_df.label_wnr_012.isin([1, 2])).to_numpy()
    mask_flicker_sleep_segments = np.zeros(len(optical_flow_flickers_df), dtype=bool)  # start as 000s, fill in with while loops below

    # update boolean array with 111s at each segment with a flicker in it
    sleep_flicker_segments = compute_flicker_segments(ixs_sleep_flickers, mask_is_sleep)

    # Get distribution of lengths of flicker segments
    s = pd.Series(mask_flicker_sleep_segments)
    v = s.groupby(s.ne(s.shift()).cumsum()).sum()
    sleep_flicker_segment_lengths = v[v > 0].tolist()

    fig = px.histogram(wake_flicker_segment_lengths, title='Flicker-wake segments distribution')
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-percent-time/flicker-wake-distribution')
    fig = px.histogram(sleep_flicker_segment_lengths, title='Flicker-sleep segments distribution')
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-percent-time/flicker-sleep-distribution')

    # NREM
    # mark only segments that have a flicker in them
    # ixs_nrem_flickers = np.where(((optical_flow_flickers_df.flicker > 0) & (optical_flow_flickers_df.flicker_flavor == 'WAKE_WITHIN_NREM')).to_numpy())[0].tolist()
    ixs_nrem_flickers = np.where((optical_flow_flickers_df.flicker > 0).to_numpy())[0].tolist()
    mask_is_nrem = (optical_flow_flickers_df.label_wnr_012.isin([1])).to_numpy()  # filters to show only nrem-in-wake flickers
    mask_flicker_nrem_segments = np.zeros(len(optical_flow_flickers_df), dtype=bool)

    # update boolean array with 111s at each segment with a flicker in it
    nrem_flicker_segments = compute_flicker_segments(ixs_nrem_flickers, mask_is_nrem)

    # Get distribution of lengths of flicker segments
    s = pd.Series(mask_flicker_nrem_segments)
    v = s.groupby(s.ne(s.shift()).cumsum()).sum()
    nrem_flicker_segment_lengths = v[v > 0].tolist()

    fig = px.histogram(wake_flicker_segment_lengths, title='Flicker-wake segments distribution')
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-percent-time/flicker-wake-distribution')
    fig = px.histogram(nrem_flicker_segment_lengths, title='Flicker-nrem segments distribution')
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-percent-time/flicker-nrem-distribution')

    #
    # PLOTTING
    #
    
    segment_length_from_min = 1
    segment_length_to_min = 120
    n_bins = 20
    
    #
    # WAKE
    #
    # Plot location of flicker on number line for all flickers in segments > X seconds
    df = pd.DataFrame(data=wake_flicker_segments, columns=['wake_ix', 'wake_segment_start_ix', 'wake_segment_end_ix'])
    df['segment_length'] = df['wake_segment_end_ix'] - df['wake_segment_start_ix']
    df['segment_length_sec'] = df['segment_length'].astype(float) / 15
    df['flicker_offset_from_segment_start'] = df['wake_ix'] - df['wake_segment_start_ix']
    df['flicker_offset_from_segment_start_min'] = df['flicker_offset_from_segment_start'].astype(float) / 15 / 60
    df['flicker_offset_from_segment_end'] = df['wake_segment_end_ix'] - df['wake_ix']
    df['flicker_offset_from_segment_end_min'] = df['flicker_offset_from_segment_end'].astype(float) / 15 / 60
    df['flicker_percent_time_in_segment'] = df['flicker_offset_from_segment_start'] / (df['wake_segment_end_ix'] - df['wake_segment_start_ix'])

    minutes = 15 * 60
    df_filtered = df[(df.segment_length >= segment_length_from_min * minutes) & (df.segment_length <= segment_length_to_min * minutes)]
    hist, bins = np.histogram(df_filtered['flicker_percent_time_in_segment'], bins=np.linspace(0, 1, n_bins + 1), density=True)

    fig = px.strip(df_filtered, x='flicker_offset_from_segment_start_min', title='Wake flickers')
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-sleep-pressure/raw-flicker-location-within-wake-segment')
    fig = go.Figure(data=[
        go.Bar(
            x=(bins + 1/n_bins/2)[:-1],
            y=hist,
            marker_color='rgb(56, 56, 56)',
        ),
        go.Scatter(
            x=df_filtered['flicker_percent_time_in_segment'],
            y=0.2 + np.random.rand(len(df_filtered)) * 0.4,
            mode='markers',
            marker=dict(size=12, color='rgb(204,204,152)'),
        ),
    ])
    fig.update_layout(title=f'Wake flickers - percent time in segment - {segment_length_from_min} min to {segment_length_to_min}')
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-sleep-pressure/raw-flicker-percent-time-wake-segment')
    fig = go.Figure(data=[
        go.Scatter(
            x=df_filtered['flicker_offset_from_segment_start_min'],
            y=np.random.rand(len(df_filtered)),
            mode='markers',
            marker=dict(size=12, color='rgba(56,56,56,0.5)'),
        ),
    ])
    fig.update_layout(title=f'Wake flickers - absolute time in segment, start segment aligned - {segment_length_from_min} min to {segment_length_to_min}', width=1000, height=400)
    fig.update_xaxes(title=f'Flicker absolute location in minutes until sleep (t=0)')
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-sleep-pressure/raw-wake-flicker-unscaled-start-segment-aligned')
    fig = go.Figure(data=[
        go.Scatter(
            x=df_filtered['flicker_offset_from_segment_end_min'] * -1,
            y=np.random.rand(len(df_filtered)),
            mode='markers',
            marker=dict(size=12, color='rgba(56,56,56,0.5)'),
        ),
    ])
    fig.update_layout(title=f'Wake flickers - absolute time in segment, end segment aligned - {segment_length_from_min} min to {segment_length_to_min}', width=1000, height=400)
    fig.update_xaxes(title=f'Flicker absolute location in minutes until sleep (t=0)')
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-sleep-pressure/raw-wake-flicker-unscaled-end-segment-aligned')

    #
    # SLEEP
    #
    # Plot location of flicker on number line for all flickers in segments > X seconds
    df = pd.DataFrame(data=sleep_flicker_segments, columns=['sleep_ix', 'sleep_segment_start_ix', 'sleep_segment_end_ix'])
    df['segment_length'] = df['sleep_segment_end_ix'] - df['sleep_segment_start_ix']
    df['segment_length_sec'] = df['segment_length'].astype(float) / 15
    df['flicker_offset_from_segment_start'] = df['sleep_ix'] - df['sleep_segment_start_ix']
    df['flicker_offset_from_segment_start_min'] = df['flicker_offset_from_segment_start'].astype(float) / 15 / 60
    df['flicker_offset_from_segment_end'] = df['sleep_segment_end_ix'] - df['sleep_ix']
    df['flicker_offset_from_segment_end_min'] = df['flicker_offset_from_segment_end'].astype(float) / 15 / 60
    df['flicker_percent_time_in_segment'] = df['flicker_offset_from_segment_start'] / (df['sleep_segment_end_ix'] - df['sleep_segment_start_ix'])

    minutes = 15 * 60
    df_filtered = df[(df.segment_length >= segment_length_from_min * minutes) & (df.segment_length <= segment_length_to_min * minutes)]
    hist, bins = np.histogram(df_filtered['flicker_percent_time_in_segment'], bins=np.linspace(0, 1, n_bins + 1), density=True)

    fig = px.strip(df_filtered, x='flicker_offset_from_segment_start_min', title='Sleep flickers')
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-sleep-pressure/raw-flicker-location-within-sleep-segment')
    fig = go.Figure(data=[
        go.Bar(
            x=(bins + 1/n_bins/2)[:-1],
            y=hist,
            marker_color='rgb(56, 56, 56)',
        ),
        go.Scatter(
            x=df_filtered['flicker_percent_time_in_segment'],
            y=0.2 + np.random.rand(len(df_filtered)) * 0.4,
            mode='markers',
            marker=dict(size=12, color='rgba(8,162,74,0.5)'),
        ),
    ])
    fig.update_layout(title=f'Sleep flickers - percent time in segment - {segment_length_from_min} min to {segment_length_to_min}')
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-sleep-pressure/raw-flicker-percent-time-sleep-segment')
    fig = go.Figure(data=[
        go.Scatter(
            x=df_filtered['flicker_offset_from_segment_start_min'],
            y=np.random.rand(len(df_filtered)),
            mode='markers',
            marker=dict(size=12, color='rgba(56,56,56,0.5)'),
        ),
    ])
    fig.update_layout(title=f'Sleep flickers - absolute time in segment, start segment aligned - {segment_length_from_min} min to {segment_length_to_min}', width=1000, height=400)
    fig.update_xaxes(title=f'Flicker absolute location in minutes until sleep (t=0)')
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-sleep-pressure/raw-sleep-flicker-unscaled-start-segment-aligned')
    fig = go.Figure(data=[
        go.Scatter(
            x=df_filtered['flicker_offset_from_segment_end_min'] * -1,
            y=np.random.rand(len(df_filtered)),
            mode='markers',
            marker=dict(size=12, color='rgba(56,56,56,0.5)'),
        ),
    ])
    fig.update_layout(title=f'Sleep flickers - absolute time in segment, end segment aligned - {segment_length_from_min} min to {segment_length_to_min}', width=1000, height=400)
    fig.update_xaxes(title=f'Flicker absolute location in minutes until sleep (t=0)')
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-sleep-pressure/raw-sleep-flicker-unscaled-end-segment-aligned')

    #
    # NREM
    #
    # Plot location of flicker on number line for all flickers in segments > X seconds
    df = pd.DataFrame(data=nrem_flicker_segments, columns=['nrem_ix', 'nrem_segment_start_ix', 'nrem_segment_end_ix'])
    df['segment_length'] = df['nrem_segment_end_ix'] - df['nrem_segment_start_ix']
    df['segment_length_sec'] = df['segment_length'].astype(float) / 15
    df['flicker_offset_from_segment_start'] = df['nrem_ix'] - df['nrem_segment_start_ix']
    df['flicker_offset_from_segment_start_min'] = df['flicker_offset_from_segment_start'].astype(float) / 15 / 60
    df['flicker_offset_from_segment_end'] = df['nrem_segment_end_ix'] - df['nrem_ix']
    df['flicker_offset_from_segment_end_min'] = df['flicker_offset_from_segment_end'].astype(float) / 15 / 60
    df['flicker_percent_time_in_segment'] = df['flicker_offset_from_segment_start'] / (df['nrem_segment_end_ix'] - df['nrem_segment_start_ix'])
    df['dataset_name'] = optical_flow_flickers_df.iloc[df.nrem_ix]['dataset_name'].tolist()
    df['probe_identifier'] = (optical_flow_flickers_df.iloc[df.nrem_ix]['dataset_name'] + '-' + optical_flow_flickers_df.iloc[df.nrem_ix]['flicker_region']).tolist()

    minutes = 15 * 60
    df_filtered = df[(df.segment_length >= segment_length_from_min * minutes) & (df.segment_length <= segment_length_to_min * minutes)]
    hist, bins = np.histogram(df_filtered['flicker_percent_time_in_segment'], bins=np.linspace(0, 1, n_bins + 1), density=True)
    per_probe_histogram = [
        np.histogram(
            df_filtered[df_filtered.probe_identifier == probe_identifier]['flicker_percent_time_in_segment'],
            bins=np.linspace(0, 1, n_bins + 1),
            density=True,
        )[0]
        for probe_identifier in df_filtered.probe_identifier.unique()
    ]
    error_y = np.std(per_probe_histogram, axis=0) / np.sqrt(len(per_probe_histogram))
    # anova_df = pd.DataFrame(data=[
    #     [df_filtered[df_filtered.probe_identifier == p]['dataset_name']] +
    #     [df_filtered[df_filtered.probe_identifier == p]['probe_identifier']] +
    #     hist_list.tolist() +
    #     for hist_list, p in zip(per_probe_histogram, df_filtered.probe_identifier.unique())
    # ], columns=[])

    fig = px.strip(df_filtered, x='flicker_offset_from_segment_start_min', title='nrem flickers')
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-sleep-pressure/raw-flicker-location-within-nrem-segment')
    fig = go.Figure(data=[
        go.Bar(
            name='Flicker Histogram',
            x=(bins + 1/n_bins/2)[:-1],
            y=hist,
            error_y=dict(type='data', array=error_y, color='rgb(128,128,128)', visible=True),
            marker_color='rgb(56, 56, 56)',
        ),
        go.Scatter(
            name='Flicker Location',
            x=df_filtered['flicker_percent_time_in_segment'],
            y=0.2 + np.random.rand(len(df_filtered)) * 0.4,
            mode='markers',
            marker=dict(size=12, color='rgba(204,204,152,0.30)'),
        ),
    ])
    fig.update_layout(
        title=f'nrem segments - percent time in segment - '
              f'[WAKE|REM]-WITHIN-NREM flickers only - '
              f'{segment_length_from_min} min to {segment_length_to_min}',
        width=1000,
        height=400,
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
            ticktext=['Begin NREM | 0%', '20%', '40%', '60%', '80%', '100% | End NREM'],
        ),
    )
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-sleep-pressure/raw-flicker-percent-time-nrem-segment')
    fig = go.Figure(data=[
        go.Scatter(
            x=df_filtered['flicker_offset_from_segment_start_min'],
            y=np.random.rand(len(df_filtered)),
            mode='markers',
            marker=dict(size=12, color='rgba(56,56,56,0.5)'),
        ),
    ])
    fig.update_layout(title=f'nrem flickers - absolute time in segment, start segment aligned - {segment_length_from_min} min to {segment_length_to_min}', width=1000, height=400)
    fig.update_xaxes(title=f'Flicker absolute location in minutes until nrem (t=0)')
    fig.update_yaxes(
        title=f'[Bar] Histogram (density) of flickers location in segments of NREM<br/>'
              f'[Scatter] individual flickers relative to NREM segment'
    )
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-sleep-pressure/raw-nrem-flicker-unscaled-start-segment-aligned')
    fig = go.Figure(data=[
        go.Scatter(
            x=df_filtered['flicker_offset_from_segment_end_min'] * -1,
            y=np.random.rand(len(df_filtered)),
            mode='markers',
            marker=dict(size=12, color='rgba(56,56,56,0.5)'),
        ),
    ])
    fig.update_layout(title=f'nrem flickers - absolute time in segment, end segment aligned - {segment_length_from_min} min to {segment_length_to_min}', width=1000, height=400)
    fig.update_xaxes(title=f'Flicker absolute location in minutes until nrem (t=0)')
    _save_figure(fig, f'{BASEPATH_FIGURES}/figure-flickers-sleep-pressure/raw-nrem-flicker-unscaled-end-segment-aligned')


def print_stats_fig1_bac_from_audit():
    human_audit_file = f'{BASEPATH_AUDIT}/human-balanced-accuracy.json'
    cnn_audit_file = f'{BASEPATH_AUDIT}/modal-bac-per-region.json'

    # load human BAC audit (modal average)
    with open(human_audit_file, 'r') as f:
        human_modal_bac = json.load(f)

    # load CNN BAC audit (modal average)
    with open(cnn_audit_file, 'r') as f:
        cnn_modal_bac = json.load(f)

    # Convert to a single dataframe
    df_data = []
    for hmb in human_modal_bac:
        dataset_name = hmb[0]
        for human_accuracy in hmb[1]:
            entity = 'human'
            region = 'whole-brain'
            modal_bac = human_accuracy[3]
            df_data.append((entity, dataset_name, region, modal_bac))
    for modal_accuracy in cnn_modal_bac:
        entity = 'cnn'
        dataset_name = modal_accuracy[2]
        region = modal_accuracy[1]
        modal_bac = modal_accuracy[4]
        df_data.append((entity, dataset_name, region, modal_bac))

    df = pd.DataFrame(data=df_data, columns=['entity', 'dataset_name', 'region', 'modal_bac'])
    print(df)

    # Run ANOVA
    mask_human = df['entity'] == 'human'
    mask_cnn = df['entity'] == 'cnn'

    t_test_human_cnn = scipy.stats.ttest_ind(df[mask_human].modal_bac, df[mask_cnn].modal_bac)
    print(t_test_human_cnn)
    anova_all_cnn = scipy.stats.f_oneway(*[df[df.region == r].modal_bac for r in df.region.unique()])
    print(anova_all_cnn)


def main():
    with PrintTiming('PROGRESS: Stats for figure 1 - BAC'):
        print_stats_fig1_bac_from_audit()

    with PrintTiming('PROGRESS: Data optical flow results'):
        optical_flow_dataframes = get_optical_flow_dataframe(DATASET_NAMES, LONG_MEDIAN_WINDOW, SHORT_MEDIAN_WINDOW, use_cached=False)

    with PrintTiming('PROGRESS: Get flicker dataframes'):
        flicker_dataframes = get_flicker_dataframes(flicker_calling_params='flicker-calling-standard', dataset_names=DATASET_NAMES, model_size='c64k', use_cache=False)
    with PrintTiming('PROGRESS: Compute time to last state change'):
        flicker_dataframes = data_compute_time_to_from_last_state_change(flicker_dataframes=flicker_dataframes)
    with PrintTiming('PROGRESS: Merge optical flow and flickers'):
        optical_flow_flickers_df = merge_flicker_optical_flow(flicker_dataframes, optical_flow_dataframes, use_cache=False)

    with PrintTiming('PROGRESS: Flicker rates by state'):
        plot_flicker_rates(optical_flow_flickers_df)

    with PrintTiming('PROGRESS: Plot flickers by percent of time into wake/sleep'):
        plot_flickers_by_percent_time_in_wake_sleep(optical_flow_flickers_df)

    exit(0)

    # with PrintTiming('PROGRESS: Plot flicker proportion of wake'):
    #     plot_flickers_proportion_wake(flicker_dataframes=flicker_dataframes, bins=5)
    #     plot_flickers_proportion_wake_aggregated(flicker_dataframes=flicker_dataframes, bins=5)
    #     for r in (reg for reg in Regions if reg != Regions.BADPROBE):
    #         plot_flickers_proportion_wake_aggregated(flicker_dataframes=flicker_dataframes, bins=5, region_filter=r)

    with PrintTiming(f'PROGRESS: Plot figure-1 human labeling compared to CNN predictions'):
        minutes = 60 * 15
        hours = 60 * minutes

        # Chosen cases
        plot_human_labels_vs_cnn(
            dataset_name='CAF42',
            versions=[
                f'wnr-v14-perregion-c64k-{p.channel_from}-{p.channel_to}-run1'
                for p in [p for p in PROBE_REGION_CHANNEL_LIST if p.dataset_name == 'CAF42']
            ],
            offset=16 * hours + int(51.534849899999996 * minutes),
            length=int(0.75 * minutes),
            testset_only=False
        )

        plot_human_labels_vs_cnn(
            dataset_name='CAF99',
            versions=[
                f'wnr-v14-perregion-c64k-{p.channel_from}-{p.channel_to}-run1'
                for p in [p for p in PROBE_REGION_CHANNEL_LIST if p.dataset_name == 'CAF99']
            ],
            offset=7 * hours + 22 * minutes,
            length=30 * minutes,
            testset_only=False
        )
        plot_human_labels_vs_cnn(
            dataset_name='CAF99',
            versions=[
                f'wnr-v14-perregion-c64k-{p.channel_from}-{p.channel_to}-run1'
                for p in [p for p in PROBE_REGION_CHANNEL_LIST if p.dataset_name == 'CAF99']
            ],
            offset=20 * hours + 17 * minutes,
            length=25 * minutes,
            testset_only=False
        )

        # Plot all 1-hr segments for all datasets
        dataset_names = DATASET_NAMES  # ['CAF106']
        versions = [
            [
                f'wnr-v14-perregion-c64k-{p.channel_from}-{p.channel_to}-run1'
                for p in [p for p in PROBE_REGION_CHANNEL_LIST if p.dataset_name == dn]
            ]
            for dn in dataset_names
        ]

        args = [(d, v, h) for (d, v), h in itertools.product(zip(dataset_names, versions), [hr * 60 * minutes for hr in range(24)])]
        # plot_human_labels_vs_cnn(dataset_name=dn, versions=versions, offset=hr * 60 * minutes, length=60 * minutes, testset_only=False)
        # plot_human_labels_vs_cnn(dataset_name='EAB40', versions=versions, offset=hr*30*minutes, length=30*minutes, testset_only=False)
        common_utils.map2(
            plot_human_labels_vs_cnn,
            args=args,
            fixed_values=dict(length=60 * minutes, testset_only=False),
            parallelism=1,
        )

    with PrintTiming(f'PROGRESS: Plot all flicker LFP/Human-predictions/CNN-predictions'):
        plot_human_cnn_lfp_all_flickers(optical_flow_flickers_df)

    with PrintTiming('PROGRESS: Data modal_bac'):
        versions_per_region = _get_versions(dataset_names=DATASET_NAMES, run_version_regex=RegexDef.PERREGION, exclude_bad_probes=True)
        perregion_modal_bac_results = compute_modal_bac(versions_per_region, version_regex=RegexDef.PERREGION, mode_length_sec=30)
        _save_json(perregion_modal_bac_results, f'{BASEPATH_AUDIT}/modal-bac-per-region.json')

    with PrintTiming('PROGRESS: Data human accuracy by region'):
        list_tuples_human_accuracy_by_region, tuple_dataset_name_human_accuracy = compute_human_accuracy_by_region_and_dataset(mode_length_samples=28)
        _save_json(tuple_dataset_name_human_accuracy, f'{BASEPATH_AUDIT}/human-balanced-accuracy.json')

    with PrintTiming('PROGRESS: Plot accuracy'):
        plot_modal_bac(perregion_modal_bac_results, tuple_dataset_name_human_accuracy)

    with PrintTiming('PROGRESS: Plot QAP'):
        plot_quiet_active_passive_bar(optical_flow_flickers_df)

    with PrintTiming(f'PROGRESS: Plot flicker alignment distribution'):
        plot_flicker_alignment_distribution(optical_flow_flickers_df, flicker_flavors=[FlickerFlavor.NREM_WITHIN_WAKE], regions=set(Regions) - {Regions.BADPROBE}, n_bins=10)
        plot_flicker_alignment_distribution(optical_flow_flickers_df, flicker_flavors=[FlickerFlavor.NREM_WITHIN_WAKE], regions=set(Regions) - {Regions.BADPROBE}, n_bins=20)

    # Plot: flicker raster with motion
    for dataset_name in DATASET_NAMES:
        with PrintTiming(f'PROGRESS: Plot flicker raster with motion {dataset_name}'):
            mask = optical_flow_flickers_df['dataset_name'] == dataset_name
            plot_flicker_raster_with_motion(optical_flow_flickers_df[mask], LONG_MEDIAN_THRESHOLD, SHORT_MEDIAN_THRESHOLD)

    # Plot: flicker raster at specific ranges for examples
    for dataset_name, plot_range in zip(['CAF42', 'CAF26', 'CAF42', 'CAF42', 'CAF26'], [(2596, 2678), (42881, 42940), (32267, 32294), (35235, 35356), (51435, 51500)]):
        with PrintTiming(f'PROGRESS: Plot flicker raster at range {dataset_name} {plot_range}'):
            mask = optical_flow_flickers_df['dataset_name'] == dataset_name
            plot_flicker_raster_with_motion(optical_flow_flickers_df[mask], LONG_MEDIAN_THRESHOLD, SHORT_MEDIAN_THRESHOLD, plot_range=plot_range)

    # Plot: plot_flow_after_flicker
    with PrintTiming('PROGRESS: Plot flow after flicker'):
        plot_flow_after_flicker(optical_flow_flickers_df, movement_threshold=MOVEMENT_THRESHOLD)

    del optical_flow_flickers_df

    # Plot: flickers against movement
    with PrintTiming('PROGRESS: Plot flickers against movement'):
        plot_flickers_against_motion(flicker_dataframes, optical_flow_dataframes)

    # Plot: flickers by time of day
    with PrintTiming('PROGRESS: Plot flickers by time of day'):
        for interval_hrs in [1, 2, 3, 4, 6]:
            plot_flickers_by_time_of_day(flicker_dataframes=flicker_dataframes, interval_hrs=interval_hrs)
        for region in [r for r in Regions if r != Regions.BADPROBE]:
            plot_flickers_by_time_of_day(flicker_dataframes=flicker_dataframes, interval_hrs=2, include_regions=[region])
            plot_flickers_by_time_of_day(
                flicker_dataframes=flicker_dataframes,
                interval_hrs=2,
                include_regions=[region],
                include_flicker_flavors=[FlickerFlavor.WAKE_WITHIN_NREM, FlickerFlavor.WAKE_WITHIN_REM,
                                         FlickerFlavor.NREM_WITHIN_REM, FlickerFlavor.REM_WITHIN_NREM],
                name='filter-by-sleep-flickers-and-region',
            )
            plot_flickers_by_time_of_day(
                flicker_dataframes=flicker_dataframes,
                interval_hrs=2,
                include_regions=[region],
                include_flicker_flavors=[FlickerFlavor.NREM_WITHIN_WAKE, FlickerFlavor.REM_WITHIN_WAKE],
                name='filter-by-wake-flickers-and-region',
            )
        plot_flickers_by_time_of_day(
            flicker_dataframes=flicker_dataframes,
            interval_hrs=2,
            include_flicker_flavors=[FlickerFlavor.WAKE_WITHIN_NREM, FlickerFlavor.WAKE_WITHIN_REM,
                                     FlickerFlavor.NREM_WITHIN_REM, FlickerFlavor.REM_WITHIN_NREM],
            name='filter-by-sleep-flickers',
        )
        plot_flickers_by_time_of_day(
            flicker_dataframes=flicker_dataframes,
            interval_hrs=2,
            include_flicker_flavors=[FlickerFlavor.NREM_WITHIN_WAKE, FlickerFlavor.REM_WITHIN_WAKE],
            name='filter-by-wake-flickers',
        )

    # Plot: flicker rate by time in state
    print('PROGRESS: Plot flicker rate by time in state')
    plot_flickers_rate_by_time_in_state(flicker_dataframes=flicker_dataframes, bins=10)

    # Plot: flicker time since last state change
    print('PROGRESS: Plot flicker scatter by time since last state change')
    flickers_by_flavor = get_flickers_by_flavor(flicker_dataframes=flicker_dataframes)
    for flicker_flavor in [f for f in FlickerFlavor if f.value[0] != f.value[1]]:
        plot_flickers_scatter_by_time_since_last_state_change(flicker_flavor=flicker_flavor, flickers_by_flavor=flickers_by_flavor)

    # Plot: quiescence vs active wake bar plot
    print('PROGRESS: Plot quiescence vs active wake bar plot')
    plot_flickers_by_quiescence_and_active(flicker_dataframes, optical_flow_dataframes)

    # Data: highpass models within-sample shuffle
    print('PROGRESS: Data for highpass models within-sample shuffle')
    runs_per_sample_shuffled_750hz_highpass = pull_models_bac_by_version(version_channel_regex=RegexDef.HIGHPASS_WITHIN_SAMPLE_SHUFFLE)
    runs_unshuffled_750hz_highpass = pull_models_bac_by_version(version_channel_regex=RegexDef.HIGHPASS_SAMPLE_SIZE)
    runs_per_sample_shuffled_750hz_highpass_df = _runs_list_to_df(runs_per_sample_shuffled_750hz_highpass, RegexDef.HIGHPASS_WITHIN_SAMPLE_SHUFFLE)
    runs_per_sample_shuffled_750hz_highpass_df.to_csv(f'{BASEPATH_AUDIT}/highpass-750-within-sample-shuffle-models.csv', index=False)
    runs_unshuffled_750hz_highpass_df = _runs_list_to_df(runs_unshuffled_750hz_highpass, RegexDef.HIGHPASS_SAMPLE_SIZE)
    runs_unshuffled_750hz_highpass_df.to_csv(f'{BASEPATH_AUDIT}/highpass-750-unshuffled-models.csv', index=False)

    # Plot: highpass models within-sample shuffle
    print('PROGRESS: Plot highpass models within-sample shuffle')
    plot_unity_line_accuracy_highpass_with_shuffle(shuffled_df=runs_per_sample_shuffled_750hz_highpass_df,
                                                   unshuffled_df=runs_unshuffled_750hz_highpass_df)

    # Data: highpass and highpass shuffled data
    print('PROGRESS: Data highpass + shuffle')
    chan_highpass_raw = 69
    offset_highpass_raw = 3069000
    data_highpass_noshuff = _get_raw_data_sample('s3://hengenlab/CAF42/Neural_Data/highpass_750/Headstages_320_Channels_int16_2020-09-14_17-35-38.bin', 320, offset_highpass_raw, 65536)
    data_highpass_withshuff = _get_raw_data_sample('s3://hengenlab/CAF42/Neural_Data/highpass_750/classpermuted/Headstages_320_Channels_int16_2020-09-14_17-35-38.bin', 320, offset_highpass_raw, 65536)

    # Plot: highpass with and without shuffle
    print('PROGRESS: Plot highpass + shuffle distributions')
    plot_ranges = [
        np.max(np.abs(np.concatenate((data_highpass_noshuff[chan_highpass_raw, :2**x], data_highpass_withshuff[chan_highpass_raw, :2**x]))))
        for x in range(18)
    ]
    plot_raw_data(data=data_highpass_noshuff[chan_highpass_raw, :], title=f'750hz highpass data, unshuffled, channel {chan_highpass_raw}, offset {offset_highpass_raw}', filename=f'unshuffled-highpass-750-ch-{chan_highpass_raw}-offset-{offset_highpass_raw}', plot_sizes=[2**x for x in range(18)], plot_ranges=plot_ranges)
    plot_raw_data(data=data_highpass_withshuff[chan_highpass_raw, :], title=f'750hz highpassed, shuffled data, channel {chan_highpass_raw}, offset {offset_highpass_raw}', filename=f'shuffled-highpass-750-ch-{chan_highpass_raw}-offset-{offset_highpass_raw}', plot_sizes=[2**x for x in range(18)], plot_ranges=plot_ranges)

    # Data: highpass models
    print('PROGRESS: Data highpass models')
    runs_per_sample_shuffled_750hz_highpass = pull_models_bac_by_version(version_channel_regex=RegexDef.HIGHPASS_SAMPLE_SIZE)
    runs_highpass_sample_size_with_shuffle = pull_models_bac_by_version(version_channel_regex=RegexDef.HIGHPASS_SAMPLE_SIZE_DATASET_SHUFFLE)
    _runs_list_to_df(runs_per_sample_shuffled_750hz_highpass, RegexDef.HIGHPASS_SAMPLE_SIZE).to_csv(f'{BASEPATH_AUDIT}/highpass-sample-size-modal-bac.csv', index=False)
    _runs_list_to_df(runs_highpass_sample_size_with_shuffle, RegexDef.HIGHPASS_SAMPLE_SIZE_DATASET_SHUFFLE).to_csv(f'{BASEPATH_AUDIT}/highpass-sample-size-modal-bac-with-shuffle.csv', index=False)

    # Plot: highpass sample size with shuffle bar plot
    print('PROGRESS: Plot highpass sample size with shuffle bar plot')
    plot_bar_highpass_sample_size_with_shuffle(runs_per_sample_shuffled_750hz_highpass, runs_highpass_sample_size_with_shuffle)

    # Plot: Confusion matrices
    print('PROGRESS: Plot confusion matrices')
    plot_human_cnn_confusion_matrices(perregion_modal_bac_results, tuple_dataset_name_human_accuracy)

    # Data: Per sample size accuracies
    print('PROGRESS: Data per-sample-size runs')
    runs_per_sample_size = pull_models_bac_by_version(version_channel_regex=RegexDef.SAMPLE_SIZE)

    # Plot: Per sample size confusion matrices
    print('PROGRESS: Per sample size modal bac calculation and confusion matrices')
    versions_per_sample_size = [Version(ba.dataset_name, ba.version) for ba in runs_per_sample_size if _get_size(ba.version, RegexDef.SAMPLE_SIZE) <= 1024 and ba.bac > 0.50]
    persamplesize_modal_bac_results = compute_modal_bac(versions_per_sample_size, version_regex=RegexDef.SAMPLE_SIZE, mode_length_sec=30)
    _save_json(persamplesize_modal_bac_results, f'{BASEPATH_AUDIT}/modal-bac-per-sample-size.json')
    plot_confusion_matrices_per_sample_size(persamplesize_modal_bac_results)

    print('PROGRESS: Data raw histogram CSV')
    with smart_open.open(f'{BASEPATH_MANUAL_DATA}/per-sample-histograms-highpass.csv', 'r') as f:
        per_sample_histogram_highpass_df = pd.read_csv(f, header=None, names=[
            'sleep_state', 'region', 'dataset_name', 'channel', 'file_name', *list(range(1024))
        ])

    print('PROGRESS: Plot single channel distributions for highpassed and highpassed shuffled')
    ds_highpass_region_chan = [('CAF34', c, _get_region('CAF34', c)) for c in range(256)]
    list(itertools.starmap(functools.partial(plot_single_channel_distributions, per_sample_histogram_df=per_sample_histogram_highpass_df, basedir='per-channel-highpass'), ds_highpass_region_chan))

    print('PROGRESS: Data raw histogram CSV')
    with smart_open.open(f'{BASEPATH_MANUAL_DATA}/per-sample-histograms.csv', 'r') as f:
        per_sample_histogram_df = pd.read_csv(f, header=None, names=[
            'sleep_state', 'region', 'dataset_name', 'channel', 'file_name', *list(range(1024))
        ])

    print('PROGRESS: Plot per sample distributions')
    plot_per_state_region_distributions(per_sample_histogram_df)

    print('PROGRESS: Plot single channel distributions for small input size with good scores')
    ds_region_chan = list(set([
        (run.dataset_name, _get_channel(run.version), _get_region(run.dataset_name, _get_channel(run.version)))
        for run in runs_per_sample_size
        # if (run.bac > 0.35 and _get_size(run.version) <= 128) or
        # (_get_region(run.dataset_name, _get_channel(run.version)) in [Regions.LGN, Regions.NAC])
    ]))
    ds_region_chan = ds_region_chan + [('CAF106', c, _get_region('CAF106', c)) for c in range(512)]
    list(itertools.starmap(functools.partial(plot_single_channel_distributions, per_sample_histogram_df=per_sample_histogram_df, basedir='per-channel'), ds_region_chan))

    # Data: Whole brain per-sample-size models
    print('PROGRESS: Data whole brain runs')
    runs_whole_brain_per_sample_size = pull_models_bac_by_version(RegexDef.WHOLE_BRAIN, exclude_bad_probes=False)
    _save_json(runs_whole_brain_per_sample_size, f'{BASEPATH_AUDIT}/runs-whole-brain-per-sample-size.json')

    # Plot: Sample size bar plot
    print('PROGRESS: Sample size bar plot')
    plot_sample_size_bar_plot(runs_per_sample_size, min_accuracy=MIN_ACCURACY)

    # Plot: Whole brain sample size graphs
    print('PROGRESS: Plot whole brain per-sample-size data')
    plot_sample_size_whole_brain(runs_whole_brain_per_sample_size)

    # ANOVA: 2-way spiking/non-spiking + brain_region and 1-way on all samples by brain region
    print('PROGRESS: Compute ANOVA data')
    with smart_open.open(f'{BASEPATH_AUDIT}/sample-size-per-run-spiking.json') as fspiking, \
            smart_open.open(f'{BASEPATH_AUDIT}/sample-size-per-run-non-spiking.json') as fnonspiking:
        sample_size_per_run_spiking = json.load(fspiking)
        sample_size_per_run_non_spiking = json.load(fnonspiking)
    compute_anova(sample_size_per_run_spiking, sample_size_per_run_non_spiking, min_accuracy=MIN_ACCURACY)

    print('PROGRESS: Proportion of models above chance by region')
    for ss in [1, 4, 16]:
        plot_proportion_of_model_above_chance(runs_per_sample_size=runs_per_sample_size, min_bac=0.34, sample_size=ss)

    print('PROGRESS: Plot highpass data')
    highpass_bac = pull_models_bac_by_version(RegexDef.HIGHPASS)
    plot_highpass_runs(highpass_bac)

    print('PROGRESS: Done')


if __name__ == '__main__':
    main()
