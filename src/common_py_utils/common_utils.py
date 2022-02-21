import types
import boto3
import os
import urllib
import pathlib
import multiprocessing
import multiprocessing.managers
from multiprocessing import shared_memory
import pickle
import numpy as np
from botocore.exceptions import ClientError
import tensorflow.compat.v1 as tf
import json
import itertools
import functools
import inspect
from typing import Iterable, Tuple, Callable, Union
import diskcache
import recordtype
from enum import Enum
import yaml


# S3 client for boto3
s3_client = boto3.client('s3', endpoint_url=os.environ['ENDPOINT_URL'] if 'ENDPOINT_URL' in os.environ else None)

BASEPATH_PARAMETER_FILES = '../parameter_files'
Probe = recordtype.recordtype(typename='Probe', field_names='dataset_name probe_id region channel_from channel_to')


class Regions(Enum):
    CA1 = 'CA1'
    V1V2 = 'VISp'
    NAC = 'ACB'
    S1 = 'SSp'
    M1M2 = 'MOp'
    CP = 'CP'
    SC = 'SC'
    ACAD = 'ACA'
    RSPV = 'RSP'
    LGN = 'LGN'
    BADPROBE = 'BADPROBE'


VALID_BRAIN_REGION_NAMES = ['CA1', 'V1V2', 'NAC', 'S1', 'M1M2', 'CP', 'SC', 'ACAD', 'RSPV', 'LGN', 'BADPROBE']

# source of record for spiking and non-spiking channels chosen for plotting. Also used in video LFP overlays.
SPIKING_CHANNELS = {
    'CAF26': [21, 31, 36, 86, 89, 104, 147, 167, 177],
    'CAF34': [14, 50, 57, 126, 99, 106, 163, 170, 178, 193, 220, 248],
    'CAF42': [38, 23, 55, 69, 118, 99, 152, 168, 183, 206, 230, 242, 271, 278, 259],
    'CAF69': [55, 29, 0, 69, 104, 122, 138, 148, 184, 212, 218, 224],
    'CAF77': [66, 93, 110, 187, 140, 178, 192, 213, 228],
    'CAF99': [3, 6, 11, 67, 78, 103, 162, 178, 134, 197, 230, 238, 283, 288, 303, 351, 356, 363, 414, 397, 406, 451, 472, 477],
    'CAF106': [12, 35, 63, 96, 78, 88, 133, 146, 147, 203, 210, 213, 309, 262, 290, 400, 412, 425, 463, 475, 421],
    'EAB40': [33, 10, 14, 64, 65, 73, 130, 163, 151, 199, 215, 225],
    'EAB50_5state': [51, 39, 40, 74, 86, 99, 135, 132, 137, 194, 199, 201, 258, 260, 274, 331, 335, 347, 386, 395, 390],
}
NON_SPIKING_CHANNELS = {
    'CAF42': [9, 14, 27, 110, 102, 81, 172, 179, 173, 219, 216, 218, 260, 264, 306],
    'CAF26': [10, 14, 28, 83, 88, 112, 136, 176, 183],
    'CAF77': [71, 96, 105, 134, 157, 132, 195, 202, 233],
    'EAB50_5state': [1, 4, 29, 73, 75, 112, 190, 180, 174, 195, 211, 215, 256, 271, 282, 339, 350, 371, 399, 409, 436],
    'CAF34': [9, 15, 17, 76, 115, 90, 144, 148, 167, 195, 202, 240],
    'CAF69': [1, 39, 63, 111, 116, 75, 135, 169, 180, 204, 216, 238],
    'EAB40': [3, 7, 11, 74, 87, 110, 145, 178, 186, 192, 200, 220],
    'CAF99': [0, 60, 4, 112, 72, 124, 182, 159, 183, 193, 234, 237, 256, 265, 281, 332, 348, 355, 433, 399, 405, 448, 465, 496],
    'CAF106': [24, 25, 59, 81, 85, 93, 162, 132, 137, 200, 215, 238, 260, 317, 308, 437, 391, 399, 496, 448, 452],
}

GAIN = 0.19073486328125     # hard coded default from hengenlab/neuraltoolkit

PROBE_REGION_CHANNEL_LIST = [
    Probe('CAF26', 'p1', Regions.M1M2, 0, 64),
    Probe('CAF26', 'p2', Regions.CA1, 64, 128),
    Probe('CAF26', 'p3', Regions.S1, 128, 192),

    Probe('CAF34', 'p1', Regions.S1, 0, 64),
    Probe('CAF34', 'p2', Regions.M1M2, 64, 128),
    Probe('CAF34', 'p3', Regions.CA1, 128, 192),
    Probe('CAF34', 'p4', Regions.CP, 192, 256),

    Probe('CAF42', 'p1', Regions.M1M2, 0, 64),
    Probe('CAF42', 'p2', Regions.ACAD, 64, 128),
    Probe('CAF42', 'p3', Regions.CA1, 128, 192),
    Probe('CAF42', 'p4', Regions.RSPV, 192, 256),
    Probe('CAF42', 'p5', Regions.V1V2, 256, 320),

    Probe('CAF69', 'p1', Regions.ACAD, 0, 64),
    Probe('CAF69', 'p2', Regions.RSPV, 64, 128),
    Probe('CAF69', 'p3', Regions.V1V2, 128, 192),
    Probe('CAF69', 'p4', Regions.CA1, 192, 256),

    Probe('CAF77', 'p1', Regions.BADPROBE, 0, 64),  # was CA1
    Probe('CAF77', 'p2', Regions.RSPV, 64, 128),
    Probe('CAF77', 'p3', Regions.ACAD, 128, 192),
    Probe('CAF77', 'p4', Regions.V1V2, 192, 256),

    Probe('EAB40', 'p1', Regions.S1, 0, 64),
    Probe('EAB40', 'p2', Regions.CA1, 64, 128),
    Probe('EAB40', 'p3', Regions.M1M2, 128, 192),
    Probe('EAB40', 'p4', Regions.M1M2, 192, 256),

    Probe('EAB50_5state', 'p1', Regions.CP, 0, 64),
    Probe('EAB50_5state', 'p2', Regions.CP, 64, 128),
    Probe('EAB50_5state', 'p3', Regions.M1M2, 128, 192),
    Probe('EAB50_5state', 'p4', Regions.CA1, 192, 256),
    Probe('EAB50_5state', 'p5', Regions.CA1, 256, 320),
    Probe('EAB50_5state', 'p6', Regions.S1, 320, 384),
    Probe('EAB50_5state', 'p7', Regions.SC, 384, 448),
    Probe('EAB50_5state', 'p8', Regions.BADPROBE, 448, 512),  # was V1/V2

    Probe('CAF99', 'p1', Regions.CP, 0, 64),
    Probe('CAF99', 'p2', Regions.S1, 64, 128),
    Probe('CAF99', 'p3', Regions.M1M2, 128, 192),
    Probe('CAF99', 'p4', Regions.NAC, 192, 256),
    Probe('CAF99', 'p5', Regions.LGN, 256, 320),
    Probe('CAF99', 'p6', Regions.V1V2, 320, 384),
    Probe('CAF99', 'p7', Regions.RSPV, 384, 448),
    Probe('CAF99', 'p8', Regions.SC, 448, 512),

    Probe('CAF106', 'p1', Regions.M1M2, 0, 64),
    Probe('CAF106', 'p2', Regions.S1, 64, 128),
    Probe('CAF106', 'p3', Regions.CP, 128, 192),
    Probe('CAF106', 'p4', Regions.NAC, 192, 256),
    Probe('CAF106', 'p5', Regions.LGN, 256, 320),
    Probe('CAF106', 'p6', Regions.BADPROBE, 320, 384),  # was RSPv
    Probe('CAF106', 'p7', Regions.SC, 384, 448),
    Probe('CAF106', 'p8', Regions.V1V2, 448, 512),
]


def get_region(dataset_name: str, channel: int) -> Regions:
    probes = [
        p for p in PROBE_REGION_CHANNEL_LIST
        if p.dataset_name == dataset_name and p.channel_from <= channel < p.channel_to
    ]
    assert len(probes) == 1, f'No region found for {dataset_name}, channel {channel}'
    return probes[0].region


def get_fps(dataset_name: str, default_fps: int = 15):
    """ Get FPS (frames per sec) from parameter file. """
    with open(f'{BASEPATH_PARAMETER_FILES}/dataset-{dataset_name}.yaml') as f:
        params = yaml.safe_load(f)
        fps = params['datasetparams'].get('video_fps', default_fps)
    return fps


class SimpleNamespace(types.SimpleNamespace):
    def __add__(self, other):
        assert isinstance(other, SimpleNamespace)
        return SimpleNamespace(**self.__dict__, **other.__dict__)

    def __getitem__(self, item):
        return self.__dict__[item]


def get_raw_data_sample(filepath: str, n_channels: int, offset: int, size: int, cache: diskcache.Cache = None):
    """ Reads (and caches) a segment of raw data from a specified file (s3 or local). """
    result = None
    cache_key = ('__get_raw_data_sample__', filepath, n_channels, offset, size)

    if cache is not None:
        result = cache.get(cache_key, default=None)

    if result is None:
        with smart_open.open(filepath, 'rb') as f_orig:
            f_orig.seek(8 + 2 * n_channels * offset)  # Seek to correct offset
            b = f_orig.read(size * 2 * n_channels)  # Read file from cache location
            data = np.frombuffer(b, dtype=np.int16).reshape((n_channels, -1), order='F')
            result = data * GAIN

            if cache is not None:
                cache[cache_key] = result

    return result


def file_exists(filename):
    """
    Simple test for whether a file exists or not, supports local and S3.
    This is implemented as a utility function because supporting multiple platforms (s3 and local) is not trivial.
    Issue history:
        - Using tensorflow for this functionality failed when libcurl rejected an expired cert.
        - Using PyFilesystem is a bad choice because it doesn't support streaming
        - Using smart_open supports streaming but not basic file operations like size and exists

    :param filename: file path + name, local or S3
    :return: boolean exists|not_exists
    """
    if filename.startswith('s3://'):
        o = urllib.parse.urlparse(filename)
        try:
            s3_client.head_object(Bucket=o.netloc, Key=o.path[1:])
            exists = True
        except ClientError:
            exists = False
    else:
        exists = os.path.isfile(filename)

    return exists


def file_size(filename):
    """
    Gets file size, supports local and S3 files, same issues as documented in function file_exists
    :param filename: file path + name, local or S3
    :return: int file size in bytes
    """
    if filename.startswith('s3://'):
        o = urllib.parse.urlparse(filename)
        sz = s3_client.head_object(Bucket=o.netloc, Key=o.path[1:])['ContentLength']
    else:
        sz = os.path.getsize(filename)

    return sz


def file_list(filepath):
    """ Returns a list of files local or S3 in order of descending order of last modified time """
    if filepath.startswith('s3://'):
        o = urllib.parse.urlparse(filepath)
        response = s3_client.list_objects(Bucket=o.netloc, Prefix=o.path[1:])

        if 'Contents' not in response:
            raise FileNotFoundError(filepath)

        files = [
            f['Key'].split('/')[-1]
            for f in sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
        ]
    else:
        files = sorted(pathlib.Path(filepath).iterdir(), key=os.path.getmtime, reverse=True)
        files = [f.name for f in files]

    return files


def create_shared_memory_dict(sample: dict):
    """
    Higher level convenience function wrapping the more generic create_shared_memory(*objects).
    Scans the dictionary for ndarray values and serializes them using the efficient shm/numpy-copy method.

    :param sample: a dictionary of key, value pairs. Nesting NOT supported.
    :return: A unique shared memory string that can be passed to read_shared_memory_dict to get the same dict back
    """
    if sample is not None:
        ndarray_keys = [k for k, v in sample.items() if isinstance(v, np.ndarray)]
        ndarray_values = [v for k, v in sample.items() if isinstance(v, np.ndarray)]
        sample_filtered = {k: None if isinstance(v, np.ndarray) else v for k, v in sample.items()}
        name = create_shared_memory(sample_filtered, ndarray_keys, *ndarray_values)
    else:
        name = create_shared_memory(None)

    return name


def deserialize_shared_memory_dict(name: str):
    """
    Higher level convenience function wrapping the more generic deserialize_shared_memory(name).

    Deserializes the output when create_shared_memory_dict() was used to serialize a
    dictionary of ndarray values efficiently

    :param name: shared memory name as returned by create_shared_memory_dict.
    :return: a single dictionary identical to the one passed to create_shared_memory_dict
    """
    vals = deserialize_shared_memory(name)
    sample = vals[0] if vals is not None else None

    if sample is not None:
        for i, k in enumerate(vals[1], 2):
            sample[k] = vals[i]

    return sample


def create_shared_memory(*objects):
    """
    Creates a shared memory object of N objects, with (large) ndarrays handled outside of pickle
    for efficiency (because pickling large ndarrays is slow and locks the GIL).

    Serialization format, all little endian byte order, repeats for N objects:
        Object type (0=pickle, 1=ndarray), 1 byte
        Pickle object length N, 4 bytes unsigned int
        Pickled object data, bytes of length N
        (optional for ndarrays) ndarray bytes data of size `length`

    For ndarrays there are two objects in sequence, the first is a pickled
    dictionary containing the ndarray information, following the format described above.
        {
            'length': int,              # byte length of ndarray that immediately follows this pickle object
            'shape': tuple of ints,     # shape of the ndarray
            'dtype': np.dtype object,   # dtype of the ndarray
        }

    Deserialization of large ndarrays needs to be done manually to achieve high IO speeds, see issue:
    https://stackoverflow.com/questions/62352670

    :param objects: Any number of objects to be serialized to a shared memory byte array
    :return: A unique shared memory string that can be passed to read_shared_memory to get the same objects back
    """
    # Pickle all objects except ndarray. For ndarrays we create a meta object describing the ndarray and serialize that meta object
    # large ndarrays are an order of magnitude slower to deserialize with pickle than with numpy directly.
    objects_serialized = [
        pickle.dumps({'length': o.nbytes, 'shape': o.shape, 'dtype': o.dtype})
        if isinstance(o, np.ndarray)
        else pickle.dumps(o)
        for o in objects
    ]
    # compute the size of the shared memory object
    size_bytes = sum([
        1 + 4 + len(obj_bytes) + (obj.nbytes if isinstance(obj, np.ndarray) else 0)
        for obj, obj_bytes in zip(objects, objects_serialized)
    ])
    shm = shared_memory.SharedMemory(create=True, size=size_bytes)

    # write to shared memory buffer
    ix = 0
    for obj, obj_bytes in zip(objects, objects_serialized):
        shm.buf[ix:ix + 1] = isinstance(obj, np.ndarray).to_bytes(1, byteorder='little')  # 1 for ndarray else 0
        ix += 1
        shm.buf[ix:ix + 4] = len(obj_bytes).to_bytes(4, byteorder='little')
        ix += 4
        shm.buf[ix:ix + len(obj_bytes)] = obj_bytes
        ix += len(obj_bytes)
        if isinstance(obj, np.ndarray):
            nd_copy_buffer = np.ndarray(obj.shape, dtype=obj.dtype, buffer=shm.buf[ix:ix + obj.nbytes])
            nd_copy_buffer[...] = obj[...]
            ix += obj.nbytes

    # close, but do not unlink the shared memory device, it will remain available to other processes
    shm.close()

    # return the string name of the shared memory device
    return shm.name


def deserialize_shared_memory(name: str):
    """
    Reads a shared memory block encoded with create_shared_memory.
    
    This will deserialize a shared memory block created with create_shared_memory. Numpy arrays will be
    efficiently deserialized and then copied with np.copy() which does not lock the GIL, therefore this
    function can be utilized by multiple threads for large memory blocks and utilize more than a single core.

    :return: either a single object or tuple of objects as were originally passed to create_shared_memory
    """
    shm = shared_memory.SharedMemory(name=name, create=False)
    results = []
    ix = 0

    while ix < len(shm.buf) - 1:
        is_ndarray = bool(shm.buf[ix])
        ix += 1
        obj_len = int.from_bytes(shm.buf[ix:ix + 4], byteorder='little')
        ix += 4
        obj = pickle.loads(shm.buf[ix:ix + obj_len])
        ix += obj_len
        if is_ndarray:
            ndarray_len = obj['length']
            obj = np.ndarray(shape=obj['shape'], dtype=obj['dtype'], buffer=shm.buf[ix:ix + ndarray_len])
            obj = obj.copy()  # numpy copy operation does not lock the GIL
            ix += ndarray_len
        results.append(obj)

    shm.unlink()
    shm.close()

    return tuple(results) if len(results) > 1 else results[0]


# def release_shared_memory(shm:shared_memory.SharedMemory):
#     """ Releases a shared memory block, call once by the reading process. """
#     # shm = open_shared_memory_cache[name]
#     # shm = shared_memory.SharedMemory(name=name, create=False)
#     shm.close()
#     shm.unlink()


# noinspection DuplicatedCode
def arg_nearest(a:np.ndarray, v:np.ndarray, sorter:np.ndarray = None):
    """
    Returns the index positions into (a) where values of (v) are closest. O(|a| + |v| log |a|).
    (a) must be sorted or argsort indexes that sort (a) must be provided to sorter.

    Breaks ties randomly. Ties occur when duplicates in a exist.

    :param a: 1D array, sorted if sorter==None or unsorted with argsort indexes passed to sorter
    :param v: a scalar or 1D array of values to be looked up in a
    :param sorter: argsort results for a if a is not sorted already
    :return: an ndarray of indexes into a with the nearest match for each value in v
    """
    v = np.atleast_1d(v)

    # sorting is necessary to efficiently sample duplicates in (a) using left/right search
    # which requires index values to be contiguous after search left/right is run
    a = a[sorter] if sorter is not None else a

    ix_search_left = np.searchsorted(a, v, side='left')

    ix0 = np.maximum(0, np.minimum(len(a) - 1, ix_search_left - 1))
    ix1 = np.maximum(0, np.minimum(len(a) - 1, ix_search_left))

    diff0 = np.abs(v - a[ix0])
    diff1 = np.abs(v - a[ix1])

    ix_nearest = ix0 + (diff1 < diff0)

    # randomly sample duplicates in (a)
    low = np.searchsorted(a, a[ix_nearest], side='left')
    high = np.searchsorted(a, a[ix_nearest], side='right')
    ix_nearest_choice = np.random.randint(low=low, high=high, size=len(ix_nearest))

    return ix_nearest_choice if sorter is None else sorter[ix_nearest_choice]


def choice_wo_replace(a:np.ndarray, size=None):
    """
    wraps np.random.choice supporting size > sample size without replacement.
    when size > sample size the full set (a) is returned, permuted, as many times as possible and the
    remainder is sampled randomly without replacement.
    """
    a = np.array(a)
    repeat = [a[np.random.permutation(len(a))] for _ in range(size//a.shape[0])]
    remainder = np.random.default_rng().choice(a, size=size % a.shape[0], replace=False)
    b = np.concatenate(repeat + [remainder])
    return b


def search_file(filename, search_path):
   """
   Given a search path, find file
   Source: http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
   """
   file_found = 0
   paths = str.split(search_path, os.pathsep)
   path = None
   for path in paths:
      if exists(join(path, filename)):
          file_found = 1
          break
   if file_found:
      return abspath(os.path.join(path, filename))
   else:
      return None


def parse_include_channels(include_channels):
    """ Parser for include_channels parameter which can contain CSV and ranges, ex: 10,16:20 """
    if include_channels is None or include_channels == 'all':
        result = None
    elif isinstance(include_channels, str):
        ic = include_channels.split(',')
        result = []
        for c in ic:
            if ':' in c:
                result += list(range(*map(int, c.split(':'))))
            else:
                result += [int(c)]
    elif isinstance(include_channels, int):
        result = [include_channels]
    elif isinstance(include_channels, (list, tuple)):
        result = include_channels
    else:
        raise TypeError(
            'include_channels is of type {}, expected None, "all", int, list or tuple.'.format(type(include_channels)))

    return result


def validate_region_naming_convention(region: str, raise_exception: bool = True) -> bool:
    """
    Validates that the region name is following the standard naming convention defined here.
    :returns: is_valid, or raises an exception if raise_exception is True.
    """
    is_valid = region in VALID_BRAIN_REGION_NAMES
    if raise_exception:
        assert is_valid, f'Region {region} is not a valid region from the list: {VALID_BRAIN_REGION_NAMES}'
    return is_valid


def map2(func: Callable, args: Iterable[Union[Tuple, object]], fixed_values: dict = None, parallelism: int = 1):
    """
    A universal version of map combining functionality of:
    map, itertools.starmap, and pool.map

    Eliminates the need to understand functools, multiprocessing.Pool, and
    argument unpacking operators which should be unnecessary to accomplish a simple
    function mapping operation.

    Usage example:
        def f(x, y):
            print(x, y)

        common_py_utils.map2(
            func=f,
            args=[1, 2, 3],  # x arguments
            fixed_values=dict(y='yellow'),  # y always is 'yellow'
            parallelism=1,
        )

        common_py_utils.map2(
            func=f,
            args=[(1, 'yellow'), (2, 'yarn'), (3, 'yack')],  # (x, y) arguments
            parallelism=3,  # use a 3 process multiprocessing pool
        )

    :param func: a callable function
    :param args: a list of arguments (if only 1 argument is left after fixed_values) or a list of tuples (if multiple arguments are left after fixed_values)
    :param fixed_values: a dictionary with parameters that will stay the same for each call to func
    :param parallelism: number of processes to use. When parallelism==1 this maps to itertools.starmap and does not use multiprocessing. If parallelism >1 then multiprocessing.pool.starmap will be used with this number of worker processes. If parallelism == -1 (or non positive) then multiprocessing.pool will be used with multiprocessing.cpu_count() processes.
    :return:
    """
    assert len(args) > 0
    assert isinstance(parallelism, int)
    parallelism = parallelism if parallelism >= 1 else multiprocessing.cpu_count()

    func_partial = functools.partial(func, **(fixed_values or {}))
    args_list = list(args)
    args_tuples = args \
        if isinstance(args_list[0], tuple) \
           and len(args_list[0]) == (len(inspect.signature(func).parameters) - len(fixed_values or {})) \
        else [(a,) for a in args_list]

    if parallelism == 1:
        result_iterator = itertools.starmap(func_partial, args_tuples)
    else:
        with multiprocessing.Pool(parallelism) as pool:
            result_iterator = pool.starmap(func_partial, args_tuples)

    return list(result_iterator)
