import unittest
import inspect
from common_py_utils import yaml_cfg_parser
from common_py_utils.test_utils import BaseTestCase
from fs import tempfs


class TestCliUtils(BaseTestCase):
    def setUp(self) -> None:
        self.tempfs = tempfs.TempFS()

        self.test_yaml = inspect.cleandoc('''
            # global variables and a list of yaml files to be parsed and added to this configuration file dynamically
            DATASET_NAME: "SCF05"
            VERSION: "v8-standard-params"
            TEMPFS_PATH: {TEMPFS_PATH}
            INCLUDE: ["dataset-include.yaml"]
            
            # model hyperparameters passed to build_model
            hyperparams:
              sample_width: 65536
              include_channels: null
              batch_size: 3
              hp_max_predict_time_fps: 1800
              include_modules: "wnr,ma,pa,time"
            
            # training specific parameters passed to train function
            trainingparams:
              model: model_sleepstate
              training_steps: 15000
              checkpoint: "s3://hengenlab/checkpoints/checkpoints/{DATASET_NAME}/"
              checkpoint_final: "s3://hengenlab/{DATASET_NAME}/Model/{VERSION}/"
              disable_comet: false
              testeval_on_checkpoint: true
              n_workers: 12
            ''')

        self.include_yaml = inspect.cleandoc('''
            GLOBAL_VAR_TEST: abc
            datasetparams:
              corrupt_neural_files: Headstages_64_Channels_int16_2018-12-05_16-17-34_CORRECTED.bin
              neural_bin_files_per_sleepstate_file: -1
              test_video_files: e3v8102-20181205T1918-2018.mp4
              global_var: "{GLOBAL_VAR_TEST}"
            
              neural_files_basepath: s3://hengenlab/SCF05/Neural_Data/
              video_files_basepath: s3://hengenlab/SCF05/Video/
              sleepstate_files_basepath: s3://hengenlab/SCF05/SleepState/
              syncpulse_files_zip: s3://hengenlab/SCF05/SyncPulse.zip
            
              labels_file: s3://hengenlab/SCF05/Labels/labels_sleepstate_v2_SCF05.npz
            ''')

        self.tempfs.writetext('base-config.yaml', self.test_yaml)
        self.tempfs.writetext('dataset-include.yaml', self.include_yaml)

    def tearDown(self) -> None:
        self.tempfs.close()

    def test_parse_cli_yaml(self):
        yaml = yaml_cfg_parser.parse_yaml_cfg(
            self.tempfs.getsyspath('base-config.yaml'),
            includes='{trainingparams: {training_steps: 30000}}',
        )

        self.assertEqual('model_sleepstate', yaml['trainingparams']['model'])
        self.assertEqual(30000, yaml['trainingparams']['training_steps'])
        self.assertEqual('SCF05', yaml['DATASET_NAME'])
        self.assertEqual('abc', yaml['datasetparams']['global_var'])
        self.assertEqual('e3v8102-20181205T1918-2018.mp4', yaml['datasetparams']['test_video_files'])
        self.assertTrue('INCLUDE' not in yaml)
        self.assertEqual(True, yaml['trainingparams']['testeval_on_checkpoint'])

    def test_enhanced_formatter_class(self):
        self.assertEqual('abcd', yaml_cfg_parser.EnhancedFormatter().format('{!l}', 'aBcD'))
        self.assertEqual('EFGH', yaml_cfg_parser.EnhancedFormatter().format('{0!u}', 'eFgH'))
        self.assertEqual('ijkl', yaml_cfg_parser.EnhancedFormatter().format('{0!l}', 'iJkL'))
        self.assertEqual('MNOP', yaml_cfg_parser.EnhancedFormatter().format('{kwarg!u}', kwarg='mNoP'))
        self.assertEqual('{a}42', yaml_cfg_parser.EnhancedFormatter().format('{a}{b}', b=42))

    def test_recursive_str_format(self):
        test_cases = [
            ('{a}', {'a': 42}),
            ('{a[b]}', {'a': {'b': 42}}),
            ('abc{a}ghi', {'a': 'def'}),
        ]

        expected_results = [
            42,
            42,
            'abcdefghi',
        ]

        for test_case, expected_result in zip(test_cases, expected_results):
            result = yaml_cfg_parser.recursive_str_format(test_case[0], test_case[1])
            self.assertEqual(result, expected_result)
