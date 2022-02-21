import unittest
import dataset
from fs.tempfs import TempFS
import numpy as np
import tensorflow as tf
import model_sleepstate
import io
import math
from common_py_utils.BaseTestCase import BaseTestCase
import textwrap


class ModelSleepStateTest(unittest.TestCase):
    def test_channel_map(self):
        """ Tests that dataset channels map to expected values from a live dataset pull. """
        manually_generated_sample = \
            b'\xb9\xb8a\r\xa6\x00\x00\x00\xa0\xfc\x04\xfc\xf0\xfc\xac\xfcx\xfc\x9c\xfd\xb4\xfc\x94\xfc\xac\xfa\x18\xfcL\xfe\xbc\xfa\xa4\xfa\xe8\xfa\xbc\xff\xb0\xfb\xe0' \
            b'\xfc\x98\xfb\x8c\xfa\xb8\x01l\xfd\xa4\xfc\xec\xfb\xd8\x00\x10\xfbl\xfb\xb4\xfb\x1c\x02,\xfd,\xf9\xd0\xfa\xd4\x00 \xfc\xb8\x03\x80\xfb\x94\xfb\x1c\x02T\x01' \
            b'\xb8\xfc<\xfbD\x04\xd0\xfa\x04\xfb\x88\xfd\x10\xfb \x05\xf4\xfe\xb0\xfb\xe4\xfa\x04\xfd\x90\xfc\xb0\x02`\xfb\xd4\x02\x08\xfd(\xfc\x1c\xfe\xd4\xfa\\\xfc4' \
            b'\xfb4\xfc\xe8\xfb\x00\xfb\x98\xfa\xf8\xfa\x14\xf8\x80\xfa\xc0\xfep\xf8\x00\x04T\xfeD\xfbx\xfbt\xfb\xfc\x02l\xfe\xb0\xfc\x00\xfd\x90\xffH\xfb\xe8\x00\x88\xff\xd8\xfa\xb4\xfc\xb0\x01\x04\xf8\xf0\xfe\xec\xfd\xd0\x00x\xfd\x18\xfa\xec\x00' \
            b'\x1c\x08(\xfc@\xfd\xd0\xfe\xac\xfep\x03\xc0\xf4\xc0\xfe|\xfb\xc8\x040\xfb\x90\xfc\x10\xfa\xec\xff\xa8\x06l\xfe\x1c\xfcd\x01\x08\xfc\xb4\xfe,\x030\x05D\xfbl\x03\xe4\xfa\x80\x07\x90\x04`\xff\xd0\x01x\x06t\xff\x14\xfe8\xff\xac\xfb\xc8' \
            b'\xfe$\xfb\xe8\xfd4\xfd,\xff\xac\xfe\xd0\xfd\xc0\x01|\xfe<\x02\xa8\xfc\xd4\xfc\xa4\x00@\x00\x0c\x00,\xfb\xb8\x01\xc4\xffT\x00\x80\xfc\x8c\xff\x94\xfe\xc4\xfe' \
            b'\xdc\xffd\xfe\xa0\xfd\xd0\xff\xc0\xffP\x02\x9c\xfe\x04\x004\x01\xb0\x00\x94\xfe\xb0\xfex\xff\xf8\x00X\xfe\x90\xffH\xfc\xd8\xff\x0c\xff4\xfeL\xfc\xb8\xff' \
            b'\xf8\xff\x80\xfe`\xfdh\xffh\xfb<\xfdl\xfe\xa4\xfcP\x00\xb0\xfb\xc0\xfd\x08\xfet\xfc\x84\x00\x9c\xff\x98\xfd\x84\xfe\xac\xfd8\xfdD\x00x\xfe\x8c\xfb\xd0' \
            b'\xfc\xbc\xfa\xa8\xfc\x18\xfd\xc8\xfd\xac\xfeH\xff\x9c\xf9\x90\xfc\xc0\xffT\xfc\x0c\xfbp\xfc,\xff\x1c\xfc\\\xff\xc8\xfd\xa0\xfd\xe4\xfdd\xfd4\xfc\x8c' \
            b'\xfb\\\xfd\x8c\x00D\x00\xc0\xfeP\xfe\xe0\xfdl\xfd`\xfd \xfcp\xfe\x10\xfdX\xff\xe4\xfe\xf4\xfe\x18\xff\xcc\xfc\x94\xfd@\xfc\xd0\xfd\xdc\xfd\x8c\xfc\xbc' \
            b'\xfdd\xff\xf8\xfc8\xfft\xfe\xc8\xfc\xd8\xfeL\xfe\x94\xfat\xfe(\xff\xac\xfb\x10\xfd\x94\xfd$\xfd\xdc\xff\x88\xfe\x84\xfet\xff`\xfe'

        s1_0_63 = np.array([
             -864, -1020,  -784,  -852,  -904,  -612,  -844,  -876, -1364,
            -1000,  -436, -1348, -1372, -1304,   -68, -1104,  -800, -1128,
            -1396,   440,  -660,  -860, -1044,   216, -1264, -1172, -1100,
              540,  -724, -1748, -1328,   212,  -992,   952, -1152, -1132,
              540,   340,  -840, -1220,  1092, -1328, -1276,  -632, -1264,
             1312,  -268, -1104, -1308,  -764,  -880,   688, -1184,   724,
             -760,  -984,  -484, -1324,  -932, -1228,  -972, -1048, -1280, -1384
        ], dtype=np.int16)
        m1_64_127 = np.array([
            -1288, -2028, -1408,  -320, -1936,  1024,  -428, -1212, -1160,
            -1164,   764,  -404,  -848,  -768,  -112, -1208,   232,  -120,
            -1320,  -844,   432, -2044,  -272,  -532,   208,  -648, -1512,
              236,  2076,  -984,  -704,  -304,  -340,   880, -2880,  -320,
            -1156,  1224, -1232,  -880, -1520,   -20,  1704,  -404,  -996,
              356, -1016,  -332,   812,  1328, -1212,   876, -1308,  1920,
             1168,  -160,   464,  1656,  -140,  -492,  -200, -1108,  -312, -1244
        ], dtype=np.int16)
        hipp_128_191 = np.array([
            -536,  -716,  -212,  -340,  -560,   448,  -388,   572,  -856,
            -812,   164,    64,    12, -1236,   440,   -60,    84,  -896,
            -116,  -364,  -316,   -36,  -412,  -608,   -48,   -64,   592,
            -356,     4,   308,   176,  -364,  -336,  -136,   248,  -424,
            -112,  -952,   -40,  -244,  -460,  -948,   -72,    -8,  -384,
            -672,  -152, -1176,  -708,  -404,  -860,    80, -1104,  -576,
            -504,  -908,   132,  -100,  -616,  -380,  -596,  -712,    68, -392
        ], dtype=np.int16)
        m2_192_255 = np.array([
            -1140,  -816, -1348,  -856,  -744,  -568,  -340,  -184, -1636,
             -880,   -64,  -940, -1268,  -912,  -212,  -996,  -164,  -568,
             -608,  -540,  -668,  -972, -1140,  -676,   140,    68,  -320,
             -432,  -544,  -660,  -672,  -992,  -400,  -752,  -168,  -284,
             -268,  -232,  -820,  -620,  -960,  -560,  -548,  -884,  -580,
             -156,  -776,  -200,  -396,  -824,  -296,  -436, -1388,  -396,
             -216, -1108,  -752,  -620,  -732,   -36,  -376,  -380,  -140, -416
        ], dtype=np.int16)

        labels_dtype = [
            ('activity', 'i1'),
            ('sleep_state', 'i1'),
            ('video_filename_ix', '<i4'),
            ('video_frame_offset', '<u4'),
            ('neural_filename_ix', '<i4'),
            ('neural_offset', '<u4')
        ]
        labels_matrix = np.empty(shape=(1,), dtype=labels_dtype)

        labels_matrix['activity'] = -1
        labels_matrix['sleep_state'] = 1
        labels_matrix['video_filename_ix'] = 0
        labels_matrix['video_frame_offset'] = 0
        labels_matrix['neural_filename_ix'] = 0
        labels_matrix['neural_offset'] = 1

        with TempFS() as tfs:
            # Write files to temp filesystem
            np.savez(tfs.getsyspath('/labels_file.npz'), labels_matrix=labels_matrix, neural_files=np.array(['neural_file_0.bin']),
                     video_files=np.array(['video_file_0.bin']))
            with tfs.openbin('neural_file_0.bin', mode='w+') as f:
                f.write(manually_generated_sample)

            dssl = dataset.DatasetSleepStateLabels(labels_file=tfs.getsyspath('/labels_file.npz'),
                                                   sample_width=1,
                                                   n_channels=256,
                                                   neural_files_basepath=tfs.getsyspath('/'),
                                                   shuffle=False)
            dnl = dataset.DatasetNeuralLoader(neural_files_basepath=tfs.getsyspath('/'), n_channels=256)
            ds = dssl.as_dataset()
            ds = dnl.as_dataset(ds)
            get_next = tf.data.make_one_shot_iterator(ds).get_next()

            with tf.Session() as sess:
                sample = sess.run(get_next)

                self.assertTrue(np.all(sample['neural_data'][0:64].reshape(64,) == s1_0_63))
                self.assertTrue(np.all(sample['neural_data'][64:128].reshape(64,) == m1_64_127))
                self.assertTrue(np.all(sample['neural_data'][128:192].reshape(64,) == hipp_128_191))
                self.assertTrue(np.all(sample['neural_data'][192:256].reshape(64,) == m2_192_255))

                with self.assertRaises(tf.errors.OutOfRangeError):
                    sess.run(get_next)

    def test_parse_channels_list(self):
        self.assertEqual(model_sleepstate.parse_channels_list(channels=None), [])
        self.assertEqual(model_sleepstate.parse_channels_list(channels=''), [])
        self.assertEqual(model_sleepstate.parse_channels_list(channels='2:4'), [2, 3])
        self.assertEqual(model_sleepstate.parse_channels_list(channels='0,1,3:4,5'), [0, 1, 3, 5])
        self.assertEqual(model_sleepstate.parse_channels_list(channels='2:5,7:9'), [2, 3, 4, 7, 8])
        self.assertEqual(model_sleepstate.parse_channels_list(channels='0,1,2,3'), [0, 1, 2, 3])


class TestScriptsModelSleepstateSummaryStatistics(BaseTestCase):
    def setUp(self) -> None:
        self.test_data_csv = \
            '''
            label,label_text,predicted_wnr_012,predicted_pa_01,predicted_ma_01,probability_wake,probability_nrem,probability_rem,probability_passive_active,probability_micro_arousal,video_filename,video_frame_ix,neural_filename,neural_offset
            1,Wake_Active,0,1,0,0.8439059257507324,0.019780347123742104,0.13631372153759003,0.00022189796436578035,3.490906408387673e-07,e3v8102-20181205T1712-1818.mp4,3750,Headstages_64_Channels_int16_2018-12-05_16-22-35.bin,67604
            1,Wake_Active,1,0,0,0.019780347123742104,0.8439059257507324,0.13631372153759003,0.00022189796436578035,3.490906408387673e-07,e3v8102-20181205T1712-1818.mp4,3750,Headstages_64_Channels_int16_2018-12-05_16-22-35.bin,67604
            2,NREM,1,0,0,0.004169863648712635,0.9261558055877686,0.06967438757419586,0.00011635545524768531,2.798273044390953e-07,e3v8102-20181205T1712-1818.mp4,3751,Headstages_64_Channels_int16_2018-12-05_16-22-35.bin,69270
            2,NREM,1,0,0,0.019284946843981743,0.9168368577957153,0.06387823075056076,6.970912363613024e-05,6.232838245523453e-08,e3v8102-20181205T1712-1818.mp4,3752,Headstages_64_Channels_int16_2018-12-05_16-22-35.bin,70937
            2,NREM,2,0,1,0.019284946843981743,0.06387823075056076,0.9168368577957153,6.970912363613024e-05,6.232838245523453e-08,e3v8102-20181205T1712-1818.mp4,3752,Headstages_64_Channels_int16_2018-12-05_16-22-35.bin,70937
            3,REM,2,0,1,0.01955333910882473,0.18342554569244385,0.7970210909843445,1.2947730283485726e-05,4.667812802949811e-09,e3v8102-20181205T1612-1718.mp4,3753,Headstages_64_Channels_int16_2018-12-05_16-22-35.bin,72604
            3,REM,1,0,0,0.3512805104255676,0.4305083155632019,0.21821115911006927,2.381421381869586e-06,2.8492691539483417e-10,e3v8102-20181205T1612-1718.mp4,3754,Headstages_64_Channels_int16_2018-12-05_16-22-35.bin,74270
            4,Micro_Arousal,1,0,1,0.10892058908939362,0.5688973665237427,0.3221819996833801,1.5229247765091714e-05,3.552010907625913e-10,e3v8102-20181205T1612-1718.mp4,3755,Headstages_64_Channels_int16_2018-12-05_16-22-35.bin,75937
            5,Wake_Passive,1,0,1,0.007136879954487085,0.9817348122596741,0.011128385551273823,2.1920226572547108e-05,1.1891626572335667e-09,e3v8102-20181205T1612-1718.mp4,3756,Headstages_64_Channels_int16_2018-12-05_16-22-35.bin,77603
            '''
        self.test_file = io.StringIO(self.test_data_csv)
        self.tfs = TempFS()

    def test_main_no_testset(self):

        expected_result = textwrap.dedent(
            """
            per-label-test-set
                NaN: WAKE            (accuracy)
                NaN: NREM            (accuracy)
                NaN: REM             (accuracy)
                NaN: Passive/Active  (accuracy)
                NaN: Micro-arousal   (F1 score)
            
            per-label-train-set
               0.33: WAKE            (accuracy)
               0.67: NREM            (accuracy)
               0.50: REM             (accuracy)
               0.67: Passive/Active  (accuracy)
               0.40: Micro-arousal   (F1 score)
            
            per-video-metrics
                                               Passive/   Micro-
                WAKE   |   NREM   |   REM    |  Active  |  arousal | video-filename
              -------- | -------- | -------- | -------- | -------- | ---------------
                0.00   |    NaN   |   0.50   |   1.00   |   0.50   | e3v8102-20181205T1612-1718.mp4
                0.50   |   0.67   |    NaN   |   0.50   |   0.00   | e3v8102-20181205T1712-1818.mp4
            """
        ).strip() + '\n'

        model_sleepstate.generate_summary_statistics(
            predictions_filename=self.test_file, output_filename=self.tfs.getsyspath('/unittest'), display=True,
            hp_max_predict_time_fps=450,
        )

        # self.assert_complex_equal(expected_result, result)
        result = self.tfs.readtext('/unittest')
        self.assertEqual(expected_result, result)

    def test_main_with_testset(self):

        expected_result = textwrap.dedent(
            """
            per-label-test-set
               0.00: WAKE            (accuracy)
                NaN: NREM            (accuracy)
               0.50: REM             (accuracy)
               1.00: Passive/Active  (accuracy)
               0.50: Micro-arousal   (F1 score)
            
            per-label-train-set
               0.50: WAKE            (accuracy)
               0.67: NREM            (accuracy)
                NaN: REM             (accuracy)
               0.50: Passive/Active  (accuracy)
               0.00: Micro-arousal   (F1 score)
            
            per-video-metrics
                                               Passive/   Micro-
                WAKE   |   NREM   |   REM    |  Active  |  arousal | video-filename
              -------- | -------- | -------- | -------- | -------- | ---------------
                0.00   |    NaN   |   0.50   |   1.00   |   0.50   | e3v8102-20181205T1612-1718.mp4 (test-set)
                0.50   |   0.67   |    NaN   |   0.50   |   0.00   | e3v8102-20181205T1712-1818.mp4
            """
        ).strip() + '\n'

        model_sleepstate.generate_summary_statistics(
            predictions_filename=self.test_file, output_filename=self.tfs.getsyspath('/unittest'), display=True,
            test_video_files='e3v8102-20181205T1612-1718.mp4', hp_max_predict_time_fps=450
        )

        # self.assert_complex_equal(expected_result, result)
        result = self.tfs.readtext('/unittest')
        self.assertEqual(expected_result, result)


class TestLossFunctions(BaseTestCase):
    def test_build_loss_wnr(self):
        model = model_sleepstate.ModelSleepState()

        output_wnr = tf.ones(shape=(2, 3), dtype=tf.float32)
        sleep_state = tf.constant([1, 2], dtype=tf.int8)

        loss_wnr, ops = model.build_loss_wnr(output_wnr, sleep_state)

        self.assertEqual(loss_wnr.sz, ())
        self.assertTrue('labels_wnr' in vars(ops))
        self.assertTrue('loss_wnr' in vars(ops))

    def test_build_loss_ma(self):
        model = model_sleepstate.ModelSleepState()

        output_ma = tf.ones(shape=(2, 1), dtype=tf.float32)
        sleep_state = tf.constant([1, 2], dtype=tf.int8)

        loss_ma, ops = model.build_loss_ma(output_ma, sleep_state)

        self.assertEqual(loss_ma.sz, ())
        self.assertTrue('labels_ma' in vars(ops))
        self.assertTrue('loss_ma' in vars(ops))

    def test_build_loss_pa(self):
        model = model_sleepstate.ModelSleepState()

        output_pa = tf.ones(shape=(2, 1), dtype=tf.float32)
        sleep_state = tf.constant([1, 2], dtype=tf.int8)

        loss_pa, ops = model.build_loss_pa(output_pa, sleep_state)

        self.assertEqual(loss_pa.sz, ())
        self.assertTrue('labels_pa' in vars(ops))
        self.assertTrue('loss_pa' in vars(ops))

    def test_build_loss_time(self):
        model = model_sleepstate.ModelSleepState()

        output_time = tf.ones(shape=(2, 6), dtype=tf.float32)
        t = tf.ones(shape=(2,), dtype=tf.float64) * 50

        loss_time, ops = model.build_loss_time(output_time, t, t, t, t, t, t, 100)
        self.assertEqual(loss_time.sz, ())
        self.assertTrue('labels_time' in vars(ops))
        self.assertTrue('loss_time' in vars(ops))
