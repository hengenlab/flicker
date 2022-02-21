""" Compute the summary statistics from splice-v4 experiment. """
import numpy as np
import itertools
import logging
import sys
import glob


#
# Basic assumptions for this script
#   - All CSVs in the working directory
#   - File naming convention: predictions-$CLASS-to-$CLASS-len$SAMPLE_LENGTHS-iter[001..100].csv
#

SAMPLE_LENGTHS = [1, 2, 3, 5, 7, 14, 35, 70, 105]  # [1, 2, 3, 5, 7, 10, 15, 30]
STATES = ['wake', 'nrem', 'rem']
CLASS_MAP = {'wake': 0, 'nrem': 1, 'rem': 2}

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()

output_file = open('splice_summary_stats.csv', 'w')
output_file.write('splice_type,length,accuracy_spliced_region,accuracy_surrounding_region,count_captured_splice_regions,total_splice_regions\n')

for (state_from, state_to), sample_length in itertools.product(itertools.permutations(STATES, 2), SAMPLE_LENGTHS):
    # logger.info(f'Processing {state_from}-to-{state_to} for length {sample_length:03}')
    files = sorted(glob.glob(f'predictions-{state_from}-to-{state_to}-len{sample_length:03}-iter*.csv'))
    # logger.debug(files)

    results_spliced_region = []
    results_surrounding_region = []
    results_count_captured_splice_regions = []
    results_total_splice_regions = 0

    # Gather all statistics from the ~100 files
    for file in files:
        records = np.recfromcsv(file)
        predictions_spliced_region = records[records['label_wnr_012'] == CLASS_MAP[state_from]]
        predictions_surrounding_region = records[records['label_wnr_012'] == CLASS_MAP[state_to]]

        accuracy_spliced_region = np.mean(
            predictions_spliced_region['label_wnr_012'] == predictions_spliced_region['predicted_wnr_012']
        )
        accuracy_surrounding_region = np.mean(
            predictions_surrounding_region['label_wnr_012'] == predictions_surrounding_region['predicted_wnr_012']
        )
        is_splice_region_captured = np.any(
            predictions_spliced_region['label_wnr_012'] == predictions_spliced_region['predicted_wnr_012']
        )
        if np.isnan(accuracy_spliced_region) or np.isnan(accuracy_surrounding_region):
            logger.warning(f'NaN encountered in file {file}')

        # logger.debug(f'accuracy_spliced_region {accuracy_spliced_region}, '
        #              f'accuracy_surrounding_region {accuracy_surrounding_region}')
        results_spliced_region.append(accuracy_spliced_region)
        results_surrounding_region.append(accuracy_surrounding_region)
        results_count_captured_splice_regions.append(is_splice_region_captured)
        results_total_splice_regions += 1

    logger.info(f'predictions-{state_from}-to-{state_to}-len{sample_length:03} - mean accuracy '
                f'spiced region {np.mean(results_spliced_region):0.4f}, mean accuracy '
                f'surrounding region {np.mean(results_surrounding_region):0.4f},'
                f'num splice regions captured {np.sum(results_count_captured_splice_regions)},'
                f'total splice regions {results_total_splice_regions}')

    output_file.write(f'{state_from}-to-{state_to},{sample_length},'
                      f'{np.mean(results_spliced_region):0.6f},'
                      f'{np.mean(results_surrounding_region):0.6f},'
                      f'{np.sum(results_count_captured_splice_regions)},'
                      f'{results_total_splice_regions}\n')
