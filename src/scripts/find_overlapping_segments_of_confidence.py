"""
deprecated

Finds overlapping segments of confidence from multi-region models.
Accompanies AWK shell command in submit_splice_experiment.sh

This is an alpha version of this script, not fully automated.
"""
import numpy as np
import glob
import yaml
import plotly.graph_objects as go
import plotly.subplots
import csv
import smart_open
import functools


SEGMENT_LENGTH = 45  # Minimum segment length (1/15th sec, a single frame)
N_PLOTS = 500          # Number of plots (ordered by most overlap of confident regions)


def main():
    # hard coded glob for now
    segments_of_confidence_glob = '../../results/EAB50_5state/per-region-timing-comparison/segments_of_confidence*.csv'
    segments_of_confidence_filenames = glob.glob(segments_of_confidence_glob)
    print('Processing files:\n', yaml.dump(segments_of_confidence_filenames))

    segments_of_confidence_npy = [np.genfromtxt(s, delimiter=',', dtype=np.int32) for s in segments_of_confidence_filenames]
    # n = sum([s.shape[0] for s in segments_of_confidence_npy])
    # print(n, '\n', segments_of_confidence_npy[0], '\n', segments_of_confidence_npy[1])

    # Search for segments common across all/some datasets, Sort by number of regions the segment occurs in, then by length
    # Brute force search because we don't have so much data to worry about a more efficient algorithm
    matched_regions = [[] for _ in range(len(segments_of_confidence_npy))]
    for i, segments_of_confidence_region in enumerate(segments_of_confidence_npy):
        for j, segment_source in enumerate(segments_of_confidence_region):
            for k, region in enumerate(segments_of_confidence_npy[i+1:]):
                # print(segment_source, '\n', region)
                overlap_mask = (region[:, 1] < segment_source[2]) & (region[:, 2] > segment_source[1])
                overlap_lengths = np.minimum(segment_source[2], region[overlap_mask, 2]) - \
                                  np.maximum(segment_source[1], region[overlap_mask, 1])
                overlap_accepted_regions = region[overlap_mask, :][overlap_lengths > SEGMENT_LENGTH]
                overlap_accepted_regions_list = [row.tolist() + [0] for row in overlap_accepted_regions]
                # print(overlap_accepted_regions_list)
                matched_regions[i+k+1].extend(overlap_accepted_regions_list)
                if len(overlap_accepted_regions_list) > 0:
                    matched_regions[i].append(segment_source.tolist() + [0])

    # merge matched_regions to get a count of how many times each region matched
    consolidated_regions = [[] for _ in range(len(segments_of_confidence_npy))]
    for i, region in enumerate(matched_regions):
        last_segment = None
        for segment in region:
            if last_segment is None:
                last_segment = segment
                consolidated_regions[i].append(segment)
            else:
                if segment[:-1] == last_segment[:-1]:
                    consolidated_regions[i][-1][4] += 1
                else:
                    last_segment = segment
                    consolidated_regions[i].append(segment)

    # print(consolidated_regions)
    flattened_regions = sorted([s for segment in consolidated_regions for s in segment], key=lambda e: e[4], reverse=True)

    #
    # Generate and save plots
    #
    lfp_data_map = {}
    region_names = ['CP', 'CP', 'M1/M2', 'CA1', 'CA1', 'S1', 'SC', 'V1/V2']
    neural_basepath = 's3://hengenlab/EAB50_5state/Neural_Data'
    smart_open.open = functools.partial(smart_open.open, transport_params={'resource_kwargs':{'endpoint_url': 'https://s3.nautilus.optiputer.net'}})
    local_basepath = '../../results/EAB50_5state/per-region-timing-comparison'
    prediction_files = [
        'predictions_EAB50_5state_CPA.csv',
        'predictions_EAB50_5state_CPB.csv',
        'predictions_EAB50_5state_M1M2.csv',
        'predictions_EAB50_5state_CA1_192.csv',
        'predictions_EAB50_5state_CA1_256.csv',
        'predictions_EAB50_5state_S1.csv',
        'predictions_EAB50_5state_SC.csv',
        'predictions_EAB50_5state_V1V2.csv',
    ]
    prediction_files_loaded = [np.genfromtxt(f'{local_basepath}/{prediction_file}', delimiter=',', skip_header=1, dtype=np.float32) for
                               prediction_file in prediction_files]
    segment_size = 80  # number of data points to include in each bitmap

    for _, segment_start, _, _, _ in flattened_regions[:N_PLOTS]:
        if segment_start not in lfp_data_map:
            with open(f'{local_basepath}/predictions_EAB50_5state_CPA.csv') as f:
                reader = csv.reader(f)
                csv_row = [row for idx, row in enumerate(reader) if idx == segment_start]
                neural_filename = csv_row[0][10]
                neural_data_offset = csv_row[0][11]
                # print(csv_row)
                # print(neural_filename)
                # print(neural_data_offset)

            with smart_open.open(f'{neural_basepath}/{neural_filename}', 'rb') as f:
                byte_offset = 8 + int(neural_data_offset) * 512 * 2
                f.seek(byte_offset)
                # print(512 * 2 * segment_size * int(1 / 15 * 25000))
                lfp_bytes = f.read(512 * 2 * segment_size * int(1 / 15 * 25000))
                lfp_data = np.frombuffer(lfp_bytes, dtype=np.int16).reshape((512, -1), order='F')
                # print(lfp_data.shape)
                # print(lfp_data)

            if lfp_data.shape[1] < segment_size * int(1 / 15 * 25000):
                print(f'Skipping file for segment_start {segment_start}')
                del lfp_data_map[segment_start]
                continue

            lfp_data_map[segment_start] = lfp_data

        #
        # Plotly code
        #

        # ------------------- BLIPS -------------------
        fig = plotly.subplots.make_subplots(rows=len(prediction_files), cols=1, shared_xaxes=True,
                                            vertical_spacing=0.013, subplot_titles=('', '', '', '', '', '', '', ''))
        fig.update_layout(
            title=f'Per brain region models EAB50_5state, segment {segment_start}, displaying {segment_size / 15:.1f}s',
            font=dict(color='white'))
        fig.update_layout(width=1800, height=700)
        fig.update_layout(paper_bgcolor='#212121')
        fig.update_layout(legend=dict(font=dict(color='white')))

        for i, predictions in enumerate(prediction_files_loaded):
            segment = predictions[segment_start:segment_start + segment_size, 3:6]

            fig.add_trace(go.Scatter(y=np.maximum(0.0, segment[:, 0]), stackgroup='one', name='WAKE', mode='lines',
                                     marker=dict(color='yellow'), showlegend=(i == 0)), row=i + 1, col=1)
            fig.add_trace(go.Scatter(y=np.maximum(0.0, segment[:, 1]), stackgroup='one', name='NREM', mode='lines',
                                     marker=dict(color='blue'), showlegend=(i == 0)), row=i + 1, col=1)
            fig.add_trace(go.Scatter(y=np.maximum(0.0, segment[:, 2]), stackgroup='one', name='REM', mode='lines',
                                     marker=dict(color='red'), showlegend=(i == 0)), row=i + 1, col=1)

            fig.update_xaxes(color='white', tickmode='array', row=i + 1, col=1)
            fig.update_yaxes(title=dict(text=region_names[i], font=dict(size=10)), color='white', range=[0, 1],
                             row=i + 1, col=1)

        fig.update_xaxes(title=dict(text='Time', font=dict(size=15)), tickvals=list(range(0, segment_size, 5)),
                         ticktext=[f'{i / 15:0.1f}s' for i in range(0, segment_size, 5)], row=len(prediction_files),
                         col=1)
        fig.add_annotation(text='Confidence', font=dict(color='white', size=15), textangle=-90, xref='paper',
                           yref='paper', x=-0.04, y=0.5)
        fig.update_layout(margin=dict(l=100))
        fig.write_image(f'{local_basepath}/img/prob_{segment_start}.png')

        # -------------------- LFP --------------------
        fig = plotly.subplots.make_subplots(rows=len(prediction_files), cols=1, shared_xaxes=True,
                                            vertical_spacing=0.013, subplot_titles=('', '', '', '', '', '', '', ''))
        fig.update_layout(
            title=f'Per brain region models EAB50_5state, segment {segment_start}, displaying {segment_size / 15:.1f}s',
            font=dict(color='white'))
        fig.update_layout(width=1800, height=700)
        fig.update_layout(paper_bgcolor='#212121')
        fig.update_layout(legend=dict(font=dict(color='white')))

        for i, prediction_file in enumerate(prediction_files):
            lfp_data = lfp_data_map[segment_start][32 + i * 64]
            print(f'i {i}, lfp_data shape:  {lfp_data_map[segment_start].shape} for segment_start {segment_start}')

            fig.add_trace(
                go.Scatter(y=lfp_data, name='LFP', mode='lines', marker=dict(color='blue'), showlegend=(i == 0)),
                row=i + 1, col=1)
            fig.update_xaxes(color='white', tickmode='array', row=i + 1, col=1)
            fig.update_yaxes(title=dict(text=region_names[i], font=dict(size=10)), color='white', row=i + 1, col=1)

            # draw prediction highlights
            start = 0
            last = None
            count = 0
            predictions = prediction_files_loaded[i][segment_start:segment_start + segment_size, 1].copy().astype(
                np.int32)
            colors = {0: 'yellow', 1: 'blue', 2: 'red'}
            for j in range(len(predictions) + 1):
                if j < len(predictions) and (predictions[j] == last or last is None):
                    count += 1
                    last = predictions[j]
                else:
                    fig.add_vrect(x0=start * int(1 / 15 * 25000), x1=(start + count) * int(1 / 15 * 25000),
                                  layer='above', line_width=0, opacity=0.5, fillcolor=colors[last], row=i + 1, col=1)
                    start = j
                    count = 1
                    last = predictions[j] if j < len(predictions) else None

        fig.update_xaxes(title=dict(text='Time', font=dict(size=15)),
                         tickvals=list(range(0, segment_size * int(1 / 15 * 25000), 5 * int(1 / 15 * 25000))),
                         ticktext=[f'{i / 15:0.1f}s' for i in range(0, segment_size, 5)], row=len(prediction_files),
                         col=1)
        fig.add_annotation(text='Confidence', font=dict(color='white', size=15), textangle=-90, xref='paper',
                           yref='paper', x=-0.045, y=0.5)
        fig.update_layout(margin=dict(l=100))
        print(f'Completed image {local_basepath}/img/lfp_{segment_start}.png')
        fig.write_image(f'{local_basepath}/img/lfp_{segment_start}.png')


if __name__ == '__main__':
    main()
