import argparse
import cv2
import numpy as np
from common_py_utils import common_utils
import multiprocessing
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import pandas as pd
import pickle


def arg_parser():
    parser = argparse.ArgumentParser(
        description='Generates a measure of movement in an image using optical flow.'
    )
    parser.add_argument(
        '--parallelism', type=int, required=False, default=-1,
        help='Number of parallel processes, default is -1 to use multiprocessing.cpu_count().'
    )
    parser.add_argument(
        '--img_count', type=int, required=True,
        help='Number of images to process.'
    )
    parser.add_argument(
        '--input_path', type=str, required=False, default='/tmp/video_frames'
    )
    parser.add_argument(
        '--output_path', type=str, required=False, default='/tmp/optical_flow_frames'
    )
    parser.add_argument(
        '--tmp_dir', type=str, required=False, default='/tmp/optical_flow_tmp'
    )

    return vars(parser.parse_args())


def main(parallelism: int, img_count: int,
         input_path: str, output_path: str, tmp_dir: str):

    top_percent_flow_mean = common_utils.map2(
        compute_optical_flow,
        args=range(0, img_count - 1),
        fixed_values=dict(input_path=input_path, tmp_dir=tmp_dir),
        parallelism=parallelism,
    )

    top_percent_flow_mean = np.array(top_percent_flow_mean)

    range_upper = np.max(top_percent_flow_mean)
    range_lower = np.min(top_percent_flow_mean)
    normalized_flow = (np.array(top_percent_flow_mean) - range_lower) / (range_upper - range_lower)
    windowed_mean_flow = pd.Series(normalized_flow).rolling(window=30, center=True, min_periods=1).mean()
    windowed_mean_flow = (windowed_mean_flow - windowed_mean_flow.min()) / (windowed_mean_flow.max() - windowed_mean_flow.min())

    # there are problems with the CSV that should have been corrected before the run.
    windowed_mean_flow[np.isnan(windowed_mean_flow)] = 0.0
    pd.DataFrame({
        f'frame_index': np.arange(windowed_mean_flow.shape[0]),
        f'normalized_flow': windowed_mean_flow,
    }).to_csv(f'{output_path}/results.csv', index=False)

    # Create image overlays
    common_utils.map2(
        func=image_overlay,
        args=[(i, mf) for i, mf in enumerate(windowed_mean_flow)],
        fixed_values=dict(input_path=input_path, output_path=output_path, tmp_dir=tmp_dir),
        parallelism=parallelism,
    )


def compute_optical_flow(i: int, input_path: str, tmp_dir: str):
    # load images
    frame1 = f'{input_path}/{i:09d}.png'
    frame2 = f'{input_path}/{i + 1:09d}.png'
    img1 = cv2.imread(frame1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(frame2, cv2.IMREAD_GRAYSCALE)
    assert img1 is not None and img2 is not None, f'{frame1}:{img1} - {frame2}{img2}'

    # crop top 46 pixels of images
    img1_cropped = img1[47:, :]
    img2_cropped = img2[47:, :]

    # compute optical flow
    flow = cv2.calcOpticalFlowFarneback(img1_cropped, img2_cropped, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.linalg.norm(flow, axis=2)

    # compute statistics
    a = int(np.product(magnitude.shape) * 0.87)
    b = int(np.product(magnitude.shape) * 0.97)
    top_ixs = np.argsort(magnitude.flatten())[a:b]
    top_percent_flow_mean = np.mean(magnitude.flatten()[top_ixs])
    top_flows = flow.reshape((-1, 2))[top_ixs]
    top_magnitudes = magnitude.flatten()[top_ixs]
    ys, xs = np.unravel_index(top_ixs, shape=magnitude.shape)

    with open(f'{tmp_dir}/{i:09d}.pickle', 'wb') as f:
        pickle.dump((top_percent_flow_mean, top_ixs, top_flows, top_magnitudes, xs, ys), f)

    # status
    if i % 100 == 0:
        print(f'Optical flow computed {i} values.')

    return top_percent_flow_mean


def image_overlay(i: int, mean_flow: float, tmp_dir:str, input_path: str, output_path: str):
    # Load optical flow data
    with open(f'{tmp_dir}/{i:09d}.pickle', 'rb') as f:
        top_percent_flow_mean, top_ixs, top_flows, top_magnitudes, xs, ys = pickle.load(f)

    img_pil = Image.open(f'{input_path}/{i:09d}.png')
    draw = ImageDraw.Draw(img_pil, 'RGBA')

    # draw optical flow vectors
    scale = 1
    for x, y, flow, mag in zip(xs, ys, top_flows, top_magnitudes):
        draw.line(xy=[(x, y), (x + flow[0] * mag * scale, y + flow[1] * mag * scale)], fill=(0, 153, 76, 10))

    mean_flow = mean_flow if not np.isnan(mean_flow) else 0.0  # beginning and end of each video aren't computed in rolling mean and are just set to 0.0

    y_start = 70
    x_start = 60
    length = 50
    rectangle = x_start + int(mean_flow * length)
    draw.rectangle(xy=[(60, y_start), (rectangle, y_start + 8)], fill='white')

    # draw beginning and end lines
    draw.rectangle(xy=[(x_start - 5, y_start - 3), (x_start - 3, y_start + 11)], fill='yellow')
    draw.rectangle(xy=[(x_start + length + 3, y_start - 3), (x_start + length + 5, y_start + 11)], fill='yellow')
    draw.text(xy=(x_start - 6, y_start + 15), text='0', fill='yellow')
    draw.text(xy=(x_start + length + 2, y_start + 15), text='1', fill='yellow')

    # draw text value
    draw.text(xy=(x_start - 30, y_start), text=f'{int(mean_flow * 100):d}%', fill='yellow')

    img_pil.save(f'{output_path}/{i:09d}.png')

    if i % 100 == 0:
        print(f'Image overlay renders complete: {i}.')


if __name__ == '__main__':
    main(**arg_parser())
