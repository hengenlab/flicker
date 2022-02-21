#!/usr/bin/env bash
# Runs optical_flow.py on a single image as a job
# Usage Example:
#     scripts/optical_flow.sh CAF42 s3://hengenlab/CAF42/Video/e3v81a6-20200915T162254-172255.mp4 32
#                             dataset                s3-url-video                                parallelism

set -e
export dataset_name=${1}
export video=${2}
export parallelism=${3:-32}

echo "Processing ${dataset_name} - ${video} - ${parallelism}"
cd /tmp
mkdir /tmp/optical_flow_frames/
mkdir /tmp/optical_flow_tmp/
mkdir /tmp/optical_flow_output/
export video_base=$(basename ${video} .mp4)

s3 get ${video}

# convert to video frames
ffmpeg -i ${video_base}.mp4 -vsync 0 -start_number 0 -vf fps=15 "optical_flow_frames/$frame_%09d.png"
export img_count=$(ls optical_flow_frames | wc -l)
#if [ "$(ls optical_flow_frames/ | wc -l)" -le 53000 ]; then
#  echo "Error: only found $(ls optical_flow_frames/ | wc -l) frames"
#  exit 1
#fi

python -u /project_neural_mouse/src/scripts/optical_flow.py \
  --img_count ${img_count} \
  --input_path /tmp/optical_flow_frames \
  --output_path /tmp/optical_flow_output \
  --tmp_dir /tmp/optical_flow_tmp \
  --parallelism ${parallelism}

# upload CSV
gzip optical_flow_output/results.csv
s3 put -f optical_flow_output/results.csv.gz s3://hengenlab/optical_flow/results/${dataset_name}_${video_base}.csv.gz

# make video of overlay and upload
ffmpeg -framerate 15 -pattern_type glob -i 'optical_flow_output/*.png' -c:v libx264 -r 15 -pix_fmt yuv420p out.mp4
s3 put -f out.mp4 s3://hengenlab/optical_flow/videos/${dataset_name}_${video_base}.mp4
