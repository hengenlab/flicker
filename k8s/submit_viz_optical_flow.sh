#!/usr/bin/env bash
# Submit job_viz_optical_flow.yaml for each video file on S3.

export DATASET_NAMES='JNM_opticalflow_CAF72' # 'CAF99 CAF106' # CAF77 EAB40 EAB50_5state CAF26 CAF34 CAF42 CAF69'
#make build pusnano h create-tmp-tag
export JOB_YAML=k8s/job_viz_optical_flow.yaml
export parallelism=8

# Exclusions
#   EAB40  / e3v8100-20190402T1128-1140.mp4  -  not in labels matrix, odd sized video
#   CAF106 / CAF00106-20210608T100851-110522.mp4  -  Error in video mapping, 56k in labels matrix, 50k in video file, all subsequent video mapping is untrustworthy
#            CAF00106-20210608T110856.mp4
#            CAF00106-20210608T120856.mp4
#            CAF00106-20210608T130857.mp4
#            CAF00106-20210608T140857-150858.mp4
#            CAF00106-20210608T150858-160858.mp4
#            CAF00106-20210608T160858-170859.mp4
#            CAF00106-20210608T170859-180900.mp4
#            CAF00106-20210608T180900-190900.mp4
#            CAF00106-20210608T190900-200901.mp4
#            CAF00106-20210608T200901-210902.mp4
#            CAF00106-20210608T210902.mp4
#            CAF00106-20210608T220903-230903.mp4
#            CAF00106-20210608T230903-000904.mp4
#   CAF99  / CAF00099-20210608T100849-110521.mp4  -  Error in video mapping, 56k in labels matrix, 50k in video file, all subsequent video mapping is untrustworthy
#            CAF00099-20210608T120857.mp4
#            CAF00099-20210608T130858.mp4
#            CAF00099-20210608T140858.mp4
#            CAF00099-20210608T150859-160859.mp4
#            CAF00099-20210608T160859-170900.mp4
#            CAF00099-20210608T170900-180900.mp4
#            CAF00099-20210608T180900-190900.mp4
#            CAF00099-20210608T190900-200901.mp4
#            CAF00099-20210608T200901-210901.mp4
#            CAF00099-20210608T210901.mp4
#            CAF00099-20210608T220902-230903.mp4
#            CAF00099-20210608T230903-000904.mp4

for DATASET_NAME in ${DATASET_NAMES}; do
    export DATASET_NAME
    if [ "${DATASET_NAME}" = "EAB50_5state" ]; then
      export DATASET_NAME_LOWERCASE="eab505"
    else
      export DATASET_NAME_LOWERCASE=$(echo ${DATASET_NAME} | tr A-Z a-z | tr _ -)
    fi

    for VIDEO_FILE in `s3 ls s3://hengenlab/${DATASET_NAME}/Video/ 2>/dev/null | grep ".mp4" | awk '{print $4}'`; do

        export VIDEO_FILE_BASENAME=$(basename ${VIDEO_FILE} .mp4)
        export UNIQUEID=$(echo ${VIDEO_FILE_BASENAME} | tr A-Z a-z | tr _ - | tail -c 20)
        export VIDEO_FILE

#        # Skip incomplete videos
#        if [ "$VIDEO_FILE_BASENAME" == "e3v8100-20190402T1128-1140" ]; then
#          echo "Skipping e3v8100-20190402T1128-1140.mp4"
#          continue
#        fi

        if aws3ceph ls s3://hengenlab/optical_flow/results/${DATASET_NAME}_${VIDEO_FILE_BASENAME}.csv.gz 1>/dev/null 2>/dev/null; then
          echo "SKIPPING: ${DATASET_NAME} - ${VIDEO_FILE_BASENAME} - s3://hengenlab/optical_flow/results/${DATASET_NAME}_${VIDEO_FILE_BASENAME}.csv.gz"
        else
          echo "PROCESSING: ${DATASET_NAME} - ${VIDEO_FILE_BASENAME} - s3://hengenlab/optical_flow/results/${DATASET_NAME}_${VIDEO_FILE_BASENAME}.csv.gz"

          make job-create-nobuild
        fi

        # Wait while pending
        while [ `kubectl get pods | grep opflow | grep -iE "Pending|ImagePullBackOff|ErrImagePull|ContainerCreating" | wc -l` -ge 2 ]; do printf "."; sleep 5; done
    done
done
