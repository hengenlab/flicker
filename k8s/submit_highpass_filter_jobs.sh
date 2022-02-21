#!/usr/bin/env bash

#
# Submit jobs to generate highpass filters for ${DATASET_NAME}
#

# Editable parameters
HZ_PARAMETERS='750'    # '2000 5000 3000 1000 500'
CONCURRENT_JOBS=${CONCURRENT_JOBS:-100}

export TAG=latest
make build push create-tmp-tag
export USER=dp
RUNNING_JOBS=$(kubectl get jobs)

for HZ in ${HZ_PARAMETERS}; do
    export HZ
    export JOB_YAML=k8s/job_filter_neural_file_data.yaml
    export INPUT_PATH=s3://hengenlab/${DATASET_NAME}/Neural_Data/
    export OUTPUT_PATH=s3://hengenlab/${DATASET_NAME}/Neural_Data/highpass_${HZ}/
    for F in `aws3swfs ls s3://hengenlab/${DATASET_NAME}/Neural_Data/ | grep Headstages_ | awk '{print $4}'`; do
        echo ${HZ} - ${F}
        export INPUT_FILE=${INPUT_PATH}${F}
        export OUTPUT_FILE=${OUTPUT_PATH}${F}
        export UNIQUEID=$(echo ${F%.*}-${HZ} | tail -c 30 | awk '{print tolower($0)}' | tr '_' '-' | tr '.' '-')
        export F

        # Skip if model already running
        if [[ "${RUNNING_JOBS}" == *"${USER}-ss-filter-lfp-${DATASET_NAME_LOWERCASE}-${UNIQUEID} "* ]]; then
          echo "Skipping ${DATASET_NAME_LOWERCASE}-${VERSION}, job already started."
          continue
        fi

        make job-create-nobuild

        while [ `kubectl get pods | grep ss-filter-lfp | grep -iE "Pending|ImagePullBackOff|ErrImagePull|ContainerCreating" | wc -l` -ge 10 ]; do printf "."; sleep 5; done
        while [[ $(kubectl get jobs | grep ss-filter-lfp | grep 0/1 | wc -l) -gt ${CONCURRENT_JOBS} ]]; do echo "Max concurrent jobs (${CONCURRENT_JOBS}), waiting 15s."; sleep 15; done
    done
done
