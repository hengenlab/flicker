#!/usr/bin/env bash
#
# A hard coded script to submit 3 runs for each probe for both 1s and 2.6s models.
# Edit DATASET_NAMES to specify which datasets to submit for
#

# EDITABLE PARAMETERS
#export DATASET_NAMES='CAF26 CAF34 CAF42 CAF69 CAF77 EAB40 EAB50_5state CAF99 CAF106'
export DATASET_NAMES='CAF99'
export RUNS='run1 run2 run3'
export MODELS='c24k c64k'

export ENDPOINT_URL=https://s3-west.nrp-nautilus.io

export USER="dp"
export JOB_YAML='k8s/job_model_sleepstate_train.yaml'
export BRAIN_REGIONS_CAF26='0:64 64:128 128:192'
export BRAIN_REGIONS_CAF34='0:64 64:128 128:192 192:256'
export BRAIN_REGIONS_CAF42='0:64 64:128 128:192 192:256 256:320'
export BRAIN_REGIONS_CAF69='0:64 64:128 128:192 192:256'
export BRAIN_REGIONS_CAF77='0:64 64:128 128:192 192:256'
export BRAIN_REGIONS_EAB40='0:64 64:128 128:192 192:256'
export BRAIN_REGIONS_EAB50_5state='0:64 64:128 128:192 192:256 256:320 320:384 384:448 448:512'
export BRAIN_REGIONS_CAF99='0:64 64:128 128:192 192:256 256:320 320:384 384:448 448:512'
export BRAIN_REGIONS_CAF106='0:64 64:128 128:192 192:256 256:320 320:384 384:448 448:512'

make build push create-tmp-tag
RUNNING_JOBS=$(kubectl get jobs)

for DATASET_NAME in ${DATASET_NAMES}; do
    for MODEL in ${MODELS}; do
        VAR="BRAIN_REGIONS_${DATASET_NAME}"
        export BRAIN_REGIONS=${!VAR}
        for INCLUDE_CHANNELS in ${BRAIN_REGIONS}; do
            for RUN in ${RUNS}; do
                export PARAMETERS="wnr-v14-perregion-${MODEL}"
                export VERSION="wnr-v14-perregion-${MODEL}-${INCLUDE_CHANNELS//:/-}-${RUN}"
                export DATASET_NAME
                export DATASET_NAME_LOWERCASE=$(echo ${DATASET_NAME} | tr A-Z a-z | tr _ -)
                export OVERRIDE="{INCLUDE_CHANNELS: '${INCLUDE_CHANNELS}', VERSION: '${VERSION}'}"

                # Skip if model already running
                if [[ "${RUNNING_JOBS}" == *"${DATASET_NAME_LOWERCASE}-${VERSION}"* ]]; then
                  echo "Skipping ${DATASET_NAME_LOWERCASE}-${VERSION}, job already started."
                  continue
                fi

                echo "Model params: ${DATASET_NAME} --- ${MODEL} --- ${INCLUDE_CHANNELS} --- ${RUN}"

                # Create job
                make job-create-nobuild

                # Show summ stats
                # docker run -v /home/davidparks21/.aws/credentials:/root/.aws/credentials:ro davidparks21/hlmice:latest aws --endpoint https://s3.nautilus.optiputer.net s3 cp s3://hengenlab/${DATASET_NAME}/Results/${VERSION}/summary_statistics_${DATASET_NAME}.txt - | head -n 10 | grep Balanced | xargs -I % bash -c 'echo "${DATASET_NAME}/${VERSION}: %"'
            done
        done
    done
done
