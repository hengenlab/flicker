#!/usr/bin/env bash
#
# A hard coded script to submit models per brain region per highpass filter.
# Edit DATASET_NAMES to specify which datasets to submit for
#

# EDITABLE PARAMETERS
export DATASET_NAMES='CAF26 CAF34 CAF42 CAF69 CAF77 CAF99 CAF106 EAB40 EAB50_5state'
export HZ_PARAMETERS='2000 5000 500 1000 3000'
export SKIP_WHEN_MODEL_EXISTS='false'  # set to 'true' to check if a Model exists and not run in those cases, 'false' to start the job regardless and produce a new model.
export SKIP_WHEN_RESULTS_EXISTS='false'  # set to 'true' to check if a Results exists and not run in those cases, 'false' to start the job regardless and produce a new model.

export ENDPOINT_URL='https://s3-west.nrp-nautilus.io'
export USER='dp'
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

#make build push create-tmp-tag
RUNNING_JOBS=$(kubectl get jobs)

for DATASET_NAME in ${DATASET_NAMES}; do
    export DATASET_NAME_LOWERCASE=${DATASET_NAME,,}
    export DATASET_NAME_LOWERCASE=${DATASET_NAME_LOWERCASE/_/-}
    if [ "${DATASET_NAME}" = "EAB50_5state" ]; then
      export DATASET_NAME_LOWERCASE="eab505"
    else
      export DATASET_NAME_LOWERCASE=$(echo ${DATASET_NAME} | tr A-Z a-z | tr _ -)
    fi
    echo using "DATASET_NAME_LOWERCASE: ${DATASET_NAME_LOWERCASE}"

    VAR="BRAIN_REGIONS_${DATASET_NAME}"
    export BRAIN_REGIONS=${!VAR}
    for INCLUDE_CHANNELS in ${BRAIN_REGIONS}; do
        for HZ_PARAMETER in ${HZ_PARAMETERS}; do
          export PARAMETERS="wnr-v14-perregion-c24k"
          export VERSION="wnr-v14-c24k-highpass-${HZ_PARAMETER}hz-region-${INCLUDE_CHANNELS//:/-}"
          export DATASET_NAME
          export OVERRIDE="{datasetparams: {neural_files_basepath: 's3://hengenlab/${DATASET_NAME}/Neural_Data/highpass_${HZ_PARAMETER}/'}, INCLUDE_CHANNELS: '${INCLUDE_CHANNELS}', VERSION: '${VERSION}'}";

          # Optionally skip if model is already generated (comment or uncomment this line)
          if [[ ${SKIP_WHEN_MODEL_EXISTS} == 'true' ]]; then
            if aws --endpoint https://s3.nautilus.optiputer.net s3 ls s3://hengenlab/${DATASET_NAME}/Runs/${VERSION}/Model/ >/dev/null 2>/dev/null; then
              echo "Model already exist for ${DATASET_NAME}/${VERSION}, skipping."
              continue
              fi
          fi

          # Optionally skip if model is already generated (comment or uncomment this line)
          if [[ ${SKIP_WHEN_RESULTS_EXISTS} == 'true' ]]; then
            if aws --endpoint https://s3.nautilus.optiputer.net s3 ls s3://hengenlab/${DATASET_NAME}/Runs/${VERSION}/Results/ >/dev/null 2>/dev/null; then
              echo "Results already exist for ${DATASET_NAME}/${VERSION}, skipping."
              continue
              fi
          fi

          # Skip if model already running
          if [[ "${RUNNING_JOBS}" == *"${DATASET_NAME_LOWERCASE}-${VERSION}"* ]]; then
            echo "Skipping ${DATASET_NAME_LOWERCASE}-${VERSION}, job already started."
            continue
          fi

          echo "Model params: ${DATASET_NAME} --- ${INCLUDE_CHANNELS} --- ${HZ_PARAMETER}"

          # Create job
          make job-create-nobuild

          while [ `kubectl get pods | grep highpass | grep -iE "Pending|ImagePullBackOff|ErrImagePull|ContainerCreating" | wc -l` -ge 40 ]; do printf "."; sleep 5; done
          # Show summ stats
          # docker run -v /home/davidparks21/.aws/credentials:/root/.aws/credentials:ro davidparks21/hlmice:latest aws --endpoint https://s3.nautilus.optiputer.net s3 cp s3://hengenlab/${DATASET_NAME}/Results/${VERSION}/summary_statistics_${DATASET_NAME}.txt - | head -n 10 | grep Balanced | xargs -I % bash -c 'echo "${DATASET_NAME}/${VERSION}: %"'
        done
    done
done
