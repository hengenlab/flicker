#!/usr/bin/env bash

#
# Submit jobs for job_viz_lfp_bandpass_plots.yaml
#
CONCURRENT_JOBS=${CONCURRENT_JOBS:-50}

# Functions
ACTIVE_PODS=""
N_AWAITING=""
N_RUNNING=""
POD_FILTER="viz-agg"
query_active_pods () {
  # Set POD_FILTER before calling.
  ACTIVE_PODS=$(kubectl get pods --field-selector=status.phase!=Succeeded --no-headers | grep "${POD_FILTER}")
  N_AWAITING=$(echo "${ACTIVE_PODS}" | grep -iE "Pending|ImagePullBackOff|ErrImagePull|ContainerCreating" | wc -l)
  N_RUNNING=$(echo "${ACTIVE_PODS}" | grep "Running" | wc -l)
}

echo "Submitting viz_aggregate_neural_data_files visualizations for dataset ${DATASET_NAME}"
export JOB_YAML=k8s/job_dataset_histogram_and_permute.yaml
export DATASET_NAME_LOWERCASE=$(echo ${DATASET_NAME} | tr A-Z a-z | tr _ -)

#make build push create-tmp-tag
ALL_JOBS=$(kubectl get jobs)

#for INPUT_FILE in `aws --no-verify-ssl --endpoint https://s3-west.nrp-nautilus.io --profile default s3 ls s3://hengenlab/${DATASET_NAME}/Neural_Data/highpass_750/ | awk '{print $4}' | grep '.bin' | grep -v 'channel-'`; do
for INPUT_FILE in `aws --no-verify-ssl --endpoint https://s3-west.nrp-nautilus.io --profile default s3 ls s3://hengenlab/${DATASET_NAME}/Neural_Data/ 2>/dev/null | awk '{print $4}' | grep '.bin' | grep -v 'channel-'`; do
#for INPUT_FILE in `aws --no-verify-ssl --endpoint https://s3-west.nrp-nautilus.io --profile default s3 ls s3://hengenlab/${DATASET_NAME}/Neural_Data/ 2>/dev/null | awk '{print $4}' | grep '.bin' | grep -v 'channel-'`; do
  export INPUT_FILE
  export UNIQUEID=$(echo ${INPUT_FILE%.*} | tail -c 30 | awk '{print tolower($0)}' | tr '_' '-' | tr '.' '-')

  # Skip if model already running
  if [[ "${ALL_JOBS}" == *"viz-agg-${DATASET_NAME_LOWERCASE}-${UNIQUEID}"* ]]; then
    echo "Skipping viz-agg-${DATASET_NAME_LOWERCASE}-${UNIQUEID}, job already started."
    continue
  fi

  make job-create-nobuild

  # Delay pod submission on various conditions
  query_active_pods
  while [[ "${N_AWAITING}" -ge 10 ]]; do echo "Max jobs in ContainerCreating, Pending, or Error, waiting 15s."; sleep 15; query_active_pods; done
  while [[ "${N_RUNNING}" -gt "${CONCURRENT_JOBS}" ]]; do echo "Max concurrent jobs, waiting 15s."; sleep 15; query_active_pods; done
  #while [ `kubectl get pods | grep viz-agg | grep -iE "Pending|ImagePullBackOff|ErrImagePull|ContainerCreating" | wc -l` -ge 10 ]; do printf "."; sleep 5; done
  #while [[ $(kubectl get jobs | grep viz-agg | grep 0/1 | wc -l) -gt ${CONCURRENT_JOBS} ]]; do echo "Max concurrent jobs, waiting 15s."; sleep 15; done

done
