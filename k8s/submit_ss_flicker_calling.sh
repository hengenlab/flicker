#!/usr/bin/env bash

#
# Editable parameters
#
DATASET_NAMES='CAF26 CAF34 CAF42 CAF69 CAF77 CAF99 CAF106 EAB40 EAB50_5state'
MODEL_SIZES='c24k c64k'
PARAM_SETS='flicker-calling-standard'


export JOB_YAML=k8s/job_ss_flicker_calling.yaml

make build push create-tmp-tag

for DATASET_NAME in ${DATASET_NAMES}; do
  for MODEL_SIZE in ${MODEL_SIZES}; do
    for PARAM_SET in ${PARAM_SETS}; do
      echo "Processing ${PARAM_SET}-${DATASET_NAME}-${MODEL_SIZE}"
      export DATASET_NAME_LOWERCASE=$(echo ${DATASET_NAME} | tr A-Z a-z | tr _ -)
      export DATASET_NAME
      export MODEL_SIZE
      export PARAM_SET
      make job-create-nobuild
    done
  done
done
