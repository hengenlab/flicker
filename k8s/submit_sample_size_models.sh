#!/usr/bin/env bash
#
# A hard coded script to submit 3 runs for each probe for both 1s and 2.6s models.
# Edit DATASET_NAMES to specify which datasets to submit for
#

# EDITABLE PARAMETERS
FS='ceph'   # 'swfs' or 'ceph'
#export DATASET_NAMES='CAF26 CAF34 CAF42 CAF69 CAF77 EAB50_5state CAF99 CAF106'
if [ ${FS} == 'ceph' ]; then
  export DATASET_NAMES='CAF42 CAF106 CAF99 EAB50_5state'  # Done: CAF34
  export ENDPOINT_URL=https://s3-west.nrp-nautilus.io
elif [ ${FS} == 'swfs' ]; then
  export DATASET_NAMES=''  #
  export ENDPOINT_URL=https://swfs-s3.nrp-nautilus.io
else echo "FS not set correctly, got ${FS} expected ceph or swfs"
fi
export SAMPLE_SIZES='65536 16384 8192 4096 1024 256 128 64 32 16 4 1'
export SKIP_WHEN_MODEL_EXISTS='false'  # set to 'true' to check if a Model exists and not run in those cases, 'false' to start the job regardless and produce a new model.
export SKIP_WHEN_RESULTS_EXISTS='false'  # set to 'true' to check if a Results exists and not run in those cases, 'false' to start the job regardless and produce a new model.


export USER="dp"
export JOB_YAML='k8s/job_model_sleepstate_train.yaml'
#export BRAIN_REGIONS_CAF26='0:192'
#export BRAIN_REGIONS_CAF26='0:64 64:128 128:192'
#export BRAIN_REGIONS_CAF26='10 14 21 28 31 36  83 86 88 89 104 112  136 147 167 176 177 183'
export BRAIN_REGIONS_CAF26='10 31  88 104  167 183'
#export BRAIN_REGIONS_CAF34='0:256'
#export BRAIN_REGIONS_CAF34='0:64 64:128 128:192 192:256'
#export BRAIN_REGIONS_CAF34='9 14 15 17 50 57  76 90 99 106 115 126  144 148 163 167 170 178  193 195 202 220 240 248'
export BRAIN_REGIONS_CAF34='9 14  76 90  144 148  193 195'  # 2 channels from each probe
#export BRAIN_REGIONS_CAF42='0:256'
#export BRAIN_REGIONS_CAF42='0:64 64:128 128:192 192:256 256:320'
#export BRAIN_REGIONS_CAF42='9 14 23 27 38 55   69 81 99 102 110 118   152 168 172 173 179 183   206 216 218 219 230 242   259 260 264 271 278 306'
export BRAIN_REGIONS_CAF42='14 38  69 110  152 172  219 230  260 278'  # 1 spike 1 non spike each probe
#export BRAIN_REGIONS_CAF69='0:256'
#export BRAIN_REGIONS_CAF69='0:64 64:128 128:192 192:256'
#export BRAIN_REGIONS_CAF69='0 1 29 39 55 63  69 75 104 111 116 122  135 138 148 169 180 184  204 212 216 218 224 238'
export BRAIN_REGIONS_CAF69='29 39  75 104  138 180  204 218'
#export BRAIN_REGIONS_CAF77='64:256 128:256'
#export BRAIN_REGIONS_CAF77='0:64 64:128 128:192 192:256'
#export BRAIN_REGIONS_CAF77='66 71 93 96 105 110  132 134 140 157 178 187  192 195 202 213 228 233'
export BRAIN_REGIONS_CAF77='71 110  140 157  192 233'
#export BRAIN_REGIONS_EAB40='0:256'
#export BRAIN_REGIONS_EAB40='0:64 64:128 128:192 192:256'
#export BRAIN_REGIONS_EAB40='3 7 10 11 14 33 64 65 73 74 87 110 130 145 151 163 178 186 192 199 200 215 220 225'
export BRAIN_REGIONS_EAB40='3 10  65 110  163 186  200 225'
#export BRAIN_REGIONS_EAB50_5state='0:512'
#export BRAIN_REGIONS_EAB50_5state='0:64 64:128 128:192 192:256 256:320 320:384 384:448 448:512'
#export BRAIN_REGIONS_EAB50_5state='1 4 29 39 40 51  73 74 75 86 99 112  132 135 137 174 180 190  194 195 199 201 211 215  256 258 260 271 274 282  331 335 339 347 350 371  386 390 395 399 409 436'
export BRAIN_REGIONS_EAB50_5state='4 39  75 99  132 180  195 201  260 271  335 350  386 409'
#export BRAIN_REGIONS_CAF99='0:512'
#export BRAIN_REGIONS_CAF99='0:64 64:128 128:192 192:256 256:320 320:384 384:448 448:512'
#export BRAIN_REGIONS_CAF99='0 3 4 6 11 60  67 72 78 103 112 124  134 159 162 182 183 178  193 197 230 234 237 238  256 265 281 283 288 303  332 348 351 355 356 363  397 399 405 406 414 433  448 451 465 472 477 496'
export BRAIN_REGIONS_CAF99='6 60  72 78 159 178  197 234  265 288  348 356  397 399  448 477'
#export BRAIN_REGIONS_CAF106='0:512'
#export BRAIN_REGIONS_CAF106='0:64 64:128 128:192 192:256 256:320 320:384 384:448 448:512'
#export BRAIN_REGIONS_CAF106='12 24 25 35 59 63  78 81 85 88 93 96  132 133 137 146 147 162  203 200 210 213 215 238  260 262 290 308 309 317   391 399 400 412 425 437  421 448 452 463 475 496'
export BRAIN_REGIONS_CAF106='12 59  78 85  132 146  200 213  260 290  391 412  448 475'
make build push create-tmp-tag
RUNNING_JOBS=$(kubectl get jobs)
unset UNIQUEID

for DATASET_NAME in ${DATASET_NAMES}; do
    VAR="BRAIN_REGIONS_${DATASET_NAME}"
    export BRAIN_REGIONS=${!VAR}
    export DATASET_NAME
    if [ "${DATASET_NAME}" = "EAB50_5state" ]; then
      export DATASET_NAME_LOWERCASE="eab505"
    else
      export DATASET_NAME_LOWERCASE=$(echo ${DATASET_NAME} | tr A-Z a-z | tr _ -)
    fi
    echo using "DATASET_NAME_LOWERCASE: ${DATASET_NAME_LOWERCASE}"

    for RUN in {1..1}; do
        for INCLUDE_CHANNELS in ${BRAIN_REGIONS}; do
            for SAMPLE_SIZE in ${SAMPLE_SIZES}; do
                export SAMPLE_SIZE_PAD=$(printf %06d ${SAMPLE_SIZE})
                export SAMPLE_WIDTH_BEFORE=$(((SAMPLE_SIZE+1)/2))
                export SAMPLE_WIDTH_AFTER=$((SAMPLE_SIZE/2))

#                export PARAMETERS="wnr-v14-samplesz"
                export PARAMETERS="wnr-v14-samplesz-permute"

#                export VERSION="wnr-v14-run7.${RUN}-szhighnoshuff-ch-${INCLUDE_CHANNELS//[:,]/-}-sz-${SAMPLE_SIZE}"
#                export VERSION="wnr-v14-run7.${RUN}-szhighwshuff-ch-${INCLUDE_CHANNELS//[:,]/-}-sz-${SAMPLE_SIZE}"
                export VERSION="wnr-v14-szhighpersampleshuff-ch-${INCLUDE_CHANNELS//[:,]/-}-sz-${SAMPLE_SIZE}"
#                export VERSION="wnr-v14-run5-ch-${INCLUDE_CHANNELS//[:,]/-}-sz-${SAMPLE_SIZE}"
#                export VERSION="wnr-v14-whole-ch-${INCLUDE_CHANNELS//[:,]/-}-sz-${SAMPLE_SIZE}"
#                export VERSION="wnr-v14-size
#                highpass-ch-${INCLUDE_CHANNELS//[:,]/-}-sz-${SAMPLE_SIZE}"

#                export OVERRIDE="{INCLUDE_CHANNELS: '${INCLUDE_CHANNELS}', VERSION: '${VERSION}', SAMPLE_WIDTH_BEFORE: '${SAMPLE_WIDTH_BEFORE}', SAMPLE_WIDTH_AFTER: '${SAMPLE_WIDTH_AFTER}'}"
#                export OVERRIDE="{INCLUDE_CHANNELS: '${INCLUDE_CHANNELS}', VERSION: '${VERSION}', SAMPLE_WIDTH_BEFORE: '${SAMPLE_WIDTH_BEFORE}', SAMPLE_WIDTH_AFTER: '${SAMPLE_WIDTH_AFTER}', datasetparams: {neural_files_basepath: 's3://hengenlab/${DATASET_NAME}/Neural_Data/highpass_750/classpermuted/'}}"
                export OVERRIDE="{INCLUDE_CHANNELS: '${INCLUDE_CHANNELS}', VERSION: '${VERSION}', SAMPLE_WIDTH_BEFORE: '${SAMPLE_WIDTH_BEFORE}', SAMPLE_WIDTH_AFTER: '${SAMPLE_WIDTH_AFTER}', datasetparams: {neural_files_basepath: 's3://hengenlab/${DATASET_NAME}/Neural_Data/highpass_750/'}}"

                # Optionally skip if model is already generated (comment or uncomment this line)
                if [[ ${SKIP_WHEN_MODEL_EXISTS} == 'true' ]]; then
                  if aws --endpoint ${ENDPOINT_URL} s3 ls s3://hengenlab/${DATASET_NAME}/Runs/${VERSION}/Model/ >/dev/null 2>/dev/null; then
                    echo "Model already exist for ${DATASET_NAME}/${VERSION}, skipping."
                    continue
                    fi
                fi

                # Optionally skip if results is already generated (comment or uncomment this line)
                if [[ ${SKIP_WHEN_RESULTS_EXISTS} == 'true' ]]; then
                  if aws --endpoint ${ENDPOINT_URL} s3 ls s3://hengenlab/${DATASET_NAME}/Runs/${VERSION}/Results/ >/dev/null 2>/dev/null; then
                    echo "Results already exist for ${DATASET_NAME}/${VERSION}, skipping."
                    continue
                    fi
                fi

                # Skip if model already running
                if [[ "${RUNNING_JOBS}" == *"${DATASET_NAME_LOWERCASE}-${VERSION} "* ]]; then
                  echo "Skipping ${DATASET_NAME_LOWERCASE}-${VERSION}, job already started."
                  continue
                fi

                echo "Model params: ${DATASET_NAME} --- ${DATASET_NAME_LOWERCASE} --- ${SAMPLE_SIZE_PAD} --- ${INCLUDE_CHANNELS} --- ${VERSION}  ---  ${OVERRIDE}"

                # Create job
                make job-create-nobuild

                # Wait while pending
                while [ `kubectl get pods | grep run5 | grep -iE "Pending|ImagePullBackOff|ErrImagePull|ContainerCreating" | wc -l` -ge 60 ]; do printf "."; sleep 5; done
    #            while [ `kubectl get pods | grep run5 | grep -iE "run5.*Running" | wc -l` -ge 10 ]; do printf "*"; sleep 5; done
            done
        done
    done
done
