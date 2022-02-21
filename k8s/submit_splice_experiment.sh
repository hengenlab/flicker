#!/usr/bin/env bash

#
# SPLICE EXPERIMENT - POSITIVE CONTROL - 100 runs for statistical significance
#   To gather statistics see src/scripts/splice_summary_stats.py
#

# The prediction filename to draw confident segments from
export PREDICTION_FILE="results/EAB50_5state/wnr-v14-centered24k-perregion-128-192/predictions_EAB50_5state.csv"
export OUTPUT_NEURAL_FILES_PREFIX="splice-v5"
export DATASET_NAME="EAB50_5state"
export TEST_SET_OFFSET=378010

generate_segments() {
    # AWK filtering of contiguous segments. Produce a csv of segments of high confidence regions in form:
    # label,start_segment,end_segment_inclusive,segment_length
    grep e3v817b-20190705T2155-2255.mp4 ${PREDICTION_FILE} | \
    awk -F, '$1 == $2 && ( $4 > 0.9 || $5 > 0.9 || $6 > 0.9 ) { print }' | \
    awk -F, 'TSO=ENVIRON["TEST_SET_OFFSET"] { if ($9 != (X + 1)) { printf("%05d\n", X + TSO); printf("%d,%05d\n", $1, $9 + TSO) } X = $9 }' | \
    sed 1,1d | \
    awk -F'[ ]' '{ if (NR % 2 == 0) {print X "," $1} else {X = $1}}' | \
    awk -F, '{ print $0 "," ($3 - $2 + 1)}' | \
    sort | \
    awk -F, 'SEG=ENVIRON["SEGMENT_LENGTH"] { for( i=$2; i <= $3 - SEG + 1; i += SEG ){print $0} }' |
    uniq \
    > /tmp/segments.tmp
}

generate_source_target() {
    case "${SOURCE}" in
        WAKE) export LABEL_SOURCE=0;;
        NREM) export LABEL_SOURCE=1;;
        REM) export LABEL_SOURCE=2;;
    esac
    case "${TARGET}" in
        WAKE) export LABEL_TARGET=0;;
        NREM) export LABEL_TARGET=1;;
        REM) export LABEL_TARGET=2;;
    esac

#    echo "Randomly selecting a SEGMENT_SOURCE_BEGIN and SEGMENT_TARGET_BEGIN from SEGMENTS for {$SOURCE} to {$TARGET}"
    unset SL SOURCE_START SOURCE_END SOURCE_COUNT TL TARGET_START TARGET_END TARGET_COUNT SEGMENT_SOURCE_BEGIN SEGMENT_TARGET_BEGIN
    IFS=, read -r SL SOURCE_START SOURCE_END SOURCE_COUNT < <(cat /tmp/segments.tmp | grep "^${LABEL_SOURCE}," | shuf -n 1)
    IFS=, read -r TL TARGET_START TARGET_END TARGET_COUNT < <(cat /tmp/segments.tmp | grep "^${LABEL_TARGET}," | shuf -n 1)

    echo $SL $SOURCE_START $SOURCE_END $SOURCE_COUNT $TL $TARGET_START $TARGET_END $TARGET_COUNT;

    export SEGMENT_SOURCE_BEGIN=$(( $((10#${SOURCE_START})) + ${RANDOM} % ( (${SOURCE_COUNT} + 1) - ${SEGMENT_LENGTH}) ));
    export SEGMENT_TARGET_BEGIN=$(( $((10#${TARGET_START})) + ${RANDOM} % ( (${TARGET_COUNT} + 1) - ${SEGMENT_LENGTH}) ));
}

SOURCES_AND_TARGETS=(
    WAKE
    NREM
    REM
)
#SEGMENT_LENGTHS=(
#    30
#    15
#    10
#    7
#    5
#    3
#    2
#    1
#)
# segment lengths for <10ms splice experiment (splice-v5)
SEGMENT_LENGTHS=(
    105
    70
    35
    14
    7
    5
    3
    2
    1
)

export REPLACE_METHOD="step-function"
export JOB_YAML=k8s/job_model_sleepstate_splice_control_experiment.yaml
for SOURCE in "${SOURCES_AND_TARGETS[@]}"; do
    for TARGET in "${SOURCES_AND_TARGETS[@]}"; do
        for I in {1..100}; do
            echo $LABEL $START $END $COUNT;
            if [ ${SOURCE} == ${TARGET} ]; then continue; fi
            for SEGMENT_LENGTH in "${SEGMENT_LENGTHS[@]}"; do
                generate_segments
                generate_source_target
                export REPLACE_METHOD SOURCE TARGET SEGMENT_LENGTH
                export SEGMENT_LENGTH_PAD=$(printf %03d $SEGMENT_LENGTH)
                export I_PAD=$(printf %03d $I)
                export SPLICE_NAME="${SOURCE,,}-to-${TARGET,,}-len${SEGMENT_LENGTH_PAD}-iter${I_PAD}"

                # Delay when too many jobs submitted
                PENDING_JOBS=$(kubectl get pods | grep Pending | wc -l)
                TOTAL_JOBS=$(kubectl get pods | grep "ss-splice" | grep "Running" | wc -l)
                while [ "${PENDING_JOBS}" -gt 3 ] || [ "${TOTAL_JOBS}" -gt 40 ]; do
                    echo "Waiting 30s on ${PENDING_JOBS} pending jobs or ${TOTAL_JOBS} total jobs at $(date).";
                    sleep 30;
                    PENDING_JOBS=$(kubectl get pods | grep Pending | wc -l)
                    TOTAL_JOBS=$(kubectl get pods | grep "ss-splice" | grep "Running" | wc -l)
                done

                if aws --endpoint https://s3.nautilus.optiputer.net s3 ls "s3://hengenlab/${DATASET_NAME}/Runs/${OUTPUT_NEURAL_FILES_PREFIX}/${SPLICE_NAME}/Results/predictions-${SPLICE_NAME}.csv.zip"; then
                  echo "${SPLICE_NAME} exists, skipping."
                else
                  echo "job-create-nobuild: $I_PAD $REPLACE_METHOD - $SOURCE - $TARGET - $SEGMENT_LENGTH - $SPLICE_NAME - $SEGMENT_SOURCE_BEGIN - $SEGMENT_TARGET_BEGIN"
                  echo "${SPLICE_NAME},${SEGMENT_LENGTH},${SEGMENT_SOURCE_BEGIN},${SEGMENT_TARGET_BEGIN}" | aws --endpoint https://s3.nautilus.optiputer.net s3 cp - "s3://hengenlab/${DATASET_NAME}/Results/${OUTPUT_NEURAL_FILES_PREFIX}/${SPLICE_NAME}/params_${SPLICE_NAME}.csv"
                  make job-create-nobuild
                fi
            done;
        done;
    done;
done;

rm /tmp/segments.tmp
