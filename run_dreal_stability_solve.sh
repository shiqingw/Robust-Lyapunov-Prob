#!/usr/bin/env bash

PRECISION=1.0E-03

log(){ echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

trap 'log "Wrapper received SIGTERM; forwarding to $PID"; kill -TERM $PID' TERM

trap 'log "Wrapper received SIGINT; forwarding to $PID";  kill -INT  $PID' INT


for EXP_NUM in 1; do
    # prepare directory + log file
    DIR=$(printf "eg4_results/%03d" "$EXP_NUM")
    LOG="$DIR/output_dreal_stability_solve_${PRECISION}.out"
    mkdir -p "$DIR"

    log "=== STARTING python: exp_num=$EXP_NUM, precision=$PRECISION ==="
    python -u eg4_quadrotor_2d/dreal_stability_solve.py \
        --exp_num $EXP_NUM --dreal_precision $PRECISION >> "$LOG" 2>&1 &
    PID=$!
    log "Launched python with PID $PID"

    # wait and capture exit status
    wait $PID
    STATUS=$?

    if [ $STATUS -eq 0 ]; then
        log "Python process $PID exited normally (0)."
    elif [ $STATUS -gt 128 ]; then
        SIG=$(( STATUS - 128 ))
        log "Python process $PID was killed by signal $SIG."
    else
        log "Python process $PID exited with code $STATUS."
    fi

done
