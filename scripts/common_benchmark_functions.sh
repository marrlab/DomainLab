#!/bin/bash

# This script contains common functions used in multiple scripts

timestamp() {
    date +"%Y-%m-%d_%H-%M-%S"
}

create_log_file() {
    local logdir="zoutput/logs"
    mkdir -p "$logdir"
    local logfile="$logdir/$(timestamp).out"
    echo "verbose log: $logfile"
    echo "$logfile"
}

extract_output_dir() {
    local config_file=$1
    local output_dir=$(awk '/output_dir:/ {print $2}' "$config_file")
    if [ -z "$output_dir" ]; then
        echo "Error: output_dir not specified in $config_file"
        exit 1
    fi
    echo "${output_dir}_$(timestamp)"
}
