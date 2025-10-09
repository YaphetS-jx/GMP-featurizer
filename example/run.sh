#!/bin/bash

# Input arguments with default values
CONFIG_PATH=${1:-./config.json}
CPU_OR_GPU=${2:-cpu}  # Options: cpu | gpu
USE_GDB=${3:-false}
USE_GPROF=${4:-false}
FEATURIZER_PATH=${5:-../build/gmp-featurizer}
GMON_PATH=${6:-../profile/gmon.out}

if [ "$USE_GDB" == "true" ]; then
    if [ "$CPU_OR_GPU" == "gpu" ]; then
        echo "running GMP Featurizer (GPU) in $FEATURIZER_PATH with cuda-gdb"
        cuda-gdb --args $FEATURIZER_PATH \
            $CONFIG_PATH
    else
        echo "running GMP Featurizer (CPU) in $FEATURIZER_PATH with gdb"
        gdb --args $FEATURIZER_PATH \
            $CONFIG_PATH
    fi
elif [ "$USE_GPROF" == "true" ]; then
    if [ "$CPU_OR_GPU" == "gpu" ]; then
        echo "running GMP Featurizer (GPU) in $FEATURIZER_PATH with nsys"
        # Use NVIDIA Nsight Systems for GPU profiling
        nsys profile --force-overwrite=true -o nsys/nsys_profile $FEATURIZER_PATH $CONFIG_PATH
    else
        echo "running GMP Featurizer (CPU) in $FEATURIZER_PATH with gprof using $GMON_PATH"
        # First run the program to generate gmon.out
        $FEATURIZER_PATH $CONFIG_PATH
        # Then analyze the profiling data
        # gprof $FEATURIZER_PATH $GMON_PATH > gprof.txt
        gprof $FEATURIZER_PATH $GMON_PATH > gprof
    fi
else
    echo "running GMP Featurizer in $FEATURIZER_PATH"
    $FEATURIZER_PATH \
        $CONFIG_PATH
fi