#!/bin/bash

# Input arguments with default values
CPU_OR_GPU=${1:-cpu}  # Options: cpu | gpu
USE_GDB=${2:-false}
USE_GPROF=${3:-false}
FEATURIZER_PATH=${4:-../build/gmp-featurizer}
GMON_PATH=${5:-../profile/gmon.out}

if [ "$USE_GDB" == "true" ]; then
    if [ "$CPU_OR_GPU" == "gpu" ]; then
        echo "running GMP Featurizer (GPU) in $FEATURIZER_PATH with cuda-gdb"
        cuda-gdb --args $FEATURIZER_PATH \
            ./config.json
    else
        echo "running GMP Featurizer (CPU) in $FEATURIZER_PATH with gdb"
        gdb --args $FEATURIZER_PATH \
            ./config.json
    fi
elif [ "$USE_GPROF" == "true" ]; then
    if [ "$CPU_OR_GPU" == "gpu" ]; then
        echo "running GMP Featurizer (GPU) in $FEATURIZER_PATH with nsys"
        # Use NVIDIA Nsight Systems for GPU profiling
        nsys profile --force-overwrite=true -o nsys_profile $FEATURIZER_PATH ./config.json
    else
        echo "running GMP Featurizer (CPU) in $FEATURIZER_PATH with gprof using $GMON_PATH"
        # First run the program to generate gmon.out
        $FEATURIZER_PATH ./config.json
        # Then analyze the profiling data
        # gprof $FEATURIZER_PATH $GMON_PATH > gprof.txt
        gprof $FEATURIZER_PATH $GMON_PATH > gprof
    fi
else
    echo "running GMP Featurizer in $FEATURIZER_PATH"
    $FEATURIZER_PATH \
        ./config.json
fi