#!/bin/bash

# Input arguments with default values
USE_GDB=${1:-false}
USE_GPROF=${2:-false}
FEATURIZER_PATH=${3:-../../build/gmp-featurizer}
GMON_PATH=${4:-../../example/gmon.out}

if [ "$USE_GDB" == "true" ]; then
    echo "running GMP Featurizer in $FEATURIZER_PATH with gdb"
    gdb --args $FEATURIZER_PATH  \
        ./config.json
elif [ "$USE_GPROF" == "true" ]; then
    echo "running GMP Featurizer in $FEATURIZER_PATH with gprof using $GMON_PATH"
    # First run the program to generate gmon.out
    $FEATURIZER_PATH ./config.json
    # Then analyze the profiling data
    gprof $FEATURIZER_PATH $GMON_PATH > gprof.txt
else
    echo "running GMP Featurizer in $FEATURIZER_PATH"
    $FEATURIZER_PATH  \
        ./config.json
fi

echo "done"