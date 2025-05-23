#!/bin/bash

# Input arguments with default values
FEATURIZER_PATH=${1:-../build/gmp-featurizer}
USE_GDB=${2:-false}

if [ "$USE_GDB" == "true" ]; then
    echo "running GMP Featurizer in $FEATURIZER_PATH with gdb"
    gdb --args $FEATURIZER_PATH  \
        systemPath ./test.cif pspPath ./QE-kjpaw.gpsp \
        orders -1,0,1,2 sigmas 0.1,0.2,0.3
else
    echo "running GMP Featurizer in $FEATURIZER_PATH"
    $FEATURIZER_PATH  \
        systemPath ./test.cif pspPath ./QE-kjpaw.gpsp \
        orders -1,0,1,2 sigmas 0.1,0.2,0.3
fi

echo "done"