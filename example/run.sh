#!/bin/bash

# Input arguments with default values
FEATURIZER_PATH=${1:-../build/gmp-featurizer}
USE_GDB=${2:-false}

if [ "$USE_GDB" == "true" ]; then
    echo "running GMP Featurizer in $FEATURIZER_PATH with gdb"
    gdb --args $FEATURIZER_PATH  \
        ./config.json
else
    echo "running GMP Featurizer in $FEATURIZER_PATH"
    $FEATURIZER_PATH  \
        ./config.json
fi

echo "done"