#!/usr/bin/env bash

# execute the provided prepare script
if [ "$PREPARE_SCRIPT" ]; then
  echo "x${PREPARE_SCRIPT}x"
  chmod +x "${PREPARE_SCRIPT}"
  $PREPARE_SCRIPT
fi

# now run the configuration using poetry

export PYTHONPATH=/jdf
/bin/bash -c "asme $*"