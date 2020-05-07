#!/bin/bash

# adapted from https://github.com/scikit-learn/scikit-learn/blob/master/build_tools/travis/test_script.sh

# Exit the script if any statement returns a non-true return value
set -e

# Activate conda environment
deactivate || :
source activate testenv

# Print test environment
conda list

# Get into a temp directory to run test from the installed scikit-learn and
# check if we do not leave artifacts
mkdir -p "$TEST_DIR"

# We need to copy the setup.cfg for the pytest settings
cp setup.cfg "$TEST_DIR"

# Change directory
cd "$TEST_DIR"

# Define test command
TEST_CMD="pytest"
TEST_ARGS=(--verbose --showlocals "--durations=20" --cov-report html
--cov-report xml "--junitxml=junit/test-results.xml" "--cov=sktime_dl"
--pyargs)

if [[ "$TEST_SLOW" == "false" ]]; then
  TEST_ARGS+=("-m=not slow")
fi

# Print command before executing
set -o xtrace

# Run tests
"$TEST_CMD" "${TEST_ARGS[@]}" "../sktime_dl/"

set +o xtrace
set +e
