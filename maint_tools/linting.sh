#!/bin/bash

# author: Markus Löning
# code quality check using flake8

set -e -x
set -o pipefail

if ! flake8 --verbose --filename=*.py sktime_dl/; then
  echo 'Linting failed.'
  # uncomment to make CI fail when linting fails
  exit 1
fi
