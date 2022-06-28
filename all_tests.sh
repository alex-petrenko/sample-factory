#!/bin/bash

python -m unittest
STATUS_ALL_TESTS=$?

echo "Unit tests status (0 = Success): $STATUS_ALL_TESTS"
