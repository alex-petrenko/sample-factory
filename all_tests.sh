#!/bin/bash

python -m unittest
STATUS_ALL_TESTS=$?

echo "Your terminal might be unresponsive (caused by Doom envs manipulating the console). Type reset in your terminal and press Enter to get it back to normal"
echo "Unit tests status (0 = Success): $STATUS_ALL_TESTS"
