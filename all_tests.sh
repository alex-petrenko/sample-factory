#!/bin/bash

python -m unittest
STATUS_ALL_TESTS=$?

python -m unittest sample_factory_examples.mujoco_examples.test_mujoco
STATUS_TEST_MUJOCO=$?

echo "Your terminal might be unresponsive (caused by Doom envs manipulating the console). Type reset in your terminal and press Enter to get it back to normal"
echo "Unit tests status (0 = Success): $STATUS_ALL_TESTS"
echo "Mujoco status (0 = Success): $STATUS_TEST_MUJOCO"
