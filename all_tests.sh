#!/bin/bash

python -m unittest
STATUS_ALL_TESTS=$?

python -m unittest sample_factory_examples.mujoco_examples.test_mujoco
STATUS_TEST_MUJOCO=$?

python -m unittest sample_factory_examples.mujoco_examples.test_mujoco
STATUS_TEST_MUJOCO=$?

echo "Your terminal might be unresponsive (caused by Doom envs manipulating the console). Type reset in your terminal and press Enter to get it back to normal"
echo "Unit tests status (0 = Success): $STATUS_ALL_TESTS"
<<<<<<< HEAD
echo "Custom env status (0 = Success): $STATUS_TEST_EXAMPLE"
echo "Custom multi env status (0 = Success): $STATUS_TEST_EXAMPLE_MULTI"
=======
>>>>>>> sf2
echo "Mujoco status (0 = Success): $STATUS_TEST_MUJOCO"
