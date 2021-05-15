#!/bin/bash

SKIP_TESTS_THAT_REQUIRE_A_SEPARATE_PROCESS=1 python -m unittest
STATUS_ALL_TESTS=$?

python -m unittest sample_factory_examples.tests.test_example
STATUS_TEST_EXAMPLE=$?

python -m unittest sample_factory_examples.tests.test_example_multi
STATUS_TEST_EXAMPLE_MULTI=$?

echo "Your terminal might be unresponsive (caused by Doom envs manipulating the console). Type reset in your terminal and press Enter to get it back to normal"
echo "Unit tests status (0 = Success): $STATUS_ALL_TESTS"
echo "Custom env status (0 = Success): $STATUS_TEST_EXAMPLE"
echo "Custom multi env status (0 = Success): $STATUS_TEST_EXAMPLE_MULTI"
