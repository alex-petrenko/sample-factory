#!/bin/bash

SKIP_TESTS_THAT_REQUIRE_A_SEPARATE_PROCESS=1 python -m unittest
STATUS_ALL_TESTS=$?

python -m unittest examples.tests.test_example
STATUS_TEST_EXAMPLE=$?

echo "Your terminal might be unresponsive (caused by Doom envs manipulating the console). Type reset in your terminal and press Enter to get it back to normal"
echo "Status: $STATUS_ALL_TESTS $STATUS_TEST_EXAMPLE"
