# Tests

To run unit tests install prereqiusites and
execute the following command from the root of the repo: 

```bash
pip install -e .[dev]
make test
```

Consider installing VizDoom for a more comprehensive set of tests.

These tests are executed after each commit/PR by Github Actions. 

## Test CI based on Github Actions
We build a test ci system based on Github Actions which will automatically run unit tests on different os(currently linux and macos) with different python version(currently 3.8, 3.9, 3.10) when you push or merge your commits.

The test workflow is defined in .github/workflows/test-ci.yml.

There's one thing noticeble. We add a Pre-check section before we formally run all the unit tests. The Pre-check section is used to make sure the torch multiprocessing memory sharing and pbt-based environments work as expected. The reason we have this Pre-check is that sometimes the running environment provided by Github Actions is unstable(mostly likely has low limits for memory) and fails our multi-policies and pbt tests.
