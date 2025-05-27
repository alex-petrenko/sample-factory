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

We build a test CI system based on Github Actions which will automatically run unit tests on different operating systems
(currently Linux and macOS) with different python versions (currently 3.8, 3.9, 3.10) when you submit PRs or merge to the main branch.

The test workflow is defined in `.github/workflows/test-ci.yml`.

[//]: # (There's one thing noticeble. We add a Pre-check section before we formally run all the unit tests. The Pre-check section is used to make sure the torch multiprocessing memory sharing and pbt-based environments work as expected. The reason we have this Pre-check is that sometimes the running environment provided by Github Actions is unstable &#40;mostly likely has low limits for memory&#41; and fails our multi-policies and pbt tests.)

### self-hosted runners
Github Actions supports self-hosted runners for more powerful hardware resources or customized needs. You need to register your runner(physical, virtual, in a container, on-premises, or in a cloud) to Github and define your workflows.

- Step 1: register your runner
  
  Basically, go to your repository, then go to "Settings", then go to Actions, then go to Runners, then click "New self-hosted runner", then choose the os type and architecture type, then execute the corresponding commands on your runner. You can find concrete steps here: https://docs.github.com/en/actions/hosting-your-own-runners/adding-self-hosted-runners

- Step 2: define your workflows
  
  Here is an example workflow config for a Google Cloud VM instance(n1-standard-8, Intel Haswell, NVIDIA Tesla P4). 

```yml
name: tests

on:
  push:
    branches:
    - self-hosted-runner

jobs:
  run-tests-self-runner:
    strategy:
      fail-fast: false
      matrix:
        python-version: ['3.8']
    runs-on: self-hosted
    steps:
    - uses: actions/checkout@v3
    - name: Create conda env
      run: |
        which conda
        conda env list
        conda env remove -n self-hosted-runner
        conda create -n self-hosted-runner python=${{ matrix.python-version }}
    - name: Add conda to system path
      run: |
        echo "/opt/conda/envs/self-hosted-runner/bin" >> $GITHUB_PATH
    - name: Install dependencies
      run: |
        pwd
        pip install -e '.[atari, mujoco, envpool]'
        nvcc -V
        pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
        conda list
    - name: Install test dependencies
      run: |
        pip install pytest
    - name: Run Pre-check
      uses: nick-fields/retry@v2
      with:
        timeout_minutes: 10
        max_attempts: 5
        command: pytest -s tests/test_precheck.py & pytest -s tests/algo/test_pbt.py
    - name: Run tests
      run: |
        # run all tests
        pytest -s -k "not torch_tensor_share"
```

If you wanna run it for scheduled times, you can change the trigger event to "schedule". For example, the following setting triggers the workflow every day at 5:30 and 17:30 UTC:

```yml
on:
  schedule:
    - cron:  '30 5,17 * * *'
```
