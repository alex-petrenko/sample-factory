from setuptools import setup

setup(
    # Information
    name='sample-factory',
    version='1.0.0',
    url='https://github.com/alex-petrenko/sample-factory',
    author='Aleksei Petrenko',
    license='MIT',
    keywords='asynchronous reinforcement learning policy gradient ppo appo impala',

    # Build instructions
    setup_requires=['setuptools>=45.2.0', 'cython>=0.29'],

    # these requirements are untested and incomplete. Follow README.md to properly setup the environment.
    # Full set of tested requirements is in environment.yml
    install_requires=[
        'numpy>=1.18.1',
        'torch>=1.4.0',
        'gym>=0.17.1',
        'tensorboard>=1.15.0',
        'tensorboardx>=2.0',
        'psutil>=5.7.0',
        'threadpoolctl>=2.0.0',
        'colorlog',
        'faster-fifo',
        'filelock',
    ]
)
