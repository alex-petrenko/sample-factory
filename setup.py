import setuptools
from setuptools import setup


with open('README.md', 'r') as f:
    long_description = f.read()
    descr_lines = long_description.split('\n')
    descr_no_gifs = []  # gifs are not supported on PyPI web page
    for dl in descr_lines:
        if not ('<img src=' in dl and 'gif' in dl):
            descr_no_gifs.append(dl)

    long_description = '\n'.join(descr_no_gifs)


setup(
    # Information
    name='sample-factory',
    description='High throughput asynchronous reinforcement learning framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='1.121.3',
    url='https://github.com/alex-petrenko/sample-factory',
    author='Aleksei Petrenko',
    license='MIT',
    keywords='asynchronous reinforcement learning policy gradient ppo appo impala ai',
    project_urls={
        'Github': 'https://github.com/alex-petrenko/sample-factory',
        'Videos': 'https://sites.google.com/view/sample-factory',
    },

    # might want to add max versions restrictions, i.e. torch < 2?
    install_requires=[
        'numpy>=1.18.1',
        'torch>=1.6',
        'gym>=0.17.1',
        'tensorboard>=1.15.0',
        'tensorboardx>=2.0',
        'psutil>=5.7.0',
        'threadpoolctl>=2.0.0',
        'colorlog',
        'faster-fifo>=1.2.0',
        'filelock',
        'opencv-python',
    ],

    package_dir={'': './'},
    packages=setuptools.find_packages(where='./', include='sample_factory*'),
    include_package_data=True,

    python_requires='>=3.7',
)
