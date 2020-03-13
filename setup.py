from Cython.Build import cythonize
from setuptools import setup, Extension


extensions = [
    Extension(
        name='fast_queue',
        sources=['fast_queue.pyx', 'cpp_fast_queue/cpp_lib/fast_queue.cpp'],
        language='c++',
        include_dirs=['cpp_fast_queue/cpp_lib'],
    ),
]

setup(
    # Information
    name='sample-factory',
    version='0.0.1',
    url='https://github.com/alex-petrenko/sample-factory',
    author='Aleksei Petrenko',
    license='MIT',
    keywords='asynchronous reinforcement learning policy gradient ppo impala',

    # Build instructions
    ext_modules=cythonize(extensions),

    setup_requires=['setuptools>=45.2.0', 'cython>=0.29'],
)
