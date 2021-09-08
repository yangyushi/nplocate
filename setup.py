import numpy
from setuptools import setup, find_packages
from Cython.Build import cythonize

__version__ = '0.2.7'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='nplocate',
    version=__version__,
    ext_modules=cythonize(["nplocate/csimulate.pyx", "nplocate/cutility.pyx"]),
    packages=find_packages(),
    include_dirs=numpy.get_include(),
    author='Yushi Yang',
    author_email='yangyushi1992@icloud.com',
    url='https://github.com/yangyushi/nplocate',
    description='Python tools to locate nano particles from confocal microscope images',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
    setup_requires=["cython", "numpy"],
    install_requires=[
        'setuptools', 'wheel', 'numpy', 'scipy', 'matplotlib', 'cython'
    ],
)
