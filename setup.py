import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

__version__ = '0.1.6'


ext_modules = cythonize(
    Extension(
        "csimulate",
        sources=["nplocate/csimulate.pyx"],
        include_dirs=[numpy.get_include()]
    )
)

ext_modules += cythonize(
    Extension(
        "cutility",
        sources=["nplocate/cutility.pyx"],
        include_dirs=[numpy.get_include()]
    )
)


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='nplocate',
    version=__version__,
    ext_modules=ext_modules,
    packages=["nplocate"],
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
    install_requires=[
        'setuptools', 'wheel', 'numpy', 'scipy', 'matplotlib', 'cython'
    ],
)
