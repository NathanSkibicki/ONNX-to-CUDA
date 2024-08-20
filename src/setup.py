from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "sigmoid",
        ["sigmoid.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++"
    )
]

setup(
    name="sigmoid",
    ext_modules=cythonize(extensions),
)