from distutils.core import setup, Extension
from distutils import sysconfig
import sys

include_dirs = ["/usr/include/eigen3/"]

setup(name='paraEigen',
      version='0.0.1',
      author='Loooo',
      author_email='sppedflyer@gmail.com',
      url="https://github.com/looooo/py-eigen",
      description='Wrap parts of Eigen3 using pybind11',
      ext_modules=[Extension('paraEigen',
                   sources=['python_eigen.cpp'],
                   include_dirs=include_dirs,
                   extra_compile_args=['-std=c++11'])])
