cmake_minimum_required(VERSION 2.6)
project(eigpy)
ADD_DEFINITIONS(-std=c++11)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
SET(CMAKE_BUILD_TYPE REALEASE)
SET(CMAKE_CXX_FLAGS_REALEASE "-O2")

set(Python_ADDITIONAL_VERSIONS ${py})
find_package(PythonInterp ${py})
find_package(PythonLibs ${py})

find_package(Eigen3)


include_directories( ${PYTHON_INCLUDE_DIRS})
include_directories( ${EIGEN3_INCLUDE_DIR})


PYTHON_ADD_MODULE(eigen SHARED python_eigen.cpp)

configure_file(test.py test.py COPYONLY)

EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c 
  "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
  OUTPUT_VARIABLE python_libs OUTPUT_STRIP_TRAILING_WHITESPACE )

message(${PYTHON_INCLUDE_DIRS})
message(${PYTHON_EXECUTABLE})
message(${python_libs})
install(TARGETS "eigen" DESTINATION ${python_libs})