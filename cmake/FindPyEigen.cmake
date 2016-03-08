# find eigen.so

string(TOUPPER eigen module_upper)
        execute_process(COMMAND "${PYTHON_EXEC}" "-c" 
                "import re, eigen; print re.compile('/__init__.py.*').sub('',eigen.__file__)"
                RESULT_VARIABLE _eigen_status 
                OUTPUT_VARIABLE _eigen_location
                ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT _eigen_status)
                set(PY_EIGEN ${_eigen_location} CACHE STRING 
                        "Location of Python module eigen")
        endif(NOT _eigen_status)
endif(NOT PY_EIGEN)
find_package_handle_standard_args(PY_eigen DEFAULT_MSG PY_EIGEN)