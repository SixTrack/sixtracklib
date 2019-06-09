# ==============================================================================
# tests/python/cuda/CMakeLists.txt

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )

    # ==========================================================================
    # CudaArgument:

    add_test( NAME Python_Cuda_ArgumentTests
        COMMAND ${PYTHON_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_argument.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )

endif()

#end: tests/python/cuda/CMakeLists.txt
