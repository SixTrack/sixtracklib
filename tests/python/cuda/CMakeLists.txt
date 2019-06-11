# ==============================================================================
# tests/python/cuda/CMakeLists.txt

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )

    # ==========================================================================
    # test_cuda_controller.py:

    add_test( NAME Python_Cuda_CudaControllerTests
              COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_controller.py
              WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )

endif()

#end: tests/python/cuda/CMakeLists.txt
