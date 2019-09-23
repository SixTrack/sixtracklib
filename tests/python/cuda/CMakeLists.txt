# ==============================================================================
# tests/python/cuda/CMakeLists.txt

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )
    set( PRG_DIR  ${CMAKE_CURRENT_SOURCE_DIR} )
    set( WORK_DIR "${CMAKE_SOURCE_DIR}/python" )

    # ==========================================================================
    # test_cuda_controller.py:

    add_test( NAME Python_Cuda_CudaControllerTests
              COMMAND ${PYTHON_EXECUTABLE} ${PRG_DIR}/test_cuda_controller.py
              WORKING_DIRECTORY ${WORK_DIR} )


    # ==========================================================================
    # test_cuda_argument.py:

    add_test( NAME Python_Cuda_CudaArgumentTests
              COMMAND ${PYTHON_EXECUTABLE} ${PRG_DIR}/test_cuda_argument.py
              WORKING_DIRECTORY ${WORK_DIR} )

    # ==========================================================================
    # test_track_job_track_line.py:

    add_test( NAME Python_Cuda_CudaTrackJobTrackLineTests
              COMMAND ${PYTHON_EXECUTABLE} ${PRG_DIR}/test_track_job_track_line.py
              WORKING_DIRECTORY ${WORK_DIR} )


    # ==========================================================================
    # test_track_job_fetch_particle_addr.py:

    add_test( NAME Python_Cuda_CudaTrackJobFetchParticleAddrTests
              COMMAND ${PYTHON_EXECUTABLE} ${PRG_DIR}/test_track_job_fetch_particle_addr.py
              WORKING_DIRECTORY ${WORK_DIR} )

    # ==========================================================================
    # test_track_job_collect_and_push_cuda.py:

    add_test( NAME Python_Cuda_CudaTrackJobCollectPushTests
              COMMAND ${PYTHON_EXECUTABLE}
                      ${PRG_DIR}/test_track_job_collect_and_push_cuda.py
              WORKING_DIRECTORY ${WORK_DIR} )

endif()

#end: tests/python/cuda/CMakeLists.txt
