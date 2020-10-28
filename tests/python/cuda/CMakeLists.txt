# ==============================================================================
# tests/python/cuda/CMakeLists.txt

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )
    set( PRG_DIR  ${CMAKE_CURRENT_SOURCE_DIR} )
    set( WORK_DIR "${CMAKE_SOURCE_DIR}/python" )

    # ==========================================================================
    # test_cuda_controller.py:

    add_test( NAME Python_Cuda_CudaController
              COMMAND ${PYTHON_EXECUTABLE} ${PRG_DIR}/test_cuda_controller.py
              WORKING_DIRECTORY ${WORK_DIR} )


    # ==========================================================================
    # test_cuda_argument.py:

    add_test( NAME Python_Cuda_CudaArgument
              COMMAND ${PYTHON_EXECUTABLE} ${PRG_DIR}/test_cuda_argument.py
              WORKING_DIRECTORY ${WORK_DIR} )

    # ==========================================================================
    # test_track_job_track_line.py:

    add_test( NAME Python_Cuda_CudaTrackJob_TrackLine
              COMMAND ${PYTHON_EXECUTABLE} ${PRG_DIR}/test_track_job_track_line.py
              WORKING_DIRECTORY ${WORK_DIR} )


    # ==========================================================================
    # test_track_job_fetch_particle_addr.py:

    add_test( NAME Python_Cuda_CudaTrackJob_FetchParticleAddr
              COMMAND ${PYTHON_EXECUTABLE} ${PRG_DIR}/test_track_job_fetch_particle_addr.py
              WORKING_DIRECTORY ${WORK_DIR} )

    # ==========================================================================
    # test_track_job_collect_and_push_cuda.py:

    add_test( NAME Python_Cuda_CudaTrackJob_CollectPush
              COMMAND ${PYTHON_EXECUTABLE}
                      ${PRG_DIR}/test_track_job_collect_and_push_cuda.py
              WORKING_DIRECTORY ${WORK_DIR} )

    # ==========================================================================
    # test_track_job_collect_and_push_cuda.py:

    add_test( NAME Python_Cuda_CudaTrackJobThreadsPerBlockTests
              COMMAND ${PYTHON_EXECUTABLE}
                      ${PRG_DIR}/test_track_job_threads_per_block.py
              WORKING_DIRECTORY ${WORK_DIR} )

    # ==========================================================================
    # test_cuda_assign_address_item.py:
    add_test( NAME Python_Cuda_CudaTrackJob_AssignAddressItem
              COMMAND ${PYTHON_EXECUTABLE} ${PRG_DIR}/test_cuda_assign_address_items.py
              WORKING_DIRECTORY ${WORK_DIR} )
endif()
