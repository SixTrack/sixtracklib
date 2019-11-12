# tests/python/CMakeLists.txt

message( STATUS "---- processing tests/python/CMakeLists.txt" )

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )
    add_subdirectory( beam_elements )

    if( SIXTRACKL_ENABLE_CUDA )
        add_subdirectory( cuda )
    endif()

    if( SIXTRACKL_ENABLE_OPENCL )
        add_subdirectory( opencl )
    endif()

    # -------------------------------------------------------------------------
    # test_cbuffer_st_buffer:

    add_test( NAME Python_CBufferStBufferTests
        COMMAND ${PYTHON_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_cbuffer_st_buffer.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )

    # -------------------------------------------------------------------------
    # test_output_buffer:

    add_test( NAME Python_OutputBufferTests
        COMMAND ${PYTHON_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_output_buffer.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )

    # -------------------------------------------------------------------------
    # test_track_job_setup:

    add_test( NAME Python_TrackJobSetupTests
        COMMAND ${PYTHON_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_track_job_setup.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )

    # -------------------------------------------------------------------------
    # test_track_job_cpu:

    add_test( NAME Python_TrackJobCpuTests
        COMMAND ${PYTHON_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_track_job_cpu.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )

endif()

#end: tests/python/CMakeLists.txt
