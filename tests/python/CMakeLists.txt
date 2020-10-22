# tests/python/CMakeLists.txt

message( STATUS "---- processing tests/python/CMakeLists.txt" )

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )
    set( PYTHON_TESTS_WORKDIR ${CMAKE_SOURCE_DIR}/python )

    add_subdirectory( beam_elements )

    if( SIXTRACKL_ENABLE_CUDA )
        add_subdirectory( cuda )
    endif()

    if( SIXTRACKL_ENABLE_OPENCL )
        add_subdirectory( opencl )
    endif()

    # -------------------------------------------------------------------------
    # test_cbuffer_st_buffer:

    add_test( NAME Python_Common_Buffer_StBuffer
        COMMAND ${PYTHON_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_cbuffer_st_buffer.py
        WORKING_DIRECTORY ${PYTHON_TESTS_WORKDIR} )

    # -------------------------------------------------------------------------
    # test_output_buffer:

    add_test( NAME Python_Common_Output_OutputBuffer
        COMMAND ${PYTHON_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_output_buffer.py
        WORKING_DIRECTORY ${PYTHON_TESTS_WORKDIR} )

    # -------------------------------------------------------------------------
    # test_track_job_setup:

    add_test( NAME Python_CPU_TrackJob_Setup
        COMMAND ${PYTHON_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_track_job_setup.py
        WORKING_DIRECTORY ${PYTHON_TESTS_WORKDIR} )

    # -------------------------------------------------------------------------
    # test_track_job_cpu:

    add_test( NAME Python_CPU_TrackJob_Init
        COMMAND ${PYTHON_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_track_job_cpu.py
        WORKING_DIRECTORY ${PYTHON_TESTS_WORKDIR} )

    # -------------------------------------------------------------------------
    # test_track_job_cpu_stored_buffer:

    add_test( NAME Python_CPU_TrackJob_StoredBuffer
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_track_job_cpu_stored_buffer.py
        WORKING_DIRECTORY ${PYTHON_TESTS_WORKDIR} )

    # -------------------------------------------------------------------------
    # test_track_job_cpu_assign_address_items:

    add_test( NAME Python_CPU_TrackJob_AssignAddressItems
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_track_job_cpu_assign_address_items.py
        WORKING_DIRECTORY ${PYTHON_TESTS_WORKDIR} )


    # -------------------------------------------------------------------------
    # test_qgauss:

    add_test( NAME Python_Math_QGauss
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_qgauss.py
        WORKING_DIRECTORY ${PYTHON_TESTS_WORKDIR} )

endif()

#end: tests/python/CMakeLists.txt
