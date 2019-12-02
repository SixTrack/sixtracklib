# tests/python/opencl/CMakeLists.txt

# ------------------------------------------------------------------------------
# test_cl_node_id_opencl:

add_test( NAME Python_OpenCL_ClNodeIdTests
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_cl_node_id.py
    WORKING_DIRECTORY ${PYTHON_TESTS_WORKDIR} )

# ------------------------------------------------------------------------------
# test_cl_controller_opencl:

add_test( NAME Python_OpenCL_ClControllerTests
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_cl_controller.py
    WORKING_DIRECTORY ${PYTHON_TESTS_WORKDIR} )

# ------------------------------------------------------------------------------
# test_track_job_opencl:

add_test( NAME Python_OpenCL_TrackJob_InitTests
   COMMAND ${PYTHON_EXECUTABLE}
           ${CMAKE_CURRENT_SOURCE_DIR}/test_track_job_opencl.py
   WORKING_DIRECTORY ${PYTHON_TESTS_WORKDIR} )

# ------------------------------------------------------------------------------
# test_track_job_collect_and_push_opencl:

 add_test( NAME Python_OpenCL_TrackJob_CollectAndPushTests
    COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_track_job_collect_and_push_opencl.py
    WORKING_DIRECTORY ${PYTHON_TESTS_WORKDIR} )

# ------------------------------------------------------------------------------
# test_track_job_cl_stored_buffer_opencl:

 add_test( NAME Python_OpenCL_TrackJob_StoredBufferTests
    COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_track_job_cl_stored_buffer.py
    WORKING_DIRECTORY ${PYTHON_TESTS_WORKDIR} )
