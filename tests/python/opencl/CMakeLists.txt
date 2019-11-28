# tests/python/opencl/CMakeLists.txt

# ------------------------------------------------------------------------------
# test_track_job_opencl:

add_test( NAME Python_OpenCL_TrackJob_InitTests
   COMMAND ${PYTHON_EXECUTABLE}
           ${CMAKE_CURRENT_SOURCE_DIR}/test_track_job_opencl.py
   WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )

# ------------------------------------------------------------------------------
# test_track_job_collect_and_push_opencl:

 add_test( NAME Python_OpenCL_TrackJob_CollectAndPushTests
    COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_track_job_collect_and_push_opencl.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )

# ------------------------------------------------------------------------------
# test_track_job_cl_stored_buffer_opencl:

 add_test( NAME Python_OpenCL_TrackJob_StoredBufferTests
    COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_track_job_cl_stored_buffer.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )
