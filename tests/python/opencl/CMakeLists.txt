# tests/python/opencl/CMakeLists.txt

# ------------------------------------------------------------------------------
# test_track_job_opencl:

 add_test( NAME Python_TrackJobOpenClTests
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_track_job_opencl.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )

