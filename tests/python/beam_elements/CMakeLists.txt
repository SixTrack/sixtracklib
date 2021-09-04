add_subdirectory( beamfields )

add_test( NAME Python_BeamElements_MultiPole
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_multipole.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )

add_test( NAME Python_BeamElements_TriCub
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_tricub.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )
