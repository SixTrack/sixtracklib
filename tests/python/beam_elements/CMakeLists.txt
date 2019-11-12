# -------------------------------------------------------------------------
# test_multipole:

add_test( NAME Python_BeamElementsMultiPole
    COMMAND ${PYTHON_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_multipole.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )

