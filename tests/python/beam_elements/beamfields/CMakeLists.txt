add_test( NAME Python_BeamElements_BeamFields_Faddeeva_CERNLib
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_cernlib.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )

add_test( NAME Python_BeamElements_BeamFields_Faddeeva_Alg680
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_alg680.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )

