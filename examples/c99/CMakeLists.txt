# examples/c99/CMakeLists.txt

set( EXAMPLE_PROGRAMS )
set( EXAMPLE_LINK_LIBRARIES sixtrack_test sixtrack m dl )

# -----------------------------------------------------------------------------
# simple_drift:

add_executable( simple_drift simple_drift.c )
set( EXAMPLE_PROGRAMS ${EXAMPLE_PROGRAMS} simple_drift )

target_compile_options( simple_drift PRIVATE
    -fno-fast-math -fno-tree-vectorize )

# -----------------------------------------------------------------------------
# simple_drift_autovec:

if( SIXTRACKL_ENABLE_AUTOVECTORIZATION )
    add_executable( simple_drift_autovec simple_drift.c )
    set( EXAMPLE_PROGRAMS ${EXAMPLE_PROGRAMS} simple_drift_autovec )

    target_compile_options( simple_drift_autovec PRIVATE
        -ftree-vectorize -ftree-vectorizer-verbose=6 -fopt-info-loop
        -fno-fast-math --param vect-max-version-for-alias-checks=50 )
endif()

# ==============================================================================
# Common settings for all programs:

set(  EXAMPLE_COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} )
list( APPEND EXAMPLE_COMPILE_OPTIONS -Wall )
list( APPEND EXAMPLE_COMPILE_OPTIONS -Werror )
list( APPEND EXAMPLE_COMPILE_OPTIONS -pedantic )

set_property( TARGET ${EXAMPLE_PROGRAMS} PROPERTY LINKER_LANGUAGE C )
set_property( TARGET ${EXAMPLE_PROGRAMS} PROPERTY C_STANDARD 99 )
set_property( TARGET ${EXAMPLE_PROGRAMS} PROPERTY C_STANDARD_REQUIRED ON )
set_property( TARGET ${EXAMPLE_PROGRAMS}
              APPEND PROPERTY COMPILE_OPTIONS ${EXAMPLE_COMPILE_OPTIONS} )

set_property( TARGET ${EXAMPLE_PROGRAMS}
              PROPERTY LINK_LIBRARIES ${EXAMPLE_LINK_LIBRARIES} )

# end: examples/c99/CMakeLists.txt
