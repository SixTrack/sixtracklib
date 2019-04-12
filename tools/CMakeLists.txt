# tools/CMakeLists.txt

set( TOOLS_C99_TARGETS )
set( TOOLS_CXX_TARGETS )

set( TOOLS_LINK_LIBRARIES sixtrack_test sixtrack )

# ============================================================================
# Tools:

# ----------------------------------------------------------------------------
# normalize_cobject_dump:

set( TOOLS_NORMALIZE_COBJECT_DUMP_HEADERS )
set( TOOLS_NORMALIZE_COBJECT_DUMP_SOURCES normalize_cobject_dump.c )

add_executable( normalize_cobject_dump
                ${TOOLS_NORMALIZE_COBJECT_DUMP_HEADERS}
                ${TOOLS_NORMALIZE_COBJECT_DUMP_SOURCES}
)

list( APPEND TOOLS_C99_TARGETS normalize_cobject_dump )


# ----------------------------------------------------------------------------
# dump_cbuffer:

set( TOOLS_DUMP_CBUFFER_HEADERS )
set( TOOLS_DUMP_CBUFFER_SOURCES dump_cbuffer.c )

add_executable(
    dump_cbuffer ${TOOLS_DUMP_CBUFFER_HEADERS} ${TOOLS_DUMP_CBUFFER_SOURCES} )

list( APPEND TOOLS_C99_TARGETS dump_cbuffer )

# ============================================================================
# set properties for all tools:

if( TOOLS_C99_TARGETS )
    set(  TOOLS_COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} )
    list( APPEND TOOLS_COMPILE_OPTIONS -Wall )
    list( APPEND TOOLS_COMPILE_OPTIONS -Werror )
    list( APPEND TOOLS_COMPILE_OPTIONS -pedantic )

    set_property( TARGET ${TOOLS_C99_TARGETS} PROPERTY LINKER_LANGUAGE C )
    set_property( TARGET ${TOOLS_C99_TARGETS} PROPERTY C_STANDARD 99 )
    set_property( TARGET ${TOOLS_C99_TARGETS} PROPERTY C_STANDARD_REQUIRED ON )
    set_property( TARGET ${TOOLS_C99_TARGETS}
                  APPEND PROPERTY COMPILE_OPTIONS ${TOOLS_COMPILE_OPTIONS} )

    set_property( TARGET ${TOOLS_C99_TARGETS} PROPERTY
                  LINK_LIBRARIES ${TOOLS_LINK_LIBRARIES} )
endif()

# end: tools/CMakeLists.txt
