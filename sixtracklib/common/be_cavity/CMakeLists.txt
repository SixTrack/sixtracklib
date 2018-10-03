set( SIXTRACKLIB_COMMON_BE_CAVITY_C99_HEADERS
     be_cavity.h
)

set( SIXTRACKLIB_COMMON_BE_CAVITY_CXX_HEADERS
     be_cavity.hpp
)

add_library( sixtrack_common_be_cavity INTERFACE
             ${SIXTRACKLIB_COMMON_BE_CAVITY_C99_HEADERS}
             ${SIXTRACKLIB_COMMON_BE_CAVITY_CXX_HEADERS}
)

if( SIXTRACKLIB_COMMON_BE_CAVITY_C99_HEADERS )
    string( "([^;]+)" "be_cavity/\\1" LOCAL_C99_HEADERS
            "${SIXTRACKLIB_COMMON_BE_CAVITY_C99_HEADERS}" )

    set(  SIXTRACKLIB_COMMON_C99_HEADERS
        ${SIXTRACKLIB_COMMON_C99_HEADERS}
        ${LOCAL_C99_HEADERS} PARENT_SCOPE )
endif()

if( SIXTRACKLIB_COMMON_BE_CAVITY_CXX_HEADERS )
    string( "([^;]+)" "be_cavity/\\1" LOCAL_CXX_HEADERS
            "${SIXTRACKLIB_COMMON_BE_CAVITY_CXX_HEADERS}" )

    set(  SIXTRACKLIB_COMMON_BE_CAVITY_CXX_HEADERS
        ${SIXTRACKLIB_COMMON_BE_CAVITY_CXX_HEADERS}
        ${LOCAL_CXX_HEADERS} PARENT_SCOPE )
endif()

# ----------------------------------------------------------------------------
# install :

if( SIXTRACKLIB_COMMON_BE_CAVITY_C99_HEADERS )

    set( SIXTRACKLIB_COMMON_BE_CAVITY_C99_INSTALL_PATH
         include/sixtracklib/common/be_cavity )

    install( FILES ${SIXTRACKLIB_COMMON_BE_CAVITY_C99_HEADERS}
             DESTINATION ${SIXTRACKLIB_COMMON_BE_CAVITY_C99_INSTALL_PATH} )
endif()


if( SIXTRACKLIB_COMMON_BE_CAVITY_CXX_HEADERS )

    set( SIXTRACKLIB_COMMON_BE_CAVITY_CXX_INSTALL_PATH
         include/sixtracklib/common/be_cavity )

    install( FILES ${SIXTRACKLIB_COMMON_BE_CAVITY_CXX_HEADERS}
             DESTINATION ${SIXTRACKLIB_COMMON_BE_CAVITY_CXX_INSTALL_PATH} )
endif()

#end: sixtracklib/common/be_cavity/CMakeLists.txt
