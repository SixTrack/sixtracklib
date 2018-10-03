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

set(  SIXTRACKL_LIBRARY_INTERFACE_SUBLIBRARIES
    ${SIXTRACKL_LIBRARY_INTERFACE_SUBLIBRARIES}
    sixtrack_common_be_cavity
    PARENT_SCOPE
)

# ----------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_COMMON_BE_CAVITY_INSTALL_PATH
     include/sixtracklib/common/be_cavity )

install( FILES ${SIXTRACKLIB_COMMON_BE_CAVITY_C99_HEADERS}
         DESTINATION ${SIXTRACKLIB_COMMON_BE_CAVITY_INSTALL_PATH} )

install( FILES ${SIXTRACKLIB_COMMON_BE_CAVITY_CXX_HEADERS}
         DESTINATION ${SIXTRACKLIB_COMMON_BE_CAVITY_INSTALL_PATH} )

#end: sixtracklib/common/be_cavity/CMakeLists.txt

