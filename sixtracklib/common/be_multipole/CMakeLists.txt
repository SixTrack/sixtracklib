set( SIXTRACKLIB_COMMON_BE_MULTIPOLE_C99_HEADERS
     be_multipole.h
)

set( SIXTRACKLIB_COMMON_BE_MULTIPOLE_CXX_HEADERS
     be_multipole.hpp
)

add_library( sixtrack_common_be_multipole INTERFACE
             ${SIXTRACKLIB_COMMON_BE_MULTIPOLE_C99_HEADERS}
             ${SIXTRACKLIB_COMMON_BE_MULTIPOLE_CXX_HEADERS}
)

set(  SIXTRACKL_LIBRARY_INTERFACE_SUBLIBRARIES
    ${SIXTRACKL_LIBRARY_INTERFACE_SUBLIBRARIES}
    sixtrack_common_be_multipole
    PARENT_SCOPE
)

# ----------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_COMMON_BE_MULTIPOLE_INSTALL_PATH
     include/sixtracklib/common/be_multipole )

install( FILES ${SIXTRACKLIB_COMMON_BE_MULTIPOLE_C99_HEADERS}
         DESTINATION ${SIXTRACKLIB_COMMON_BE_MULTIPOLE_INSTALL_PATH} )

install( FILES ${SIXTRACKLIB_COMMON_BE_MULTIPOLE_CXX_HEADERS}
         DESTINATION ${SIXTRACKLIB_COMMON_BE_MULTIPOLE_INSTALL_PATH} )

#end: sixtracklib/common/be_multipole/CMakeLists.txt


