
set( SIXTRACKLIB_COMMON_BE_XYSHIFT_C99_HEADERS
     be_xyshift.h
)

set( SIXTRACKLIB_COMMON_BE_XYSHIFT_CXX_HEADERS
     be_xyshift.hpp
)

add_library( sixtrack_common_be_xyshift INTERFACE
             ${SIXTRACKLIB_COMMON_BE_XYSHIFT_C99_HEADERS}
             ${SIXTRACKLIB_COMMON_BE_XYSHIFT_CXX_HEADERS}
)

set(  SIXTRACKL_LIBRARY_INTERFACE_SUBLIBRARIES
    ${SIXTRACKL_LIBRARY_INTERFACE_SUBLIBRARIES}
    sixtrack_common_be_xyshift
    PARENT_SCOPE
)

# ----------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_COMMON_BE_XYSHIFT_INSTALL_PATH
     include/sixtracklib/common/be_xyshift )

install( FILES ${SIXTRACKLIB_COMMON_BE_XYSHIFT_C99_HEADERS}
         DESTINATION ${SIXTRACKLIB_COMMON_BE_XYSHIFT_INSTALL_PATH} )

install( FILES ${SIXTRACKLIB_COMMON_BE_XYSHIFT_CXX_HEADERS}
         DESTINATION ${SIXTRACKLIB_COMMON_BE_XYSHIFT_INSTALL_PATH} )

#end: sixtracklib/common/be_xyshift/CMakeLists.txt
