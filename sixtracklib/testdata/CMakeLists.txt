set( SIXTRACKLIB_TESTDATA_DIR ${CMAKE_CURRENT_BINARY_DIR} )

configure_file( tracking_testfiles.h.template 
                "${CMAKE_CURRENT_SOURCE_DIR}/tracking_testfiles.h" 
                @ONLY )

set( SIXTRACKL_TESTDATA_LINK_LIBRARIES sixtrack ${SIXTRACKL_LINK_LIBRARIES} )
                
# -----------------------------------------------------------------------------
# ---- Generate testdata:

set( SIXTRACKL_TESTDATA_GENERATE_TRACKING_TESTDATA_HEADERS
     tracking_testfiles.h 
)

set( SIXTRACKL_TESTDATA_GENERATE_TRACKING_TESTDATA_SOURCES
     generate_testdata_tracking.cpp 
)

add_executable( generate_testdata_tracking 
                ${SIXTRACKL_TESTDATA_GENERATE_TRACKING_TESTDATA_HEADERS}
                ${SIXTRACKL_TESTDATA_GENERATE_TRACKING_TESTDATA_SOURCES} )
                
target_link_libraries( generate_testdata_tracking 
    ${SIXTRACKL_TESTDATA_LINK_LIBRARIES} )

set_property( TARGET generate_testdata_tracking 
              PROPERTY LINKER_LANGUAGE CXX )
    
set_property( TARGET generate_testdata_tracking 
              PROPERTY CXX_STANDARD 11 )
    
set_property( TARGET generate_testdata_tracking 
              PROPERTY CXX_STANDARD_REQUIRED ON )
    
set_property( TARGET generate_testdata_tracking 
    PROPERTY COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic )



