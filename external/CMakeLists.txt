# ============================================================================
# external/CMakeLists.txt

message( STATUS "---- processing external/CMakeLists.txt" )

add_subdirectory( toml11 )

if( SIXTRACKL_ENABLE_OPENCL )
    add_subdirectory( CL )
endif()
