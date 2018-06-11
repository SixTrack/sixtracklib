# -----------------------------------------------------------------------------
# simple_drift.c -> simple_drift: 

add_executable( simple_drift simple_drift.c )
target_include_directories( simple_drift PRIVATE ${CMAKE_SOURCE_DIR} )
target_link_libraries( simple_drift sixtrack m )

target_compile_options( simple_drift BEFORE PRIVATE 
    -g -fno-tree-vectorize -fopt-info-loop -fno-fast-math )
        
set_target_properties( simple_drift PROPERTIES LINKER_LANGUAGE C )
set_target_properties( simple_drift PROPERTIES C_STANDARD 99 )

# -----------------------------------------------------------------------------
# simple_drift.c -> simple_drift_autovec : 

if( SIXTRACKL_ENABLE_AUTOVECTORIZATION )

    message( STATUS "---- Building autovectorization examples ...." )
    
    set( SIXTRACKL_EXAMPLE_AUTOVEC_CPU_ARCH )
    
    if(  SIXTRACKL_CPU_ARCH MATCHES "avx" )
        message( STATUS "---- -> Build for AVX architecture" )
        set( SIXTRACKL_EXAMPLE_AUTOVEC_CPU_ARCH -mavx )
        
    elseif( SIXTRACKL_CPU_ARCH MATCHES "sse2" )
        message( STATUS "---- -> Build for SSE2 architecture" )
        set( SIXTRACKL_EXAMPLE_AUTOVEC_CPU_ARCH -msse2 )
        
    elseif( SIXTRACKL_CPU_ARCH MATCHES "native" )
        message( STATUS "---- -> Build for native architecture" )
        set( SIXTRACKL_EXAMPLE_AUTOVEC_CPU_ARCH -march=native )
    else()
        message( STATUS "---- -> Build for default sse architecture" )
        set( SIXTRACKL_EXAMPLE_AUTOVEC_CPU_ARCH -msse )        
    endif()
        
    add_executable( simple_drift_autovec simple_drift.c )
    target_include_directories( simple_drift_autovec PRIVATE ${CMAKE_SOURCE_DIR} )
    target_link_libraries( simple_drift_autovec sixtrack m )

    target_compile_options( simple_drift_autovec BEFORE PRIVATE 
        -g -ftree-vectorize -ftree-vectorizer-verbose=6 -fopt-info-loop 
        -fno-fast-math --param vect-max-version-for-alias-checks=50 
        ${SIXTRACKL_EXAMPLE_AUTOVEC_CPU_ARCH} )
            
    set_target_properties( simple_drift_autovec PROPERTIES LINKER_LANGUAGE C )
    set_target_properties( simple_drift_autovec PROPERTIES C_STANDARD 99 )
    
endif()

# ------------------------------------------------------------------------------
#simple_drift_simd.c -> simple_drift_simd :

if( SIXTRACKL_ENABLE_MANUAL_SIMD )
    
    set( SIXTRACKL_EXAMPLE_SIMD_CPU_ARCH )
    
    message( STATUS "Building manual SIMD vectorization example .... " )
    if(  SIXTRACKL_CPU_ARCH MATCHES "avx" )
        message( STATUS "---- -> Build for AVX architecture" )
        set( SIXTRACKL_EXAMPLE_SIMD_CPU_ARCH -mavx )
        
    elseif( SIXTRACKL_CPU_ARCH MATCHES "sse2" )
        message( STATUS "---- -> Build for SSE2 architecture" )
        set( SIXTRACKL_EXAMPLE_SIMD_CPU_ARCH -msse2 )
        
    elseif( SIXTRACKL_CPU_ARCH MATCHES "native" )
        message( STATUS "---- -> Build for native architecture" )
        set( SIXTRACKL_EXAMPLE_SIMD_CPU_ARCH -march=native )
    else()
        message( STATUS "---- -> Build for default sse architecture" )
        set( SIXTRACKL_EXAMPLE_SIMD_CPU_ARCH -msse )        
    endif()
        
    add_executable( simple_drift_simd simple_drift_simd.c )
    target_include_directories( simple_drift_simd PRIVATE ${CMAKE_SOURCE_DIR} )
    target_link_libraries( simple_drift_simd sixtrack m )

    target_compile_options( simple_drift_simd BEFORE PRIVATE 
       -g -fno-fast-math ${SIXTRACKL_EXAMPLE_SIMD_CPU_ARCH} -Wall -Werror -pedantic -ansi )
            
    set_target_properties( simple_drift_simd PROPERTIES LINKER_LANGUAGE C )
    set_target_properties( simple_drift_simd PROPERTIES C_STANDARD 99 )
    

endif()

# ------------------------------------------------------------------------------
# simple_drift_opencl.c -> simple_drift_opencl :

if( SIXTRACKL_ENABLE_OPENCL )

    message( STATUS "Building OpenCL example .... " )
    
    add_executable( simple_drift_opencl simple_drift_opencl.c )
    
    target_include_directories( simple_drift_opencl PRIVATE ${CMAKE_SOURCE_DIR} )
    target_link_libraries( simple_drift_opencl sixtrack m 
                           ${SIXTRACKL_OPENCL_LIBRARY} )

    target_compile_options( simple_drift_opencl BEFORE PRIVATE 
       -g  -fno-fast-math -Wall -Werror -pedantic -ansi )
            
    set_target_properties( simple_drift_opencl PROPERTIES LINKER_LANGUAGE C )
    set_target_properties( simple_drift_opencl PROPERTIES C_STANDARD 99 )

endif()



