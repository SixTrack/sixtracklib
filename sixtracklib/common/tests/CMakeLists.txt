# sixtracklib/common/tests/CMakeLists.txt 

if( GTEST_FOUND )

    include_directories( ${SIXTRACKL_GTEST_INCLUDE_DIRS} )
    set( SIXTRACKL_TEST_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES} 
         ${SIXTRACKL_GTEST_LIBRARIES} )

    # --------------------------------------------------------------------------
    # test_mem_pool:
    
    add_executable( test_mem_pool test_mem_pool.cpp )
    target_link_libraries( test_mem_pool ${SIXTRACKL_TEST_LIBRARIES} )
    target_compile_options( test_mem_pool BEFORE PUBLIC -std=c++11 -Wall -Werror -pedantic )
    add_test( MemPoolTests test_mem_pool )
    
    # --------------------------------------------------------------------------
    # test_particles:
    
    add_executable( test_particles test_particles.cpp )
    target_link_libraries( test_particles ${SIXTRACKL_TEST_LIBRARIES} )
    target_compile_options( test_particles BEFORE PUBLIC -std=c++11 -Wall -Werror -pedantic )
    add_test( ParticlesTests test_particles )
    
endif()
