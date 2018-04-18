# sixtracklib/common/tests/CMakeLists.txt 

if( GTEST_FOUND )

    include_directories( ${SIXTRACKL_GTEST_INCLUDE_DIRS} )
    set( SIXTRACKL_TEST_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES} 
         ${SIXTRACKL_GTEST_LIBRARIES} )

    # --------------------------------------------------------------------------
    # test_tools:
    
    add_executable( test_tools test_tools.cpp )
    target_link_libraries( test_tools ${SIXTRACKL_TEST_LIBRARIES} )
    target_compile_options( test_tools BEFORE PUBLIC -std=c++11 -Wall -Werror -pedantic )
    add_test( CommonToolTests test_tools )
         
    # --------------------------------------------------------------------------
    # test_mem_pool:
    
    add_executable( test_mem_pool test_mem_pool.cpp )
    target_link_libraries( test_mem_pool ${SIXTRACKL_TEST_LIBRARIES} )
    target_compile_options( test_mem_pool BEFORE PUBLIC -std=c++11 -Wall -Werror -pedantic )
    add_test( CommonMemPoolTests test_mem_pool )
    
    # --------------------------------------------------------------------------
    # test_particles:
    
    add_executable( test_particles test_particles.cpp )
    target_link_libraries( test_particles ${SIXTRACKL_TEST_LIBRARIES} )
    target_compile_options( test_particles BEFORE PUBLIC -std=c++11 -Wall -Werror -pedantic )
    add_test( CommonParticlesTests test_particles )    
    
    # --------------------------------------------------------------------------
    # test_particles_sequence:
    
    add_executable( test_particles_sequence test_particles_sequence.cpp )
    target_link_libraries( test_particles_sequence ${SIXTRACKL_TEST_LIBRARIES} )
    target_compile_options( test_particles_sequence BEFORE PUBLIC -std=c++11 -Wall -Werror -pedantic )
    add_test( CommonParticlesSequenceTests test_particles_sequence )    
    
    # --------------------------------------------------------------------------
    # test_block_drift:
    
    add_executable( test_block_drift test_block_drift.cpp )
    target_link_libraries( test_block_drift ${SIXTRACKL_TEST_LIBRARIES} )
    target_compile_options( test_block_drift BEFORE PUBLIC -std=c++11 -Wall -Werror -pedantic )
    add_test( CommonBlockDriftTests test_block_drift )
    
    # --------------------------------------------------------------------------
    # test_blocks:
    
    add_executable( test_blocks test_blocks.cpp )
    target_link_libraries( test_blocks ${SIXTRACKL_TEST_LIBRARIES} )
    target_compile_options( test_blocks BEFORE PUBLIC -std=c++11 -Wall -Werror -pedantic )
    add_test( CommonBlocksTests test_blocks )
    
endif()
