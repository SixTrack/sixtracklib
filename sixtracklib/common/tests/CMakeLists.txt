# sixtracklib/common/tests/CMakeLists.txt 

if( GTEST_FOUND )

    include_directories( ${SIXTRACKL_GTEST_INCLUDE_DIRS} )

    set( UNIT_TEST_TARGETS ) 
    set( SIXTRACKL_TEST_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES} 
         ${SIXTRACKL_GTEST_LIBRARIES} )
    
    # --------------------------------------------------------------------------
    # test_tools:
    
    add_executable( test_tools test_tools.cpp )        
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_tools )         
    add_test( CommonToolTests test_tools )
    
    # --------------------------------------------------------------------------
    # test_gpu_kernel_tools:
    
    add_executable( test_gpu_kernel_tools test_gpu_kernel_tools.cpp )        
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_gpu_kernel_tools )         
    add_test( CommonToolTests test_gpu_kernel_tools )
    
    # --------------------------------------------------------------------------
    # test_mem_pool:
    
    add_executable( test_mem_pool test_mem_pool.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_mem_pool )
    add_test( CommonMemPoolTests test_mem_pool )
    
    # --------------------------------------------------------------------------
    # test_particles:
    
    add_executable( test_particles test_particles.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_particles )
    add_test( CommonParticlesTests test_particles )    
    
    # --------------------------------------------------------------------------
    # test_particles_sequence:
    
    add_executable( test_particles_sequence test_particles_sequence.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_particles_sequence )
    add_test( CommonParticlesSequenceTests test_particles_sequence )    
    
    # --------------------------------------------------------------------------
    # test_block_drift:
    
    add_executable( test_block_drift test_block_drift.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_block_drift )
    add_test( CommonBlockDriftTests test_block_drift )
    
    # --------------------------------------------------------------------------
    # test_blocks:
    
    add_executable( test_blocks test_blocks.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_blocks )
    add_test( CommonBlocksTests test_blocks )
    
    # *************************************************************************
    # Set all properties: 
    
    set_property( TARGET ${UNIT_TEST_TARGETS} 
                  PROPERTY LINK_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES} )
                  
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD 11 )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD_REQUIRED ON )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY COMPILE_OPTIONS 
                  ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic )
    
endif()
