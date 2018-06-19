add_executable( generate_track_test_data_mpfr4 
                generate_track_test_data.c ../details/track.c )

target_include_directories( 
    generate_track_test_data_mpfr4 PRIVATE ${Gmp_INCLUDES} ${MPFR_INCLUDES} )
    
target_compile_definitions( 
    generate_track_test_data_mpfr4 
    PRIVATE -DSIXTRL_REAL_T=st_MpfrWrapper )

set_target_properties( 
    generate_track_test_data_mpfr4 
    PROPERTIES LINKER_LANGUAGE C )
                       
set_target_properties( 
    generate_track_test_data_mpfr4 
    PROPERTIES C_STANDARD 99 )
    
target_link_libraries( 
    generate_track_test_data_mpfr4 ${SIXTRACKL_TEST_LIBRARIES} m
    ${MPFR_LIBRARIES} ${Gmp_LIBRARIES} 
)


