if( SIXTRACKL_ENABLE_MPFR4 )

    if( NOT ( Gmp_FOUND AND MPFR_FOUND ) )
        message( FATAL_ERROR "GMP and MPFR4 not found!" )
    endif()

    set( SIXTRACKL_MPFR4_HEADERS
         impl/particles_impl.h
         beam_elements.h
         track.h
    )

    set( SIXTRACKL_MPFR4_SOURCES
         details/track.c
    )

#     add_library( sixtrack_mpfr4 OBJECT
#         ${SIXTRACKL_MPFR4_HEADERS}
#         ${SIXTRACKL_MPFR4_SOURCES}
#     )

    if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )
        add_subdirectory( tests )
    endif()

endif()

# end:  sixtracklib/mpfr4/CMakeLists.txt
