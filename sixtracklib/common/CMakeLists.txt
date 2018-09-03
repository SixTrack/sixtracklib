# sixtracklib/common/CMakeLists.txt

set( SIXTRACKLIB_COMMON_IMPL_HEADERS
     impl/alignment_impl.h
     impl/be_drift.h
     impl/be_drift.hpp
     impl/be_multipole.h
     impl/be_multipole.hpp
     impl/be_cavity.h
     impl/be_cavity.hpp
     impl/be_xyshift.h
     impl/be_xyshift.hpp
     impl/be_srotation.h
     impl/be_srotation.hpp
#    impl/be_beam_beam.h
#    impl/be_beam_beam.hpp
     impl/buffer_type.h
     impl/buffer_object.h
     impl/buffer_generic.h
     impl/managed_buffer_minimal.h
     impl/managed_buffer_remap.h
     impl/managed_buffer.h
#    impl/faddeeva.h
#    impl/faddeeva_mit.h
)

set( SIXTRACKLIB_COMMON_DETAILS_HEADERS
     details/tools.h
)

set( SIXTRACKLIB_COMMON_HEADERS
     alignment.h
     beam_elements.h
     beam_elements.hpp
     buffer.h
     buffer.hpp
     compute_arch.h
     mem_pool.h
     particles.h
     particles.hpp
     track.h
)

set( SIXTRACKLIB_COMMON_SOURCES
     details/alignment.c
     details/buffer.c
     details/compute_arch.c
#    details/faddeeva_mit.c
     details/mem_pool.c
)

add_library( sixtrack_common OBJECT
             ${SIXTRACKLIB_COMMON_IMPL_HEADERS}
             ${SIXTRACKLIB_COMMON_DETAILS_HEADERS}
             ${SIXTRACKLIB_COMMON_HEADERS}
             ${SIXTRACKLIB_COMMON_SOURCES} )

target_include_directories( sixtrack_common
                            PUBLIC $<INSTALL_INTERFACE:include>
                                   $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> )

set_target_properties(  sixtrack_common PROPERTIES LINKER_LANGUAGE C )
set_target_properties(  sixtrack_common PROPERTIES POSITION_INDEPENDENT_CODE True )
set_target_properties(  sixtrack_common PROPERTIES C_STANDARD 99 )

target_compile_definitions( sixtrack_common PUBLIC
                            ${SIXTRACKLIB_NAMESPACE_FLAGS} )

target_compile_options( sixtrack_common BEFORE PUBLIC
                        ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic -ansi )

# ------------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_COMMON_INSTALL_PATH include/sixtracklib/common )

install( FILES ${SIXTRACKLIB_COMMON_HEADERS}
         DESTINATION ${SIXTRACKLIB_COMMON_INSTALL_PATH} )

install( FILES ${SIXTRACKLIB_COMMON_IMPL_HEADERS}
         DESTINATION "${SIXTRACKLIB_COMMON_INSTALL_PATH}/impl" )

#end sixtracklib/common/CMakeLists.txt
