    # -----------------------------------------------------------------------------
# simple_drift.c: 

add_executable( simple_drift simple_drift.c )
target_include_directories( simple_drift PRIVATE ${CMAKE_SOURCE_DIR} )
target_link_libraries( simple_drift sixtrack m )

target_compile_options( simple_drift BEFORE PRIVATE 
        -ftree-vectorize -ftree-vectorizer-verbose=6 -fopt-info-loop -fno-fast-math --param vect-max-version-for-alias-checks=50 -msse2 -m64 )
        
set_target_properties( simple_drift PROPERTIES LINKER_LANGUAGE C )
set_target_properties( simple_drift PROPERTIES C_STANDARD 99 )

