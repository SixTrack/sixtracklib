# tests/benchmark/CMakeLists.txt

set( TOMLC99_HEADERS toml.h )
set( TOMLC99_SOURCES toml.c )

add_library( ext_tomlc99 OBJECT ${TOMLC99_HEADERS} ${TOMLC99_SOURCES} )

set( RUN_BENCHMARK_CXX_HEADERS )
set( RUN_BENCHMARK_CXX_SOURCES run_benchmark_cxx.cpp )

add_executable( run_benchmark_cxx
    ${RUN_BENCHMARK_CXX_HEADERS} ${RUN_BENCHMARK_CXX_SOURCES}
    $<TARGET_OBJECTS:ext_tomlc99> )

target_link_libraries( run_benchmark_cxx sixtrack_test sixtrack )

#end: tests/benchmark/CMakeLists.txt
