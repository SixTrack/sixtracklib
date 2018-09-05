


find_package( OpenCL REQUIRED )

include_directories( ${OpenCL_INCLUDE_DIR} )

# parallel_beam_elements-finalversion
add_executable( parallel_beam_elements-finalversion parallel_beam_elements-finalversion.cpp )
             
target_link_libraries( parallel_beam_elements-finalversion sixtrack m  ${OpenCL_LIBRARY})

set_property( TARGET parallel_beam_elements-finalversion PROPERTY CXX_STANDARD 11 )
set_property( TARGET parallel_beam_elements-finalversion PROPERTY CXX_STANDARD_REQUIRED ON )
set_property( TARGET parallel_beam_elements-finalversion PROPERTY COMPILE_OPTIONS -Wall -Werror -pedantic) 

# parallel_beam_elements_allinsequence
add_executable( parallel_beam_elements_allinsequence parallel_beam_elements_allinsequence.cpp )
             
target_link_libraries( parallel_beam_elements_allinsequence sixtrack m  ${OpenCL_LIBRARY})

set_property( TARGET parallel_beam_elements_allinsequence PROPERTY CXX_STANDARD 11 )
set_property( TARGET parallel_beam_elements_allinsequence PROPERTY CXX_STANDARD_REQUIRED ON )
set_property( TARGET parallel_beam_elements_allinsequence PROPERTY COMPILE_OPTIONS -Wall -Werror -pedantic) 

# parallel_beam_elements_oneatatime
add_executable( parallel_beam_elements_oneatatime parallel_beam_elements_oneatatime.cpp )
             
target_link_libraries( parallel_beam_elements_oneatatime sixtrack m  ${OpenCL_LIBRARY})

set_property( TARGET parallel_beam_elements_oneatatime PROPERTY CXX_STANDARD 11 )
set_property( TARGET parallel_beam_elements_oneatatime PROPERTY CXX_STANDARD_REQUIRED ON )
set_property( TARGET parallel_beam_elements_oneatatime PROPERTY COMPILE_OPTIONS -Wall -Werror -pedantic) 

# parallel_beam_elements_switchcaseoncpu
add_executable( parallel_beam_elements_switchcaseoncpu parallel_beam_elements_switchcaseoncpu.cpp )
             
target_link_libraries( parallel_beam_elements_switchcaseoncpu sixtrack m  ${OpenCL_LIBRARY})

set_property( TARGET parallel_beam_elements_switchcaseoncpu PROPERTY CXX_STANDARD 11 )
set_property( TARGET parallel_beam_elements_switchcaseoncpu PROPERTY CXX_STANDARD_REQUIRED ON )
set_property( TARGET parallel_beam_elements_switchcaseoncpu PROPERTY COMPILE_OPTIONS -Wall -Werror -pedantic) 

# parallel_beam_elements_switchcaseoncpuremoved
add_executable( parallel_beam_elements_switchcaseoncpuremoved parallel_beam_elements_switchcaseoncpuremoved.cpp )
             
target_link_libraries( parallel_beam_elements_switchcaseoncpuremoved sixtrack m  ${OpenCL_LIBRARY})

set_property( TARGET parallel_beam_elements_switchcaseoncpuremoved PROPERTY CXX_STANDARD 11 )
set_property( TARGET parallel_beam_elements_switchcaseoncpuremoved PROPERTY CXX_STANDARD_REQUIRED ON )
set_property( TARGET parallel_beam_elements_switchcaseoncpuremoved PROPERTY COMPILE_OPTIONS -Wall -Werror -pedantic) 
