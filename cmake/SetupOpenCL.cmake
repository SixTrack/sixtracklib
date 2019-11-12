if( NOT  SIXTRACKL_CMAKE_SETUP_OPENCL_FINISHED )
    set( SIXTRACKL_CMAKE_SETUP_OPENCL_FINISHED 1 )

    message(STATUS "---- Processing cmake/SetupOpenCL.cmake")

    # --------------------------------------------------------------------------
    # Add OPENCL to the list of supported modules and track its state:

    list( APPEND SIXTRACKLIB_SUPPORTED_MODULES "OPENCL" )

    if( SIXTRACKL_ENABLE_OPENCL )
        list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES "1" )
    else()
        list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES "0" )
    endif()

    # --------------------------------------------------------------------------
    # Provide include directories and library directories for OpenCL, if enabled

    if( NOT  SIXTRACKL_OPENCL_INCLUDE_DIR )
        set( SIXTRACKL_OPENCL_INCLUDE_DIR   )
    endif()

    if( NOT  SIXTRACKL_OPENCL_LIBRARY )
        set( SIXTRACKL_OPENCL_LIBRARY )
    endif()

    if( NOT  SIXTRACKL_OPENCL_VERSION_STR )
        set( SIXTRACKL_OPENCL_VERSION_STR "" )
    endif()

    set( khr_cxx_ocl_UPDATED 0 )
    set( khr_cxx_ocl_SYNC 0 )
    set( khr_cxx_ocl_EXT_DIR "${CMAKE_SOURCE_DIR}/external/CL" )

    if( SIXTRACKL_ENABLE_OPENCL )
        if( NOT OpenCL_FOUND )
            find_package( OpenCL REQUIRED )
        endif()

        if( OpenCL_FOUND )
            set( SIXTRACKL_OPENCL_LIBRARY
               ${SIXTRACKL_OPENCL_LIBRARY} ${OpenCL_LIBRARY} )

            set( SIXTRACKL_OPENCL_VERSION_STR
               ${SIXTRACKL_OPENCL_VERSION_STR} ${OpenCL_VERSION_STRING} )

            set( SIXTRACKL_OPENCL_INCLUDE_DIR
               ${SIXTRACKL_OPENCL_INCLUDE_DIR} ${OpenCL_INCLUDE_DIR} )

            if( NOT SIXTRACKL_USE_LEGACY_CL_HPP )
                set( CXX_OPENCL_HEADER_NAME "cl2.hpp" )
            else()
                set( CXX_OPENCL_HEADER_NAME "cl.hpp" )
            endif()

            set( CXX_OPENCL_HEADER "${OpenCL_INCLUDE_DIR}/CL/${CXX_OPENCL_HEADER_NAME}" )

            if( NOT EXISTS ${CXX_OPENCL_HEADER} )
                message( STATUS "------ Unable to find OpenCl 1.x C++ header" )

                include( SetupGit )
                include( SetupPython )

                if( Git_FOUND )
                    message( STATUS "------ Attempt to download headers ... " )
                    set( khr_cxx_ocl_GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git )
                    set( khr_cxx_ocl_GIT_BRANCH master )

                    Git_sync_with_repo( TARGET khr_cxx_ocl
                        GIT_REPOSITORY ${khr_cxx_ocl_GIT_REPOSITORY}
                        GIT_BRANCH ${khr_cxx_ocl_GIT_BRANCH} )

                    if( NOT khr_cxx_ocl_DIR )
                        message( FATAL_ERROR "------ Unable to fetch C++ OpenCL headers from git repository" )
                    endif()

                    if( ${khr_cxx_ocl_SYNC} EQUAL 1 )
                        if( ${khr_cxx_ocl_UPDATED} EQUAL 0 )
                            message( STATUS "------ C++ OpenCL headers already sync, no update for embedded library" )
                        elseif( ${khr_cxx_ocl_UPDATED} EQUAL 1 )
                            message( STATUS "------ C++ OpenCL headers successfully cloned/pulled from ${khr_cxx_ocl_GIT_REPOSITORY}/${khr_cxx_ocl_GIT_BRANCH}" )
                            message( STATUS "------ Attempting to update the embedded library at ${khr_cxx_ocl_EXT_DIR} ..." )

                            set( khr_cxx_ocl_TRANSFERRED_HEADER_FILES 0 )
                            if( EXISTS "${khr_cxx_ocl_DIR}/LICENSE.txt" )
                                configure_file( "${khr_cxx_ocl_DIR}/LICENSE.txt"
                                                "${khr_cxx_ocl_EXT_DIR}/LICENSE.txt" COPYONLY )
                            endif()

                            if( EXISTS "${khr_cxx_ocl_DIR}/README.md" )
                                configure_file( "${khr_cxx_ocl_DIR}/README.md"
                                                "${khr_cxx_ocl_EXT_DIR}/OpenCL-CLHPP_README.md" COPYONLY )
                            endif()

                            if( EXISTS "${khr_cxx_ocl_DIR}/CODE_OF_CONDUCT.md" )
                                configure_file( "${khr_cxx_ocl_DIR}/CODE_OF_CONDUCT.md"
                                                "${khr_cxx_ocl_EXT_DIR}/CODE_OF_CONDUCT.md" COPYONLY )
                            endif()

                            if( EXISTS "${khr_cxx_ocl_DIR}/include/CL/cl2.hpp" )
                                configure_file( "${khr_cxx_ocl_DIR}/include/CL/cl2.hpp"
                                                "${khr_cxx_ocl_EXT_DIR}/cl2.hpp" COPYONLY )
                                set( khr_cxx_ocl_TRANSFERRED_HEADER_FILES 1 )
                            else()
                                message( WARNING "------ No CL/cl2.hpp header found inside ${khr_cxx_ocl_DIR}/include -> skipping!" )
                            endif()

                            if( PYTHONINTERP_FOUND
                                AND EXISTS "${khr_cxx_ocl_DIR}/gen_cl_hpp.py"
                                AND EXISTS "${khr_cxx_ocl_DIR}/input_cl.hpp" )

                                execute_process( COMMAND ${PYTHON_EXECUTABLE} gen_cl_hpp.py
                                    WORKING_DIRECTORY ${khr_cxx_ocl_DIR}
                                    OUTPUT_VARIABLE EXE_PROCESS_OUTPUT
                                    RESULT_VARIABLE EXE_PROCESS_RESULT
                                    ERROR_VARIABLE  EXE_PROCESS_ERROR )

                                if( NOT ( ${EXE_PROCESS_RESULT} EQUAL 0 ) AND EXE_PROCESS_OUTPUT )
                                    message( STATUS "------ Error python gen_cl_hpp.py: ${EXE_PROCESS_ERROR}" )
                                endif()

                                if( EXISTS "${khr_cxx_ocl_DIR}/cl.hpp" )
                                    configure_file( "${khr_cxx_ocl_DIR}/cl.hpp"
                                                    "${khr_cxx_ocl_EXT_DIR}/cl.hpp" COPYONLY )
                                    set( khr_cxx_ocl_TRANSFERRED_HEADER_FILES 1 )
                                else()
                                    message( WARNING "------ No cl.hpp file present to add to ${khr_cxx_ocl_EXT_DIR} -> skipping!" )
                                endif()

                            else()
                                message( WARNING "------ Unable to run generator script gen_cl_hpp.py to create cl.hpp -> skipping!" )
                            endif()

                            if( ${khr_cxx_ocl_TRANSFERRED_HEADER_FILES} EQUAL 1 )
                                message( STATUS "------ transfered header files to ${khr_cxx_ocl_EXT_DIR}" )
                            endif()
                        else()
                            message( FATAL_ERROR "------ internal error Git_sync_with_repo" )
                        endif()
                    else()
                        message( WARNING "----- Unable to sync external OpenCL C++ headers -> rely on existing headers instead" )
                    endif()
                endif()

                if( ${khr_cxx_ocl_SYNC} EQUAL 1 AND
                    EXISTS "${khr_cxx_ocl_EXT_DIR}/${CXX_OPENCL_HEADER_NAME}" )
                    set( CXX_OPENCL_HEADER "${khr_cxx_ocl_EXT_DIR}/${CXX_OPENCL_HEADER_NAME}" )
                    set( SIXTRACKL_OPENCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external" )
                endif()
            endif()

            if( EXISTS ${CXX_OPENCL_HEADER} )
                set( SIXTRL_OPENCL_ENABLE_EXCEPTION_STR "" )
                set( SIXTRL_OPENCL_ENABLES_EXCEPTION_FLAG 0 )

                if( SIXTRACKL_USE_LEGACY_CL_HPP )
                    set( SIXTRL_OPENCL_CL_HPP "CL/cl.hpp" )
                    set( SIXTRL_USES_CL2_HPP 0 )
                    set( SIXTRL_USES_CL_HPP  1 )

                    if( SIXTRL_OPENCL_ENABLE_HOST_EXCEPTIONS )
                        set( SIXTRL_OPENCL_ENABLES_EXCEPTION_FLAG 1 )
                        string( APPEND SIXTRL_OPENCL_ENABLE_EXCEPTION_STR
                                "    #if !defined( __CL_ENABLE_EXCEPTIONS )\r\n" )
                        string( APPEND SIXTRL_OPENCL_ENABLE_EXCEPTION_STR
                                "        #define __CL_ENABLE_EXCEPTIONS  \r\n" )
                        string( APPEND SIXTRL_OPENCL_ENABLE_EXCEPTION_STR
                                "    #endif /* !defined( __CL_ENABLE_EXCEPTIONS ) */ \r\n" )
                    endif()
                else()
                    set( SIXTRL_OPENCL_CL_HPP "CL/cl2.hpp" )
                    set( SIXTRL_USES_CL2_HPP 1 )
                    set( SIXTRL_USES_CL_HPP  0 )

                    if( SIXTRACKL_OPENCL_CXX_ENABLE_EXCEPTIONS )
                        set( SIXTRL_OPENCL_ENABLES_EXCEPTION_FLAG 1 )
                        string( APPEND SIXTRL_OPENCL_ENABLE_EXCEPTION_STR
                                "    #if !defined( CL_HPP_ENABLE_EXCEPTIONS )\r\n" )
                        string( APPEND SIXTRL_OPENCL_ENABLE_EXCEPTION_STR
                                "        #define CL_HPP_ENABLE_EXCEPTIONS \r\n" )
                        string( APPEND SIXTRL_OPENCL_ENABLE_EXCEPTION_STR
                                "    #endif /* !defined( CL_HPP_ENABLE_EXCEPTIONS ) */ \r\n" )
                    endif()
                endif()
            endif()
        endif()
    endif()
endif()

#end: cmake/SetupOpenCL.cmake
