if( NOT  SIXTRACKL_CMAKE_SETUP_OPENCL_FINISHED )
    set( SIXTRACKL_CMAKE_SETUP_OPENCL_FINISHED 1 )

    message(STATUS "---- Processing cmake/SetupOpenCL.cmake")

    # --------------------------------------------------------------------------
    # Add OPENCL to the list of supported modules and track its state:
    set( SIXTRACKLIB_MODULE_VALUE_OPENCL 0 )

    # --------------------------------------------------------------------------
    # Provide include directories and library directories for OpenCL, if enabled

    if( SIXTRACKL_ENABLE_OPENCL )
        if( NOT DEFINED SIXTRACKL_OPENCL_ALWAYS_DOWNLOAD_HEADERS )
            set( SIXTRACKL_OPENCL_ALWAYS_DOWNLOAD_HEADERS OFF )
        endif()

        set( SIXTRL_OPENCL_INCLUDE_DIRS )
        set( SIXTRL_OPENCL_LIBRARIES )
        set( SIXTRL_OPENCL_C99_HEADERS_DOWNLOADED 0 )
        set( SIXTRL_OPENCL_CXX_HEADERS_DOWNLOADED 0 )

        if( NOT OpenCL_FOUND )
            find_package( OpenCL QUIET )
        endif()

        if( OpenCL_FOUND )
            set( SIXTRL_TEMP_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS} )
            set( SIXTRL_OPENCL_LIBRARIES ${OpenCL_LIBRARIES} )
            message( STATUS "---- OpenCL environment found (ver. ${OpenCL_VERSION_STRING})" )
        elseif( SIXTRACKL_REQUIRE_OFFLINE_BUILD )
            message( FATAL_ERROR
                "---- Unable to find OpenCL setup, unable to download since offline build required" )
        else()
            message( STATUS "---- OpenCL not found, use fallback headers due to offline build" )
            set( SIXTRL_TEMP_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/external" )
        endif()

        foreach( dir ${SIXTRL_TEMP_INCLUDE_DIRS} )
            if( NOT DEFINED SIXTRL_OPENCL_C99_HEADER_FILE )
                if( EXISTS "${dir}/CL/opencl.h" )
                    set( SIXTRL_OPENCL_C99_HEADER_FILE "CL/opencl.h" )
                    set( SIXTRL_OPENCL_C99_INCLUDE_DIR ${dir} )
                    set( SIXTRL_OPENCL_C99_HEADER_FILE_VERSION 3 )
                elseif( EXISTS "${dir}/CL/cl.h" )
                    set( SIXTRL_OPENCL_C99_HEADER_FILE "CL/cl.h" )
                    set( SIXTRL_OPENCL_C99_INCLUDE_DIR ${dir} )
                    set( SIXTRL_OPENCL_C99_HEADER_FILE_VERSION 1 )
                endif()
            endif()

            if( NOT DEFINED SIXTRL_OPENCL_CXX_HEADER_FILE )
                if( EXISTS "${dir}/CL/opencl.hpp" )
                    set( SIXTRL_OPENCL_CXX_HEADER_FILE "CL/opencl.hpp" )
                    set( SIXTRL_OPENCL_CXX_INCLUDE_DIR ${dir} )
                    set( SIXTRL_OPENCL_CXX_HEADER_FILE_VERSION 3 )
                elseif( EXISTS "${dir}/CL/cl2.hpp" )
                    set( SIXTRL_OPENCL_CXX_HEADER_FILE "CL/cl2.hpp" )
                    set( SIXTRL_OPENCL_CXX_INCLUDE_DIR ${dir} )
                    set( SIXTRL_OPENCL_CXX_HEADER_FILE_VERSION 2 )
                elseif( EXISTS "${dir}/CL/cl.hpp" )
                    set( SIXTRL_OPENCL_CXX_HEADER_FILE "CL/cl.hpp" )
                    set( SIXTRL_OPENCL_CXX_INCLUDE_DIR ${dir} )
                    set( SIXTRL_OPENCL_CXX_HEADER_FILE_VERSION 1 )
                endif()
            endif()
        endforeach()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if( NOT OpenCL_FOUND OR
            SIXTRACKL_OPENCL_ALWAYS_DOWNLOAD_HEADERS OR
            NOT DEFINED SIXTRL_OPENCL_C99_HEADER_FILE OR
            NOT DEFINED SIXTRL_OPENCL_CXX_HEADER_FILE )

            set( SIXTRL_OPENCL_EXT_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include" )
            include( FetchContent )
        endif()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if( ( NOT SIXTRACKL_REQUIRE_OFFLINE_BUILD ) AND
            ( SIXTRACKL_OPENCL_ALWAYS_DOWNLOAD_HEADERS OR
              NOT DEFINED SIXTRL_OPENCL_C99_HEADER_FILE OR
              NOT DEFINED SIXTRL_OPENCL_C99_HEADER_FILE_VERSION ) )

            if( NOT EXISTS "${SIXTRL_OPENCL_EXT_INCLUDE_DIR}/CL" )
                file( MAKE_DIRECTORY "${SIXTRL_OPENCL_EXT_INCLUDE_DIR}/CL" )
            endif()

            FetchContent_Declare( opencl_c99_headers
                GIT_REPOSITORY "https://github.com/KhronosGroup/OpenCL-Headers.git"
                GIT_TAG "v2020.06.16"
                GIT_SHALLOW 1
                CONFIGURE_COMMAND ${CMAKE_COMMAND} -E echo "Configure: no operation"
                BUILD_COMMAND ${CMAKE_COMMAND} -E echo "Build: no operation"
                INSTALL_COMMAND ${CMAKE_COMMAND} -E echo "Install: handled outside of this step" )

            FetchContent_GetProperties( opencl_c99_headers )
            message( STATUS "------ Using external OpenCL C99 headers" )
            if( NOT opencl_c99_headers_POPULATED )
                message( STATUS "------ Downloading external OpenCL C99 headers ..." )
                FetchContent_Populate( opencl_c99_headers )
                message( STATUS "------ Downloading external OpenCL C99 headers [DONE]" )
            endif()

            if( opencl_c99_headers_POPULATED )
                file( COPY "${opencl_c99_headers_SOURCE_DIR}/CL/"
                      DESTINATION "${SIXTRL_OPENCL_EXT_INCLUDE_DIR}/CL"
                      PATTERN "*.h" )

                if( EXISTS "${SIXTRL_OPENCL_EXT_INCLUDE_DIR}/CL/opencl.h" )
                    set( SIXTRL_OPENCL_C99_HEADER_FILE "CL/opencl.h" )
                    set( SIXTRL_OPENCL_C99_INCLUDE_DIR ${SIXTRL_OPENCL_EXT_INCLUDE_DIR} )
                    set( SIXTRL_OPENCL_C99_HEADER_FILE_VERSION 3 )
                    set( SIXTRL_OPENCL_C99_HEADERS_DOWNLOADED 1 )
                elseif( EXISTS "${SIXTRL_OPENCL_EXT_INCLUDE_DIR}/CL/cl.h" )
                    set( SIXTRL_OPENCL_C99_HEADER_FILE "CL/cl.h" )
                    set( SIXTRL_OPENCL_C99_INCLUDE_DIR ${SIXTRL_OPENCL_EXT_INCLUDE_DIR} )
                    set( SIXTRL_OPENCL_C99_HEADER_FILE_VERSION 1 )
                    set( SIXTRL_OPENCL_C99_HEADERS_DOWNLOADED 1 )
                endif()
            endif()
        endif()

        if( ( NOT SIXTRACKL_REQUIRE_OFFLINE_BUILD ) AND
            ( SIXTRACKL_OPENCL_ALWAYS_DOWNLOAD_HEADERS OR
              NOT DEFINED SIXTRL_OPENCL_CXX_HEADER_FILE OR
              NOT DEFINED SIXTRL_OPENCL_CXX_HEADER_FILE_VERSION ) )
            if( NOT EXISTS ${SIXTRL_OPENCL_EXT_INCLUDE_DIR} )
                file( MAKE_DIRECTORY ${SIXTRL_OPENCL_EXT_INCLUDE_DIR} )
            endif()

            FetchContent_Declare( opencl_cxx_headers
                GIT_REPOSITORY "https://github.com/KhronosGroup/OpenCL-CLHPP.git"
                GIT_TAG "master"
                GIT_SHALLOW 1
                CONFIGURE_COMMAND ${CMAKE_COMMAND} -E echo "Configure: no operation"
                BUILD_COMMAND ${CMAKE_COMMAND} -E echo "Build: no operation"
                INSTALL_COMMAND ${CMAKE_COMMAND} -E echo "Install: handled outside of this step" )

            FetchContent_GetProperties( opencl_cxx_headers )
            message( STATUS "------ Using external OpenCL C++ headers" )
            if( NOT opencl_cxx_headers_POPULATED )
                message( STATUS "------ Downloading external OpenCL C++ headers ..." )
                FetchContent_Populate( opencl_cxx_headers )
                message( STATUS "------ Downloading external OpenCL C++ headers [DONE]" )
            endif()

            if( opencl_cxx_headers_POPULATED )
                file( COPY "${opencl_cxx_headers_SOURCE_DIR}/include/CL/"
                      DESTINATION "${SIXTRL_OPENCL_EXT_INCLUDE_DIR}/CL"
                      PATTERN "*.hpp" )

                if( EXISTS "${SIXTRL_OPENCL_EXT_INCLUDE_DIR}/CL/opencl.hpp" )
                    set( SIXTRL_OPENCL_CXX_HEADER_FILE "CL/opencl.hpp" )
                    set( SIXTRL_OPENCL_CXX_INCLUDE_DIR ${SIXTRL_OPENCL_EXT_INCLUDE_DIR} )
                    set( SIXTRL_OPENCL_CXX_HEADER_FILE_VERSION 3 )
                    set( SIXTRL_OPENCL_CXX_HEADERS_DOWNLOADED 1 )
                elseif( EXISTS "${SIXTRL_OPENCL_EXT_INCLUDE_DIR}/CL/cl2.hpp" )
                    set( SIXTRL_OPENCL_CXX_HEADER_FILE "CL/cl2.hpp" )
                    set( SIXTRL_OPENCL_CXX_INCLUDE_DIR ${SIXTRL_OPENCL_EXT_INCLUDE_DIR} )
                    set( SIXTRL_OPENCL_CXX_HEADER_FILE_VERSION 2 )
                    set( SIXTRL_OPENCL_CXX_HEADERS_DOWNLOADED 1 )
                elseif( EXISTS "${SIXTRL_OPENCL_EXT_INCLUDE_DIR}/CL/cl.hpp" )
                    set( SIXTRL_OPENCL_CXX_HEADER_FILE "CL/cl.hpp" )
                    set( SIXTRL_OPENCL_CXX_INCLUDE_DIR ${SIXTRL_OPENCL_EXT_INCLUDE_DIR} )
                    set( SIXTRL_OPENCL_CXX_HEADER_FILE_VERSION 1 )
                    set( SIXTRL_OPENCL_CXX_HEADERS_DOWNLOADED 1 )
                endif()
            endif()
        endif()

        if( NOT DEFINED SIXTRL_OPENCL_CXX_HEADER_FILE OR
            NOT DEFINED SIXTRL_OPENCL_CXX_INCLUDE_DIR )
            message( FATAL_ERROR "---- No C++ OpenCL headers available" )
        endif()

        if( NOT DEFINED SIXTRL_OPENCL_C99_HEADER_FILE OR
            NOT DEFINED SIXTRL_OPENCL_C99_INCLUDE_DIR )
            message( FATAL_ERROR "---- No C OpenCL headers available" )
        endif()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if( OpenCL_FOUND )
            set( SIXTRACKLIB_MODULE_VALUE_OPENCL 1 )
        elseif( NOT OpenCL_FOUND AND NOT SIXTRACKL_REQUIRE_OFFLINE_BUILD )
            FetchContent_Declare( opencl_icd_loader
                GIT_REPOSITORY "https://github.com/KhronosGroup/OpenCL-ICD-Loader.git"
                GIT_TAG "v2020.06.16"
                GIT_SHALLOW 1 )

            if( NOT opencl_icd_loader_POPULATED )
                message( STATUS "------ Downloading external OpenCL ICD Loader ..." )
                FetchContent_Populate( opencl_icd_loader )
                message( STATUS "------ Downloading external OpenCL ICD Loader [DONE]" )
            endif()

            get_filename_component( SIXTRL_TEMP_INCLUDE_DIRS
                "${SIXTRL_OPENCL_C99_INCLUDE_DIR}/${SIXTRL_OPENCL_C99_HEADER_FILE}"
                DIRECTORY )

            file( COPY "${SIXTRL_TEMP_INCLUDE_DIRS}/"
                  DESTINATION "${opencl_icd_loader_SOURCE_DIR}/inc"
                  PATTERN "*.h" )

            FetchContent_MakeAvailable( opencl_icd_loader )
            set( SIXTRL_OPENCL_LIBRARIES ${SIXTRL_OPENCL_LIBRARIES} OpenCL )
            set( SIXTRACKLIB_MODULE_VALUE_OPENCL 1 )

        elseif( NOT OpenCL_FOUND )
            message( FATAL_ERROR
                "---- Unable to download OpenCL icd loader due to offline build requirement" )
        endif()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        set( SIXTRL_OPENCL_INCLUDE_DIRS ${SIXTRL_OPENCL_INCLUDE_DIRS}
             ${SIXTRL_OPENCL_C99_INCLUDE_DIR} )

        if( NOT ( "${SIXTRL_OPENCL_CXX_INCLUDE_DIR}" STREQUAL
                  "${SIXTRL_OPENCL_C99_INCLUDE_DIR}" ) )
            set(   SIXTRL_OPENCL_INCLUDE_DIRS ${SIXTRL_OPENCL_INCLUDE_DIRS}
                 ${SIXTRL_OPENCL_CXX_INCLUDE_DIR} )
        endif()
    endif()

    # ---------------------------------------------------------------------------

    set( SIXTRL_OPENCL_ENABLE_EXCEPTION_STR "" )
    set( SIXTRL_OPENCL_ENABLES_EXCEPTION_FLAG 0 )

    if( ${SIXTRACKLIB_MODULE_VALUE_OPENCL} EQUAL 1 )
        if( SIXTRACKL_OPENCL_DEFAULT_COMPILER_FLAGS )
            set( SIXTRL_DEFAULT_OPENCL_COMPILER_FLAGS
                ${SIXTRACKL_OPENCL_DEFAULT_COMPILER_FLAGS} )
        endif()

        if( SIXTRACKL_OPENCL_CXX_ENABLE_EXCEPTIONS )
            if( SIXTRL_OPENCL_CXX_HEADER_FILE_VERSION EQUAL 2 OR
                SIXTRL_OPENCL_CXX_HEADER_FILE_VERSION EQUAL 3 )
                set( SIXTRL_OPENCL_ENABLE_EXCEPTION_STR_MACRO
                    "CL_HPP_ENABLE_EXCEPTIONS" )
            else()
                set( SIXTRL_OPENCL_ENABLE_EXCEPTION_STR_MACRO
                    "__CL_ENABLE_EXCEPTIONS" )
            endif()

            set( SIXTRL_OPENCL_ENABLES_EXCEPTION_FLAG 1 )
            set( SIXTRL_OPENCL_ENABLE_EXCEPTION_STR
                "#if !defined( ${SIXTRL_OPENCL_ENABLE_EXCEPTION_STR_MACRO} )
                    #define ${SIXTRL_OPENCL_ENABLE_EXCEPTION_STR_MACRO}
                 #endif /* !defined( ${SIXTRL_OPENCL_ENABLE_EXCEPTION_STR_MACRO} ) */" )
        endif()
    endif()

    # ---------------------------------------------------------------------------

    list( APPEND SIXTRACKLIB_SUPPORTED_MODULES "OPENCL" )
    list( APPEND SIXTRACKLIB_SUPPORTED_MODULES_VALUES
            "${SIXTRACKLIB_MODULE_VALUE_OPENCL}" )
endif()
