if(  NOT SETUP_UNIT_TESTING_FINISHED )
    set( SETUP_UNIT_TESTING_FINISHED 1 )

    message( STATUS "---- Processing cmake/SetupUnitTesting.cmake" )

    if( NOT  SIXTRACKL_GTEST_INCLUDE_DIRS )
        set( SIXTRACKL_GTEST_INCLUDE_DIRS )
    endif()

    if( NOT  SIXTRACKL_GTEST_LIBRARIES )
        set( SIXTRACKL_GTEST_LIBRARIES )
    endif()

    if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )

        message( STATUS "---- Enable creation of unit-tests using CTest / GTest" )

        if( SIXTRACKL_ENABLE_PROGRAM_MEMTESTS AND SIXTRACKL_MEMCHECK_COMMAND )
            message( STATUS "---- Enable creation of memory/leack tests using
             ${SIXTRACKL_MEMCHECK_COMMAND}" )

            find_program( MEMORYCHECK_COMMAND ${SIXTRACKL_MEMCHECK_COMMAND} )
            set( MEMORYCHECK_COMMAND_OPTIONS  ${SIXTRACKL_MEMCHECK_COMMAND_OPTIONS} )

            if( SIXTRACKL_MEMCHECK_SUPPRESSIONS_FILE AND
                EXISTS ${SIXTRACKL_MEMCHECK_SUPPRESSIONS_FILE} )

                set( MEMORYCHECK_SUPPRESSIONS_FILE
                     ${SIXTRACKL_MEMORYCHECK_SUPPRESSIONS_FILE} )

            endif()

        endif()

        include( CTest )

        if( NOT GTEST_FOUND )

            set( CMAKE_THREAD_PREFER_PTHREAD ON )
            set( THREADS_PREFER_PTHREAD_FLAG ON )

            find_package( Threads REQUIRED )

            set( SIXTRACKL_GTEST_LIBRARIES ${SIXTRACKL_GTEST_LIBRARIES}
                ${CMAKE_THREAD_LIBS_INIT} )

            if( DEFINED SIXTRACKL_GOOGLETEST_ROOT AND
                ( DEFINED GTEST_ROOT AND
                  NOT ( GTEST_ROOT STREQUAL SIXTRACKL_GOOGLETEST_ROOT ) ) OR
                ( NOT DEFINED GTEST_ROOT ) )
                unset( GTEST_ROOT CACHE )
                set( GTEST_ROOT ${SIXTRACKL_GOOGLETEST_ROOT} )
            endif()

            find_package( GTest )

            if( GTEST_FOUND )

                set( SIXTRACKL_GTEST_INCLUDE_DIRS ${SIXTRACKL_GTEST_INCLUDE_DIRS}
                    ${GTEST_INCLUDE_DIRS} )

                set( SIXTRACKL_GTEST_LIBRARIES     ${SIXTRACKL_GTEST_LIBRARIES}
                    ${GTEST_BOTH_LIBRARIES} )

            elseif( NOT SIXTRACKL_REQUIRE_OFFLINE_BUILD )

                set( EXT_GTEST_IN_DIR "${CMAKE_SOURCE_DIR}/cmake/" )
                set( EXT_GTEST_TMPL "SetupUnitTestingGTestsCMakeLists.txt.in" )

                set( EXT_GTEST_IN_FILE "${EXT_GTEST_IN_DIR}${EXT_GTEST_TMPL}" )
                set( EXT_GTEST_EXT_DIR "${CMAKE_BINARY_DIR}/ext_googletest/download/"  )
                set( EXT_GTEST_OUT     "${EXT_GTEST_EXT_DIR}/CMakeLists.txt"  )

                configure_file( ${EXT_GTEST_IN_FILE} ${EXT_GTEST_OUT} )
                message( STATUS "Attempt downloading and building GTest ... " )

                execute_process(
                    COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${EXT_GTEST_EXT_DIR} )

                if( NOT result )
                    message( STATUS "Successfully run cmake for external GTest" )
                else()
                    message( FATAL_ERROR "Cmake for external GTest failed: ${result}" )
                endif()

                execute_process(
                    COMMAND ${CMAKE_COMMAND} --build .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${EXT_GTEST_EXT_DIR} )

                if( NOT result )
                    message( STATUS "Successfully completed building external GTest" )
                else()
                    message( FATAL_ERROR "Building for external GTest failed: ${result}" )
                endif()

                set( gtest_force_shared_crt ON CACHE BOOL "" FORCE )

                add_subdirectory( ${CMAKE_BINARY_DIR}/ext_googletest/src
                                  ${CMAKE_BINARY_DIR}/ext_googletest/build
                                  EXCLUDE_FROM_ALL )

                set( SIXTRACKL_GTEST_INCLUDE_DIRS ${SIXTRACKL_GTEST_INCLUDE_DIRS}
                    "${gtest_SOURCE_DIR}/include" )

                set( SIXTRACKL_GTEST_LIBRARIES ${SIXTRACKL_GTEST_LIBRARIES}
                     gtest_main )

                set( GTEST_FOUND ON )

            elseif( SIXTRACKL_REQUIRE_OFFLINE_BUILD )
                message( FATAL_ERROR
                         "No system-wide googletest installation "
                         "found and offline installation required\r\n"
                         "set SIXTRACKL_GOOGLETEST_ROOT in Settings.cmake "
                         "to pick up googletest at a specific location" )
            endif()
        endif()

         # GTEST_FOUND

    else()

        message( STATUS "---- Disable creation unit-tests using CTest / GTest" )

    endif() # SIXTRACKL_ENABLE_PROGRAMM_TESTS

endif()
