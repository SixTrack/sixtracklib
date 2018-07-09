if(  NOT SETUP_UNIT_TESTING_FINISHED )
    set( SETUP_UNIT_TESTING_FINISHED 1 )

    message( STATUS "---- Processing sixtracklib/cmake/SetupUnitTesting.cmake" )

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

            find_package( GTest REQUIRED )

        endif()

        if( GTEST_FOUND )

            set( SIXTRACKL_GTEST_INCLUDE_DIRS ${SIXTRACKL_GTEST_INCLUDE_DIRS}
                ${GTEST_INCLUDE_DIRS} )

            set( SIXTRACKL_GTEST_LIBRARIES     ${SIXTRACKL_GTEST_LIBRARIES}
                ${GTEST_BOTH_LIBRARIES} )

        endif() # GTEST_FOUND

    else()

        message( STATUS "---- Disable creation unit-tests using CTest / GTest" )

    endif() # SIXTRACKL_ENABLE_PROGRAMM_TESTS

endif()
