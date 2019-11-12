if(  NOT SETUP_GIT_FINISHED )
    set( SETUP_GIT_FINISHED 1 )

    message( STATUS "---- Processing cmake/SetupGit.cmake" )

    if( NOT Git_FOUND )
        find_package( Git )
    endif()

    function( Git_sync_with_repo )
        set( MULTI_VALUE_ARGS )
        set( OPTIONS )
        set( ONE_VALUE_ARGS TARGET GIT_REPOSITORY GIT_BRANCH GIT_REMOTE
                            OUTPUT_DIRECTORY )
        cmake_parse_arguments( GIT_SYNC_WITH_REPO
            "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN} )

        if( NOT GIT_SYNC_WITH_REPO_TARGET )
            message( FATAL_ERROR "Git_sync_with_repo has to be called with a TARGET name" )
        endif()

        if( NOT GIT_SYNC_WITH_REPO_GIT_REPOSITORY )
            message( FATAL_ERROR "Git_sync_with_repo has to be called with a GIT_REPOSITORY name" )
        endif()

        if( NOT GIT_SYNC_WITH_REPO_GIT_BRANCH )
            message( WARNING "Git_sync_with_repo no GIT_BRANCH provided" )
            message( WARNING "Git_sync_with_repo set GIT_BRANCH to \"master\"" )
            set( GIT_SYNC_WITH_REPO_GIT_BRANCH "master" )
        endif()

        if( NOT GIT_SYNC_WITH_REPO_GIT_REMOTE )
            set( GIT_SYNC_WITH_REPO_GIT_REMOTE "origin" )
        endif()

        string( TOLOWER "${GIT_SYNC_WITH_REPO_TARGET}" LC_TARGET_NAME )

        if( NOT GIT_SYNC_WITH_REPO_OUTPUT_DIRECTORY )
            set( GIT_SYNC_WITH_REPO_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} )
        endif()

        set( TARGET_REPO_NAME "ext_${LC_TARGET_NAME}_git" )
        set( TARGET_DIR "${GIT_SYNC_WITH_REPO_OUTPUT_DIRECTORY}/${TARGET_REPO_NAME}" )


        set( "${LC_TARGET_NAME}_DIR" ${TARGET_DIR} PARENT_SCOPE )
        set( "${LC_TARGET_NAME}_SYNC"    0 PARENT_SCOPE )
        set( "${LC_TARGET_NAME}_UPDATED" 0 PARENT_SCOPE )

        if( Git_FOUND )
            set( REMOTE_REF_HEADS_MASTER )
            set( LOCAL_REF_HEADS_MASTER )
            execute_process(
                COMMAND ${GIT_EXECUTABLE} ls-remote ${GIT_SYNC_WITH_REPO_GIT_REPOSITORY} ${GIT_SYNC_WITH_REPO_GIT_BRANCH}
                WORKING_DIRECTORY ${GIT_SYNC_WITH_REPO_OUTPUT_DIRECTORY}
                OUTPUT_VARIABLE REMOTE_REF_HEADS_MASTER
                RESULT_VARIABLE EXE_PROCESS_RESULT
                ERROR_VARIABLE  EXE_PROCESS_ERROR )

            if( NOT ( ${EXE_PROCESS_RESULT} EQUAL 0 ) AND EXE_PROCESS_ERROR )
                message( STATUS "------ Error git ls-remote: ${EXE_PROCESS_ERROR}" )
            endif()

            string( SUBSTRING "${REMOTE_REF_HEADS_MASTER}" 0 40 REMOTE_REF_HEADS_MASTER )

            if( EXISTS ${TARGET_DIR} )
                execute_process(
                    COMMAND ${GIT_EXECUTABLE} show-ref refs/heads/${GIT_SYNC_WITH_REPO_GIT_BRANCH}
                    WORKING_DIRECTORY ${TARGET_DIR}
                    OUTPUT_VARIABLE LOCAL_REF_HEADS_MASTER
                    RESULT_VARIABLE EXE_PROCESS_RESULT
                    ERROR_VARIABLE  EXE_PROCESS_ERROR )

                if( NOT ( ${EXE_PROCESS_RESULT} EQUAL 0 ) AND EXE_PROCESS_ERROR )
                    message( STATUS "------ Error git show-ref refs/heads/master: ${EXE_PROCESS_ERROR}" )
                endif()

                string( SUBSTRING "${LOCAL_REF_HEADS_MASTER}" 0 40 LOCAL_REF_HEADS_MASTER )
                string( COMPARE NOTEQUAL "${REMOTE_REF_HEADS_MASTER}"
                        "${LOCAL_REF_HEADS_MASTER}" REFS_NOT_EQUAL )

                if( REFS_NOT_EQUAL )
                    # Remote head has changed -> we have to fetch & Merge
                    message( STATUS "------ Remote: ${REMOTE_REF_HEADS_MASTER}" )
                    message( STATUS "------ Local : ${LOCAL_REF_HEADS_MASTER}" )
                    message( STATUS "------ Attempting to fetch & merge from remote ... " )

                    execute_process(
                        COMMAND ${GIT_EXECUTABLE} fetch ${GIT_SYNC_WITH_REPO_GIT_REMOTE}
                        WORKING_DIRECTORY ${TARGET_DIR}
                        OUTPUT_VARIABLE EXE_PROCESS_OUTPUT
                        RESULT_VARIABLE EXE_PROCESS_RESULT
                        ERROR_VARIABLE  EXE_PROCESS_ERROR )

                    if( NOT ( ${EXE_PROCESS_RESULT} EQUAL 0 ) AND EXE_PROCESS_OUTPUT )
                        message( STATUS "------ Output git fetch origin: ${EXE_PROCESS_OUTPUT}" )
                    endif()

                    execute_process(
                        COMMAND ${GIT_EXECUTABLE} checkout ${GIT_SYNC_WITH_REPO_GIT_BRANCH}
                        WORKING_DIRECTORY ${TARGET_DIR}
                        OUTPUT_VARIABLE EXE_PROCESS_OUTPUT
                        RESULT_VARIABLE EXE_PROCESS_RESULT
                        ERROR_VARIABLE  EXE_PROCESS_ERROR )

                    if( NOT ( ${EXE_PROCESS_RESULT} EQUAL 0 ) AND EXE_PROCESS_OUTPUT )
                        message( STATUS "------ Output git checkout ${GIT_SYNC_WITH_REPO_GIT_BRANCH}: ${EXE_PROCESS_OUTPUT}" )
                    endif()

                    execute_process(
                        COMMAND ${GIT_EXECUTABLE} merge --ff-only ${GIT_SYNC_WITH_REPO_GIT_REMOTE}/${GIT_SYNC_WITH_REPO_GIT_BRANCH}
                        WORKING_DIRECTORY ${TARGET_DIR}
                        OUTPUT_VARIABLE EXE_PROCESS_OUTPUT
                        RESULT_VARIABLE EXE_PROCESS_RESULT
                        ERROR_VARIABLE  EXE_PROCESS_ERROR )

                    if( NOT ( ${EXE_PROCESS_RESULT} EQUAL 0 ) AND EXE_PROCESS_OUTPUT )
                        message( STATUS "------ Output git merge --ff-only ${GIT_SYNC_WITH_REPO_GIT_REMOTE}/${GIT_SYNC_WITH_REPO_GIT_BRANCH}: ${EXE_PROCESS_OUTPUT}" )
                    endif()

                    set( "${LC_TARGET_NAME}_UPDATED" 1 PARENT_SCOPE )
                endif()
            else()
                message( STATUS "------ Attempting to clone remote git repository ... " )

                execute_process(
                    COMMAND ${GIT_EXECUTABLE} clone ${GIT_SYNC_WITH_REPO_GIT_REPOSITORY} ${TARGET_REPO_NAME}
                    WORKING_DIRECTORY ${GIT_SYNC_WITH_REPO_OUTPUT_DIRECTORY}
                    OUTPUT_VARIABLE EXE_PROCESS_OUTPUT
                    RESULT_VARIABLE EXE_PROCESS_RESULT
                    ERROR_VARIABLE  EXE_PROCESS_ERROR )

                if( NOT ( ${EXE_PROCESS_RESULT} EQUAL 0 ) AND EXE_PROCESS_OUTPUT )
                    message( STATUS "------ Output git clone: ${EXE_PROCESS_OUTPUT}" )
                endif()

                execute_process(
                    COMMAND ${GIT_EXECUTABLE} checkout ${GIT_SYNC_WITH_REPO_GIT_BRANCH}
                    WORKING_DIRECTORY ${TARGET_DIR}
                    OUTPUT_VARIABLE EXE_PROCESS_OUTPUT
                    RESULT_VARIABLE EXE_PROCESS_RESULT
                    ERROR_VARIABLE  EXE_PROCESS_ERROR )

                if( NOT ( ${EXE_PROCESS_RESULT} EQUAL 0 ) AND EXE_PROCESS_OUTPUT )
                    message( STATUS "------ Output git checkout ${GIT_SYNC_WITH_REPO_GIT_BRANCH}: ${EXE_PROCESS_OUTPUT}" )
                endif()

                execute_process(
                    COMMAND ${GIT_EXECUTABLE} show-ref refs/heads/${GIT_SYNC_WITH_REPO_GIT_BRANCH}
                    WORKING_DIRECTORY ${TARGET_DIR}
                    OUTPUT_VARIABLE LOCAL_REF_HEADS_MASTER
                    RESULT_VARIABLE EXE_PROCESS_RESULT
                    ERROR_VARIABLE  EXE_PROCESS_ERROR )

                if( NOT ( ${EXE_PROCESS_RESULT} EQUAL 0 ) AND LOCAL_REF_HEADS_MASTER )
                    message( STATUS "------ Error git show-ref refs/heads/${GIT_SYNC_WITH_REPO_GIT_BRANCH}: ${EXE_PROCESS_ERROR}" )
                endif()

                string( SUBSTRING "${LOCAL_REF_HEADS_MASTER}" 0 40 LOCAL_REF_HEADS_MASTER )
                set( "${LC_TARGET_NAME}_UPDATED" 1 PARENT_SCOPE )
            endif()

            string( COMPARE EQUAL "${REMOTE_REF_HEADS_MASTER}"
                                  "${LOCAL_REF_HEADS_MASTER}" REFS_EQUAL )

            if( REFS_EQUAL )
                set( "${LC_TARGET_NAME}_SYNC" 1 PARENT_SCOPE )
            endif()

        endif()
    endfunction()
endif()
