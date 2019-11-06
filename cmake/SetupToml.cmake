if(  NOT SETUP_TOML_FINISHED )
    set( SETUP_TOML_FINISHED 1 )

    message( STATUS "---- Processing cmake/SetupToml.cmake" )

    if( NOT  SIXTRACKL_TOML_INCLUDE_DIRS )
        set( SIXTRACKL_TOML_INCLUDE_DIRS )
    endif()

    include( SetupGit )

    set( toml11_GIT_REPOSITORY https://github.com/ToruNiina/toml11.git )
    set( toml11_GIT_BRANCH master )
    set( toml11_EXT_DIR "${CMAKE_SOURCE_DIR}/external/toml11" )

    Git_sync_with_repo( TARGET toml11
        GIT_REPOSITORY ${toml11_GIT_REPOSITORY}
        GIT_BRANCH ${toml11_GIT_BRANCH} )

    if( NOT toml11_DIR  )
        message( FATAL_ERROR "------ unable to fetch toml11 from git repository" )
    endif()

    if( ${toml11_SYNC} EQUAL 1 )
        if( ${toml11_UPDATED} EQUAL 0 )
            message( STATUS "------ toml11 already sync, no update for embedded library" )
        elseif( ${toml11_UPDATED} EQUAL 1 )
            message( STATUS "------ toml11 successfully cloned/pulled from ${toml11_GIT_REPOSITORY}/${toml11_GIT_BRANCH}" )
            message( STATUS "------ attempting to update the embedded library at ${toml11_EXT_DIR} ..." )

            set( toml11_TRANSFERED_ANY_FILES 0 )
            if( EXISTS "${toml11_DIR}/LICENSE" )
                configure_file( "${toml11_DIR}/LICENSE" "${toml11_EXT_DIR}/LICENSE" COPYONLY )
                set( toml11_TRANSFERED_ANY_FILES 1 )
            endif()

            if( EXISTS "${toml11_DIR}/README.md" )
                configure_file( "${toml11_DIR}/README.md"
                                "${toml11_EXT_DIR}/toml11_README.md" COPYONLY )
                set( toml11_TRANSFERED_ANY_FILES 1 )
            endif()

            if( EXISTS "${toml11_DIR}/toml.hpp" )
                configure_file( "${toml11_DIR}/toml.hpp" "${toml11_EXT_DIR}/toml.hpp" COPYONLY )
                set( toml11_TRANSFERED_ANY_FILES 1 )
            endif()

            file( GLOB toml11_IMPL_FILES "${toml11_DIR}/toml/*" )

            if( toml11_IMPL_FILES )
                foreach( PATH_TO_FILE IN LISTS toml11_IMPL_FILES )
                    get_filename_component( FILE_NAME ${PATH_TO_FILE} NAME )
                    configure_file( ${PATH_TO_FILE}
                                    "${toml11_EXT_DIR}/toml/${FILE_NAME}" COPYONLY )
                    set( toml11_TRANSFERED_ANY_FILES 1 )
                endforeach()
            endif()

            if( ${toml11_TRANSFERED_ANY_FILES} EQUAL 1 )
                message( STATUS "------ transfered updated files to ${toml11_EXT_DIR}" )
            endif()
        else()
            message( FATAL_ERROR "------ internal error Git_sync_with_repo" )
        endif()

        set(  SIXTRACKL_TOML_INCLUDE_DIRS
            ${SIXTRACKL_TOML_INCLUDE_DIRS} ${toml11_DIR} )
    else()
        message( WARNING "------ unable to sync toml11 with external repository ${toml11_GIT_REPOSITORY}/${toml11_GIT_BRANCH}" )
        message( WARNING "------ rely on the contents of ${toml11_EXT_DIR}" )
    endif()
endif()
