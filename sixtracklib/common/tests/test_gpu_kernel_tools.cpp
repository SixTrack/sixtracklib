#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>

#include <gtest/gtest.h>

#if defined( __unix__ )
    #include <unistd.h>

    #if defined( _POSIX_VERSION )
        #include <sys/stat.h>
    #endif /* defined( _POSIX_VERSION ) */

#endif /* defined( __unix__ ) */

#if defined( __NAMESPACE )
    #define __SAVED_NAMESPACE __NAMESPACE
    #undef  __NAMESPACE     
#endif /* !defiend( __NAMESPACE ) */

#if !defined( __NAMESPACE )
    #define __NAMESPACE st_    
#endif /* !defiend( __NAMESPACE ) */

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/path.h"
#include "sixtracklib/common/details/gpu_kernel_tools.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */


#if defined( _POSIX_VERSION )

TEST( CommonGpuKernelToolsTests, FileExistsAndGetFileSize )
{
    std::string const prefix( st_PATH_TO_BASE_DIR );
    
    std::string const path_to_file1( 
        prefix + "sixtracklib/common/tests/testdata/first_file.txt" );
    
    std::string const path_to_file2(
        prefix + "sixtracklib/common/tests/testdata/second_file.txt" );
    
    std::string const path_to_file3(
        prefix + "sixtracklib/common/tests/testdata/single_really_long_line.txt" );
    
    std::string const path_to_empty_file(
        prefix + "sixtracklib/common/tests/testdata/totally_empty_file.txt" );
    
    std::string const path_to_existing_directory(
        prefix + "sixtracklib/common/tests/testdata" );
    
    std::string const path_to_non_existing_file(
        prefix + "sixtracklib/common/tests/testdata/not_existing_file.txt" );
    
    /* --------------------------------------------------------------------- */
    /* 1.) Check for existance of files: */
    
    struct ::stat stat_buffer;
    
    ASSERT_TRUE( ( 0 == stat( path_to_file1.c_str(), &stat_buffer ) ) ==
                 st_File_exists( path_to_file1.c_str() ) );
    
    ASSERT_TRUE( ( 0 == stat( path_to_file2.c_str(), &stat_buffer ) ) ==
                 st_File_exists( path_to_file2.c_str() ) );
    
    ASSERT_TRUE( ( 0 == stat( path_to_file3.c_str(), &stat_buffer ) ) ==
                 st_File_exists( path_to_file3.c_str() ) );
    
    ASSERT_TRUE( ( 0 == stat( path_to_empty_file.c_str(), &stat_buffer ) ) ==
                 st_File_exists( path_to_empty_file.c_str() ) );
    
    /* directories are a special case -> warn the user about the status of
     * st_File_exists( directory ) */
    
    bool const A = ( 0 == stat( path_to_existing_directory.c_str(), &stat_buffer ) );
    bool const B = ( st_File_exists( path_to_existing_directory.c_str() ) );
    
    if( A )
    {
        if( B )
        {
            std::cerr << "[          ] [Warning]: NS(File_exists) returns true "
                      << "for an existing directory, might not be portable!" 
                      << std::endl;
        }
        else
        {
            std::cerr << "[          ] [Info   ]: NS(File_exists) returns false "
                      << "for an existing directory; this might not be portable!"
                      << std::endl;
        }
    }
    else
    {
        ASSERT_TRUE( !B );
    }
    
    /* Non existing files should work the same in both implementations: */
    
    ASSERT_TRUE( ( 0 != stat( path_to_non_existing_file.c_str(), &stat_buffer ) ) &&
                 ( !st_File_exists( path_to_non_existing_file.c_str() ) ) );
    
    /* --------------------------------------------------------------------- */
    /* 2) Check Filesize: */
    
    /* Important: 
     * Verify that the stat() implementation can handle the file-sizes that
     * are part of the test! the st_File_get_size() call is not designed to 
     * handle files with > 2 GB size */
    
    ASSERT_TRUE( stat( path_to_file1.c_str(), &stat_buffer ) == 0 );
    ASSERT_TRUE( stat_buffer.st_size >= 0 );
    ASSERT_TRUE( static_cast< std::size_t >( stat_buffer.st_size ) == 
                 st_File_get_size( path_to_file1.c_str() ) );
    
    ASSERT_TRUE( stat( path_to_file2.c_str(), &stat_buffer ) == 0 );
    ASSERT_TRUE( stat_buffer.st_size >= 0 );
    ASSERT_TRUE( static_cast< std::size_t >( stat_buffer.st_size ) == 
                 st_File_get_size( path_to_file2.c_str() ) );
    
    ASSERT_TRUE( stat( path_to_file3.c_str(), &stat_buffer ) == 0 );
    ASSERT_TRUE( stat_buffer.st_size >= 0 );
    ASSERT_TRUE( static_cast< std::size_t >( stat_buffer.st_size ) == 
                 st_File_get_size( path_to_file3.c_str() ) );
    
    ASSERT_TRUE( stat( path_to_empty_file.c_str(), &stat_buffer ) == 0 );
    ASSERT_TRUE( stat_buffer.st_size >= 0 );
    ASSERT_TRUE( static_cast< std::size_t >( stat_buffer.st_size ) == 
                 st_File_get_size( path_to_empty_file.c_str() ) );
    
    /* --------------------------------------------------------------------- */    
}

#endif /* defined( _POSIX_VERSION )  */

TEST( CommonGpuKernelToolsTests, BuildSimpleKernelFileList )
{   
    SIXTRL_SIZE_T const CMP_NUM_FILES = ( SIXTRL_SIZE_T )5u;
    SIXTRL_SIZE_T num_of_input_files  = ( SIXTRL_SIZE_T )0u;
    
    char INPUT_FILES_STR[] = 
        "sixtracklib/common/tests/testdata/first_file.txt,"
        "sixtracklib/common/tests/testdata/second_file.txt,"
        "sixtracklib/common/tests/testdata/first_file.txt,"
        "sixtracklib/common/tests/testdata/totally_empty_file.txt,"
        "sixtracklib/common/tests/testdata/single_really_long_line.txt";
    
    char** files = st_GpuKernel_create_file_list(
        INPUT_FILES_STR, &num_of_input_files, st_PATH_TO_BASE_DIR, "," );
    
    
    ASSERT_TRUE( files != nullptr );
    ASSERT_TRUE( CMP_NUM_FILES == num_of_input_files );
    
    for( std::size_t ii = 0 ; ii < CMP_NUM_FILES ; ++ii )
    {
        char const* path_to_ii_file = files[ ii ];
        
        ASSERT_TRUE( std::strlen( path_to_ii_file ) > 0 );
        ASSERT_TRUE( st_File_exists( path_to_ii_file  ) );
        ASSERT_TRUE( st_File_get_size( path_to_ii_file ) >= 0 );
    }
    
    st_GpuKernel_free_file_list( files, num_of_input_files );
    files = nullptr;
}

TEST( CommonGpuKernelToolsTests, BuildSimpleKernelFileListWithWhitespaces )
{   
    SIXTRL_SIZE_T const CMP_NUM_FILES = ( SIXTRL_SIZE_T )5u;
    SIXTRL_SIZE_T num_of_input_files  = ( SIXTRL_SIZE_T )0u;
    
    char INPUT_FILES_STR[] = 
        "sixtracklib/common/tests/testdata/first_file.txt,"
        ","
        "sixtracklib/common/tests/testdata/second_file.txt ,"
        "   sixtracklib/common/tests/testdata/first_file.txt  ,   "
        "sixtracklib/common/tests/testdata/totally_empty_file.txt,"
        "            ,          "
        "sixtracklib/common/tests/testdata/single_really_long_line.txt,   ";
    
    char** files = st_GpuKernel_create_file_list(
        INPUT_FILES_STR, &num_of_input_files, st_PATH_TO_BASE_DIR, "," );
    
    
    ASSERT_TRUE( files != nullptr );
    ASSERT_TRUE( CMP_NUM_FILES == num_of_input_files );
    
    for( std::size_t ii = 0 ; ii < CMP_NUM_FILES ; ++ii )
    {
        char const* path_to_ii_file = files[ ii ];
        ASSERT_TRUE( path_to_ii_file != nullptr );
        
        ASSERT_TRUE( std::strlen( path_to_ii_file ) > 0 );
        ASSERT_TRUE( st_File_exists( path_to_ii_file  ) );
        ASSERT_TRUE( st_File_get_size( path_to_ii_file ) >= 0 );
    }
    
    st_GpuKernel_free_file_list( files, num_of_input_files );
    files = nullptr;
}

TEST( CommonGpuKernelToolsTests, CollectKernelFilesIntoSingleTextFile )
{
    SIXTRL_SIZE_T const CMP_NUM_FILES = ( SIXTRL_SIZE_T )3u;
    SIXTRL_SIZE_T num_of_input_files  = ( SIXTRL_SIZE_T )0u;
    
    char INPUT_FILES_STR[] = 
        "sixtracklib/common/tests/testdata/first_file.txt,"
        "sixtracklib/common/tests/testdata/second_file.txt,"
        "sixtracklib/common/tests/testdata/single_really_long_line.txt";
    
    char** files = st_GpuKernel_create_file_list(
        INPUT_FILES_STR, &num_of_input_files, st_PATH_TO_BASE_DIR, "," );
    
    ASSERT_TRUE( files != nullptr );
    ASSERT_TRUE( num_of_input_files == CMP_NUM_FILES );
    
    std::size_t expected_compiled_size = std::size_t{ 0 };
    
    for( std::size_t ii = 0 ; ii < num_of_input_files ; ++ii )
    {
        expected_compiled_size += st_File_get_size( files[ ii ] );
    }
    
    char* collected_source = st_GpuKernel_collect_source_string(
        files, num_of_input_files, 0, nullptr );
    
    ASSERT_TRUE( collected_source != nullptr );
    ASSERT_TRUE( std::strlen( collected_source ) == expected_compiled_size );
    
    free( collected_source );
    collected_source = nullptr;
    
    st_GpuKernel_free_file_list( files, num_of_input_files );
    files = nullptr;
}

TEST( CommonGpuKernelToolsTests, 
      CollectKernelFilesIntoSingleTextFileWithLinesOffset )
{
    SIXTRL_SIZE_T const CMP_NUM_FILES = ( SIXTRL_SIZE_T )3u;
    SIXTRL_SIZE_T num_of_input_files  = ( SIXTRL_SIZE_T )0u;
    
    char INPUT_FILES_STR[] = 
        "sixtracklib/common/tests/testdata/first_file.txt,"
        "sixtracklib/common/tests/testdata/second_file.txt,"
        "sixtracklib/common/tests/testdata/single_really_long_line.txt";
    
    char** files = st_GpuKernel_create_file_list(
        INPUT_FILES_STR, &num_of_input_files, st_PATH_TO_BASE_DIR, "," );
    
    std::size_t line_offsets[ 3 ] = { 0, 0, 0 };
    
    /* we need a huge line because of the single_really_long_line.txt file */
    std::size_t const MAX_LINE_WIDTH = st_File_get_size( files[ 2 ] ) + 2; 
    
    ASSERT_TRUE( files != nullptr );
    ASSERT_TRUE( num_of_input_files == CMP_NUM_FILES );
    
    std::size_t expected_compiled_size = std::size_t{ 0 };
    
    for( std::size_t ii = 0 ; ii < num_of_input_files ; ++ii )
    {
        expected_compiled_size += st_File_get_size( files[ ii ] );
    }
    
    char* collected_source = st_GpuKernel_collect_source_string(
        files, num_of_input_files, MAX_LINE_WIDTH, &line_offsets[ 0 ] );
    
    ASSERT_TRUE( collected_source != nullptr );
    ASSERT_TRUE( std::strlen( collected_source ) == expected_compiled_size );
    
    free( collected_source );
    collected_source = nullptr;
    
    st_GpuKernel_free_file_list( files, num_of_input_files );
    files = nullptr;
}

TEST( CommonGpuKernelToolsTests, 
      CollectKernelFilesIntoSingleTextFileWithLinesOffsetFailDueToLineLength )
{
    SIXTRL_SIZE_T const CMP_NUM_FILES = ( SIXTRL_SIZE_T )3u;
    SIXTRL_SIZE_T num_of_input_files  = ( SIXTRL_SIZE_T )0u;
    
    char INPUT_FILES_STR[] = 
        "sixtracklib/common/tests/testdata/first_file.txt,"
        "sixtracklib/common/tests/testdata/second_file.txt,"
        "sixtracklib/common/tests/testdata/single_really_long_line.txt";
    
    char** files = st_GpuKernel_create_file_list(
        INPUT_FILES_STR, &num_of_input_files, st_PATH_TO_BASE_DIR, "," );
    
    std::size_t line_offsets[ 3 ] = { 0, 0, 0 };
    std::size_t const MAX_LINE_WIDTH = 100; 
    
    ASSERT_TRUE( files != nullptr );
    ASSERT_TRUE( num_of_input_files == CMP_NUM_FILES );
    
    std::size_t expected_compiled_size = std::size_t{ 0 };
    
    for( std::size_t ii = 0 ; ii < num_of_input_files ; ++ii )
    {
        expected_compiled_size += st_File_get_size( files[ ii ] );
    }
    
    char* collected_source = st_GpuKernel_collect_source_string(
        files, num_of_input_files, MAX_LINE_WIDTH, &line_offsets[ 0 ] );
    
    ASSERT_TRUE( collected_source == nullptr );
        
    st_GpuKernel_free_file_list( files, num_of_input_files );
    files = nullptr;
}

/* end: sixtracklib/common/tests/test_gpu_kernel_tools.cpp */
