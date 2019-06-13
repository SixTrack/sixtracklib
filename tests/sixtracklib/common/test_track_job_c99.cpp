#include "sixtracklib/common/track_job.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/buffer.h"

#include "sixtracklib/testlib/testdata/testdata_files.h"

TEST( C99_TrackJobTests, DeviceIdAndConfigStrTests )
{
    using buf_size_t = ::NS(buffer_size_t);
    std::string conf_str( "" );

    ::NS(buffer_size_t) const max_out_str_len = 32u;
    char device_id_str[ 32 ];
    std::memset( &device_id_str[ 0 ], ( int )'\0', max_out_str_len );

    int ret = ::NS(TrackJob_extract_device_id_str)(
        conf_str.c_str(), &device_id_str[ 0 ], max_out_str_len );

    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( std::strlen( &device_id_str[ 0 ] ) == buf_size_t{ 0 } );

    conf_str = "0.0";
    std::memset( &device_id_str[ 0 ], ( int )'\0', max_out_str_len );

    ret = ::NS(TrackJob_extract_device_id_str)(
        conf_str.c_str(), &device_id_str[ 0 ], max_out_str_len );

    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( std::strcmp( &device_id_str[ 0 ], "0.0" ) == 0 );

    conf_str = "  0.0  ";
    std::memset( &device_id_str[ 0 ], ( int )'\0', max_out_str_len );

    ret = ::NS(TrackJob_extract_device_id_str)(
        conf_str.c_str(), &device_id_str[ 0 ], max_out_str_len );

    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( std::strcmp( &device_id_str[ 0 ], "0.0" ) == 0 );

//     conf_str = "0.0;a=b;#this is a comment";
//     std::memset( &device_id_str[ 0 ], ( int )'\0', max_out_str_len );
//
//     ret = ::NS(TrackJob_extract_device_id_str)(
//         conf_str.c_str(), &device_id_str[ 0 ], max_out_str_len );
//
//     ASSERT_TRUE( ret == 0 );
//     ASSERT_TRUE( std::strcmp( &device_id_str[ 0 ], "0.0" ) == 0 );
}

/* end: tests/sixtracklib/common/test_track_job_c99.cpp */
