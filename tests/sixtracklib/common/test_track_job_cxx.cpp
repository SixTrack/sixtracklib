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

TEST( CXX_TrackJobTests, DeviceIdAndConfigStrTests )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    std::string conf_str( "" );
    std::string device_id_str = st::TrackJob_extract_device_id_str(conf_str );
    ASSERT_TRUE( device_id_str.empty() );

    conf_str = "0.0";
    device_id_str = st::TrackJob_extract_device_id_str( conf_str );

    ASSERT_TRUE( device_id_str.compare( "0.0" ) == 0 );

    conf_str = "  0.0  ";
    device_id_str = st::TrackJob_extract_device_id_str( conf_str );

    ASSERT_TRUE( device_id_str.compare( "0.0" ) == 0 );

//     conf_str = "0.0;a=b;#this is a comment";
//     device_id_str = st::TrackJob_extract_device_id_str( conf_str );
//
//     ASSERT_TRUE( device_id_str.compare( "0.0" ) == 0 );
}

/* end: tests/sixtracklib/common/test_track_job_cxx.cpp */
