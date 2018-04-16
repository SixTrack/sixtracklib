#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <gtest/gtest.h>

#if defined( __NAMESPACE )
    #define __SAVED_NAMESPACE __NAMESPACE
    #undef  __NAMESPACE     
#endif /* !defiend( __NAMESPACE ) */

#if !defined( __NAMESPACE )
    #define __NAMESPACE st_    
#endif /* !defiend( __NAMESPACE ) */

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/details/tools.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

TEST( CommonToolTests, GreatestCommonDivisor )
{   
    ASSERT_TRUE( st_greatest_common_divisor(  1,  1 ) ==  1 );
    
    ASSERT_TRUE( st_greatest_common_divisor(  4,  1 ) ==  1 );
    ASSERT_TRUE( st_greatest_common_divisor(  4,  2 ) ==  2 );
    ASSERT_TRUE( st_greatest_common_divisor(  4,  3 ) ==  1 );
    ASSERT_TRUE( st_greatest_common_divisor(  4,  4 ) ==  4 );
    ASSERT_TRUE( st_greatest_common_divisor(  4,  5 ) ==  1 );
    ASSERT_TRUE( st_greatest_common_divisor(  4,  6 ) ==  2 );
    ASSERT_TRUE( st_greatest_common_divisor(  4,  7 ) ==  1 );
    ASSERT_TRUE( st_greatest_common_divisor(  4,  8 ) ==  4 );
    
    ASSERT_TRUE( st_greatest_common_divisor(  1,  4 ) ==  1 );
    ASSERT_TRUE( st_greatest_common_divisor(  2,  4 ) ==  2 );
    ASSERT_TRUE( st_greatest_common_divisor(  3,  4 ) ==  1 );
    ASSERT_TRUE( st_greatest_common_divisor(  4,  4 ) ==  4 );
    ASSERT_TRUE( st_greatest_common_divisor(  5,  4 ) ==  1 );
    ASSERT_TRUE( st_greatest_common_divisor(  6,  4 ) ==  2 );
    ASSERT_TRUE( st_greatest_common_divisor(  7,  4 ) ==  1 );
    ASSERT_TRUE( st_greatest_common_divisor(  8,  4 ) ==  4 );
    
    ASSERT_TRUE( st_greatest_common_divisor( 63,  1 ) ==  1 );
    ASSERT_TRUE( st_greatest_common_divisor( 63,  2 ) ==  1 );
    ASSERT_TRUE( st_greatest_common_divisor( 63,  3 ) ==  3 );
    ASSERT_TRUE( st_greatest_common_divisor( 63,  4 ) ==  1 );
    ASSERT_TRUE( st_greatest_common_divisor( 63,  5 ) ==  1 );
    ASSERT_TRUE( st_greatest_common_divisor( 63,  6 ) ==  3 );
    ASSERT_TRUE( st_greatest_common_divisor( 63,  7 ) ==  7 );
    ASSERT_TRUE( st_greatest_common_divisor( 63,  9 ) ==  9 );
    ASSERT_TRUE( st_greatest_common_divisor( 63, 21 ) == 21 );
    ASSERT_TRUE( st_greatest_common_divisor( 63, 63 ) == 63 );
}


TEST( CommonToolTests, LeastCommonMultiple )
{
    ASSERT_TRUE( st_least_common_multiple(  1,  1 ) ==  1 );
    
    ASSERT_TRUE( st_least_common_multiple(  4,  1 ) ==  4 );
    ASSERT_TRUE( st_least_common_multiple(  4,  2 ) ==  4 );
    ASSERT_TRUE( st_least_common_multiple(  4,  3 ) == 12 );
    ASSERT_TRUE( st_least_common_multiple(  4,  4 ) ==  4 );
    ASSERT_TRUE( st_least_common_multiple(  4,  5 ) == 20 );
    ASSERT_TRUE( st_least_common_multiple(  4,  6 ) == 12 );
    ASSERT_TRUE( st_least_common_multiple(  4,  7 ) == 28 );
    ASSERT_TRUE( st_least_common_multiple(  4,  8 ) ==  8 );
    
    ASSERT_TRUE( st_least_common_multiple(  1,  4 ) ==  4 );
    ASSERT_TRUE( st_least_common_multiple(  2,  4 ) ==  4 );
    ASSERT_TRUE( st_least_common_multiple(  3,  4 ) == 12 );
    ASSERT_TRUE( st_least_common_multiple(  4,  4 ) ==  4 );
    ASSERT_TRUE( st_least_common_multiple(  5,  4 ) == 20 );
    ASSERT_TRUE( st_least_common_multiple(  6,  4 ) == 12 );
    ASSERT_TRUE( st_least_common_multiple(  7,  4 ) == 28 );
    ASSERT_TRUE( st_least_common_multiple(  8,  4 ) ==  8 );
    
    ASSERT_TRUE( st_least_common_multiple(  7,  3 ) == 21 );
    ASSERT_TRUE( st_least_common_multiple(  7,  7 ) ==  7 );
    ASSERT_TRUE( st_least_common_multiple(  7,  9 ) == 63 );
    
    ASSERT_TRUE( st_least_common_multiple(  3,  7 ) == 21 );
    ASSERT_TRUE( st_least_common_multiple(  7,  7 ) ==  7 );
    ASSERT_TRUE( st_least_common_multiple(  9,  7 ) == 63 );
}

/* end: sixtracklib/common/tests/test_tools.cpp */
