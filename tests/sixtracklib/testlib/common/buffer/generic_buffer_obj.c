#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/testlib/common/buffer/generic_buffer_obj.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/testlib/common/random.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE )

void NS(GenericObj_init_random)( SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj)* obj )
{
    typedef SIXTRL_UINT8_T    u8_t;
    typedef SIXTRL_INT32_T   i32_t;
    typedef SIXTRL_UINT64_T  u64_t;
    typedef SIXTRL_REAL_T   real_t;

    if( obj != SIXTRL_NULLPTR )
    {
        u64_t const num_d_elements = obj->num_d;
        u64_t const num_e_elements = obj->num_e;

        i32_t const MIN_A     = INT32_C( -131072 );
        i32_t const MAX_A     = INT32_C( +131072 );
        i32_t const DELTA_A   = MAX_A - MIN_A;

        real_t const MIN_B    = ( real_t )-10.0;
        real_t const MAX_B    = ( real_t )+10.0;
        real_t const DELTA_B  = MAX_B - MIN_B;

        real_t const MIN_C    = ( real_t )0.0;
        real_t const MAX_C    = ( real_t )1.0;
        real_t const DELTA_C  = MAX_C - MIN_C;

        u8_t const MIN_D      = ( u8_t )0x01;
        u8_t const MAX_D      = ( u8_t )0x10;
        u8_t const DELTA_D    = MAX_D - MIN_D;

        real_t const MIN_E    = ( real_t )0.0;
        real_t const MAX_E    = ( real_t )1024.0;
        real_t const DELTA_E  = MAX_E - MIN_E;

        obj->a = MIN_A + DELTA_A * ( i32_t )NS(Random_genrand64_real1)();
        obj->b = MIN_B + DELTA_B * NS(Random_genrand64_real1)();

        obj->c[ 0 ] = MIN_C + DELTA_C * NS(Random_genrand64_real1)();
        obj->c[ 1 ] = MIN_C + DELTA_C * NS(Random_genrand64_real1)();
        obj->c[ 2 ] = MIN_C + DELTA_C * NS(Random_genrand64_real1)();
        obj->c[ 3 ] = MIN_C + DELTA_C * NS(Random_genrand64_real1)();

        if( ( num_d_elements > ( u64_t )0 ) && ( obj->d != SIXTRL_NULLPTR ) )
        {
            u64_t ii = ( u64_t )0u;

            for( ; ii < num_d_elements ; ++ii )
            {
                obj->d[ ii ] = MIN_D +
                    DELTA_D * ( u8_t )NS(Random_genrand64_real1)();
            }
        }

        if( ( num_e_elements > ( u64_t )0 ) && ( obj->e != SIXTRL_NULLPTR ) )
        {
            u64_t ii = ( u64_t )0u;

            for( ; ii < num_e_elements ; ++ii )
            {
                obj->e[ ii ] = MIN_E + DELTA_E * NS(Random_genrand64_real1)();
            }
        }
    }
}

#endif /* !defined( _GPUCODE ) */

/* end: tests/sixtracklib/testlib/common/generic_buffer_obj.c */
