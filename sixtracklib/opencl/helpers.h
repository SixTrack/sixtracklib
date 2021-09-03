#ifndef SIXTRACKLIB_OPENCL_HELPERS_H__
#define SIXTRACKLIB_OPENCL_HELPERS_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #if defined( _GPUCODE ) && defined( __OPENCL_C_VERSION__ )
        #include "sixtracklib/opencl/opencl.h"
    #endif /* defined( _GPUCODE ) && defined( __OPENCL_C_VERSION__ ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_STATIC SIXTRL_FN SIXTRL_UINT64_T NS(Grid_global_id)( void ) SIXTRL_NOEXCEPT;
SIXTRL_STATIC SIXTRL_FN SIXTRL_UINT64_T NS(Grid_gloabl_size)( void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_UINT64_T NS(Grid_local_id)( void ) SIXTRL_NOEXCEPT;
SIXTRL_STATIC SIXTRL_FN SIXTRL_UINT64_T NS(Grid_local_size)( void ) SIXTRL_NOEXCEPT;

#if defined( _GPUCODE ) && defined( __OPENCL_VERSION__ ) && \
           ( __OPENCL_VERSION__ >= 120 )

    #if !defined( SIXTRL_SHARED_BUILD_ARRAY )
        #define SIXTRL_SHARED_BUILD_ARRAY( T, values, local_value ) \
        do {\
            unsigned int const local_id = ( unsigned int )get_local_id( 0 ); \
            values[ local_id ] = ( local_value ); \
            barrier( CLK_LOCAL_MEM_FENCE ); \
        } while( false )

    #endif /* !defined( SIXTRL_SHARED_BUILD_ARRAY ) */

    #if !defined( SIXTRL_SHARED_FIND_MAX_PER_W )
        #define SIXTRL_SHARED_FIND_MAX_PER_W( T, values, N_w, result ) \
        do {\
            unsigned int const local_id = ( unsigned int )get_local_id( 0 ); \
            unsigned int const n_wavefront = ( N_w ) * ( ( local_id ) / ( N_w ) ); \
            unsigned int const id_in_wavefront = ( local_id ) % ( N_w ); \
            unsigned int n = ( N_w ) >> 1u; \
            for( ; n > 0 ; n >>= 1u ) { \
                unsigned int const cmp_idx = local_id + n;\
                barrier( CLK_LOCAL_MEM_FENCE );\
                if( id_in_wavefront < n ) \
                    values[ local_id ] = max( values[ local_id ], values[ cmp_idx ] ); }\
            result = values[ n_wavefront ]; } while( false )

    #endif /* !defined( SIXTRL_SHARED_FIND_MAX_PER_W ) */

    #if !defined( SIXTRL_SHARED_OR_PER_W )
        #define SIXTRL_SHARED_OR_PER_W( T, values, flag, N_w, result ) \
        do {\
            unsigned int const local_id = ( unsigned int )get_local_id( 0 ); \
            unsigned int const n_wavefront = ( N_w ) * ( ( local_id ) / ( N_w ) ); \
            atomic_or( &values[ n_wavefront ], ( T )( flag ) ); \
            barrier( CLK_LOCAL_MEM_FENCE ); \
            result = ( values[ n_wavefront ] != 0 ); \
        } while( false )

    #endif /* !defined( SIXTRL_SAFE_OR_PER_W ) */

    /* ********************************************************************* */

    SIXTRL_INLINE SIXTRL_UINT64_T NS(Grid_global_id)() SIXTRL_NOEXCEPT {
        return ( SIXTRL_UINT64_T )get_global_id( 0 ); }

    SIXTRL_INLINE SIXTRL_UINT64_T NS(Grid_gloabl_size)() SIXTRL_NOEXCEPT {
        return ( SIXTRL_UINT64_T )get_global_size( 0 ); }

    SIXTRL_INLINE SIXTRL_UINT64_T NS(Grid_local_id)() SIXTRL_NOEXCEPT {
        return ( SIXTRL_UINT64_T )get_local_id( 0 ); }

    SIXTRL_INLINE SIXTRL_UINT64_T NS(Grid_local_size)() SIXTRL_NOEXCEPT {
        return ( SIXTRL_UINT64_T )get_local_size( 0 ); }

#else /* defined( _GPUCODE ) && defined( __OPENCL_VERSION__ ) &&
                ( __OPENCL_VERSION__ >= 120 ) */

    #if !defined( SIXTRL_SHARED_BUILD_ARRAY ) && defined( __OPENCL_VERSION__ )
        #define SIXTRL_SHARED_BUILD_ARRAY( T, values, local_value ) \
        do { } while( false )

    #endif /* !defined( SIXTRL_SHARED_BUILD_ARRAY ) */

    #if !defined( SIXTRL_SHARED_FIND_MAX_PER_W ) && defined( __OPENCL_VERSION__ )
        #define SIXTRL_SHARED_FIND_MAX_PER_W( T, values, local_id, N_w, result ) \
        do { result = ( T )0u; } while( false )

    #endif /* !defined( SIXTRL_SHARED_FIND_MAX_PER_W ) */

    #if !defined( SIXTRL_SHARED_OR_PER_W ) && defined( __OPENCL_VERSION__ )
        #define SIXTRL_SHARED_OR_PER_W( T, values, flag, local_id, N_w, result ) \
        do { result = ( flag ); } while( false )

    #endif /* !defined( SIXTRL_SAFE_OR_PER_W ) */

    SIXTRL_INLINE SIXTRL_UINT64_T NS(Grid_global_id)() SIXTRL_NOEXCEPT {
        return ( SIXTRL_UINT64_T )0u; }

    SIXTRL_INLINE SIXTRL_UINT64_T NS(Grid_gloabl_size)() SIXTRL_NOEXCEPT {
        return ( SIXTRL_UINT64_T )0u; }

    SIXTRL_INLINE SIXTRL_UINT64_T NS(Grid_local_id)() SIXTRL_NOEXCEPT {
        return ( SIXTRL_UINT64_T )0u; }

    SIXTRL_INLINE SIXTRL_UINT64_T NS(Grid_local_size)() SIXTRL_NOEXCEPT {
        return ( SIXTRL_UINT64_T )0u; }

#endif /* defined( _GPUCODE ) && defined( __OPENCL_VERSION__ ) &&
                 ( __OPENCL_VERSION__ >= 120 ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRACKLIB_OPENCL_HELPERS_H__ */
