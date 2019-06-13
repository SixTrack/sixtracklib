#ifndef SIXTRACKLIB_COMMON_BUFFER_VARIANT_BUFFER_HPP_
#define SIXTRACKLIB_COMMON_BUFFER_VARIANT_BUFFER_HPP_

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdlib>
    #else /* defined( __cplusplus ) */
        #include <stddef.h>
        #include <stdlib.h>
        #include <stdint.h>
        #include <limits.h>
    #endif /* defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"

    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
        !defined( __CUDA_ARCH__ )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* C++, Host */

    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, Host */

typedef NS(buffer_size_t) NS(variant_size_t);

#if !defined( _GPUCODE )

SIXTRL_STATIC_VAR NS(object_type_id_t) const NS(VARIANT_TYPE_UNDEFINED) =
    ( NS(object_type_id_t) )SIXTRL_OBJECT_TYPE_UNDEFINED;

SIXTRL_STATIC_VAR NS(object_type_id_t) const
    NS(VARIANT_TYPE_CSTRING) = NS(OBJECT_TYPE_CSTRING);

SIXTRL_STATIC_VAR NS(object_type_id_t) const
    NS(VARIANT_TYPE_ARRAY)   = NS(OBJECT_TYPE_ARRAY);

SIXTRL_STATIC_VAR NS(object_type_id_t) const
    NS(VARIANT_TYPE_CHAR)    = ( NS(object_type_id_t) )0x40010001;

SIXTRL_STATIC_VAR NS(object_type_id_t) const
    NS(VARIANT_TYPE_UCHAR)   = ( NS(object_type_id_t) )0x40010002;

SIXTRL_STATIC_VAR NS(object_type_id_t) const
    NS(VARIANT_TYPE_INT8)    = NS(VARIANT_TYPE_CHAR);

SIXTRL_STATIC_VAR NS(object_type_id_t) const
    NS(VARIANT_TYPE_UINT8)   = NS(VARIANT_TYPE_UCHAR);

SIXTRL_STATIC_VAR NS(object_type_id_t) const
    NS(VARIANT_TYPE_INT16)   = ( NS(object_type_id_t) )0x40010003;

SIXTRL_STATIC_VAR NS(object_type_id_t) const
    NS(VARIANT_TYPE_UINT16)  = ( NS(object_type_id_t) )0x40010003;

SIXTRL_STATIC_VAR NS(object_type_id_t) const
    NS(VARIANT_TYPE_INT32)   = ( NS(object_type_id_t) )0x40010004;

SIXTRL_STATIC_VAR NS(object_type_id_t) const
    NS(VARIANT_TYPE_UINT32)  = ( NS(object_type_id_t) )0x40010005;

SIXTRL_STATIC_VAR NS(object_type_id_t) const
    NS(VARIANT_TYPE_INT64)   = ( NS(object_type_id_t) )0x40010006;

SIXTRL_STATIC_VAR NS(object_type_id_t) const
    NS(VARIANT_TYPE_UINT64)  = ( NS(object_type_id_t) )0x40010007;

SIXTRL_STATIC_VAR NS(object_type_id_t) const
    NS(VARIANT_TYPE_FLOAT32) = ( NS(object_type_id_t) )0x40010008;

SIXTRL_STATIC_VAR NS(object_type_id_t) const
    NS(VARIANT_TYPE_FLOAT64) = ( NS(object_type_id_t) )0x40010009;

SIXTRL_STATIC_VAR NS(object_type_id_t) const
    NS(VARIANT_TYPE_BOOL)    = ( NS(object_type_id_t) )0x4001000a;

#if !defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}
#endif /* C++, Host */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    typedef ::NS(variant_size_t)   variant_size_t;

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        VARIANT_TYPE_UNDEFINED = static_cast< object_type_id_t >(
            SIXTRL_OBJECT_TYPE_UNDEFINED );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        VARIANT_TYPE_CSTRING = SIXTRL_CXX_NAMESPACE::OBJECT_TYPE_CSTRING;

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        VARIANT_TYPE_ARRAY = SIXTRL_CXX_NAMESPACE::OBJECT_TYPE_ARRAY;

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        VARIANT_TYPE_CHAR= static_cast< object_type_id_t >( 0x40010001 );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        VARIANT_TYPE_UCHAR= static_cast< object_type_id_t >( 0x40010002 );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        VARIANT_TYPE_INT8) = SIXTRL_CXX_NAMESPACE::VARIANT_TYPE_CHAR;

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        VARIANT_TYPE_UINT8) = SIXTRL_CXX_NAMESPACE::VARIANT_TYPE_UCHAR;

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        VARIANT_TYPE_INT16= static_cast< object_type_id_t >( 0x40010003 );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        VARIANT_TYPE_UINT16= static_cast< object_type_id_t >( 0x40010003 );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        VARIANT_TYPE_INT32= static_cast< object_type_id_t >( 0x40010004 );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        VARIANT_TYPE_UINT32= static_cast< object_type_id_t >( 0x40010005 );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        VARIANT_TYPE_INT64= static_cast< object_type_id_t >( 0x40010006 );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        VARIANT_TYPE_UINT64= static_cast< object_type_id_t >( 0x40010007 );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        VARIANT_TYPE_FLOAT32= static_cast< object_type_id_t >( 0x40010008 );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        VARIANT_TYPE_FLOAT64= static_cast< object_type_id_t >( 0x40010009 );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        VARIANT_TYPE_BOOL= static_cast< object_type_id_t >( 0x4001000a );



    class VariantBuffer
    {
        private:

        public:

        typedef SIXTRL_CXX_NAMESPACE::Buffer    buffer_t;
        typedef buffer_t::size_type             size_type;
        typedef buffer_t::type_id_t             type_id_t;
        typedef buffer_t::c_api_t               c_buffer_t;

        explicit VariantBuffer( size_type const capacity = size_type{ 0 } );


        private:

        buffer_t        m_buffer;

    };

}

#endif /* defined( __cplusplus ) */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_BUFFER_VARIANT_BUFFER_HPP_ */

/* end: sixtracklib/common/buffer/variant_buffer.hpp */
