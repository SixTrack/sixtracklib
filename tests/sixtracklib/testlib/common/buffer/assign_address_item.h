#ifndef SIXTRACKLIB_TESTLIB_COMMON_BUFFER_ASSIGN_ADDRESS_ITEM_C99_H__
#define SIXTRACKLIB_TESTLIB_COMMON_BUFFER_ASSIGN_ADDRESS_ITEM_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdio.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/assign_address_item.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN void NS(AssignAddressItem_print_out)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(AssignAddressItem_print)(
    FILE* SIXTRL_RESTRICT fp, SIXTRL_BUFFER_DATAPTR_DEC const
        NS(AssignAddressItem) *const SIXTRL_RESTRICT item );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(AssignAddressItem_print_out_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

/* ************************************************************************* */
/* ****              Inline function implementation                     **** */
/* ************************************************************************* */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#if defined( _GPUCODE )

SIXTRL_INLINE void NS(AssignAddressItem_print_out)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item )
{
    SIXTRL_ASSERT( item != SIXTRL_NULLPTR );
    printf( "|assign_address_item: | dest_elem_type_id   = %16d ;\r\n"
            "                      | dest_buffer_id      = %16d ;\r\n"
            "                      | dest_elem_index     = %16d ;\r\n"
            "                      | dest_pointer_offset = %16d ;\r\n"
            "                      | src_elem_type_id    = %16d ;\r\n"
            "                      | src_buffer_id       = %16d ;\r\n"
            "                      | src_elem_index      = %16d ;\r\n"
            "                      | src_pointer_offset  = %16d ;\r\n",
            ( int )NS(AssignAddressItem_dest_elem_type_id)( item ),
            ( int )NS(AssignAddressItem_dest_buffer_id)( item ),
            ( int )NS(AssignAddressItem_dest_elem_index)( item ),
            ( int )NS(AssignAddressItem_dest_pointer_offset)( item ),
            ( int )NS(AssignAddressItem_src_elem_type_id)( item ),
            ( int )NS(AssignAddressItem_src_buffer_id)( item ),
            ( int )NS(AssignAddressItem_src_elem_index)( item ),
            ( int )NS(AssignAddressItem_src_pointer_offset)( item ) );
}

#else /* defined( _GPUCODE ) */

SIXTRL_INLINE void NS(AssignAddressItem_print_out)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item )
{
    NS(AssignAddressItem_print)( stdout, item );
}

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /*define SIXTRACKLIB_TESTLIB_COMMON_BUFFER_ASSIGN_ADDRESS_ITEM_C99_H__*/
