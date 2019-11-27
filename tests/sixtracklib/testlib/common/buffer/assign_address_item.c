#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/testlib/common/buffer/assign_address_item.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer/assign_address_item.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

void NS(AssignAddressItem_print)( FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( item != SIXTRL_NULLPTR ) )
    {
        fprintf( fp,
                "|assign_address_item: | dest_elem_type_id   = %16d ;\r\n"
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
}

void NS(AssignAddressItem_print_out_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(AssignAddressItem)
        *const SIXTRL_RESTRICT item )
{
    NS(AssignAddressItem_print)( stdout, item );
}
