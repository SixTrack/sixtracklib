#include "sixtracklib/common/be_rfmultipole.h"
#include "sixtracklib/common/buffer.h"

SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole) const* NS(RFMultiPole_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    return NS(RFMultiPole_const_from_obj_index)( NS(Buffer_get_const_object)(
        buffer, index ) );
}

SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* NS(RFMultiPole_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    return NS(RFMultiPole_from_obj_index)( NS(Buffer_get_object)(
        buffer, index ) );
}

NS(arch_status_t) NS(RFMultiPole_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const slot_size )
{

}

NS(arch_status_t) NS(RFMultiPole_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const slot_size )
{

}

NS(arch_status_t) NS(RFMultiPole_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{

}

/* ------------------------------------------------------------------------- */

bool NS(RFMultiPole_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    ( void )buffer;
    ( void )order;
    ( void )ptr_requ_objects;
    ( void )ptr_requ_slots;
    ( void )ptr_requ_dataptrs;
    return false;
}

SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order )
{
    ( void )buffer;
    ( void )order;

    return SIXTRL_NULLPTR;
}

SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order,
    NS(rf_multipole_real_t) const voltage,
    NS(rf_multipole_real_t) const frequency,
    NS(rf_multipole_real_t) const lag,
    SIXTRL_ARGPTR_DEC NS(rf_multipole_real_t) const* SIXTRL_RESTRICT bal_values,
    SIXTRL_ARGPTR_DEC NS(rf_multipole_real_t) const* SIXTRL_RESTRICT p_values )
{
    ( void )buffer;
    ( void )order;
    ( void )voltage;
    ( void )frequency;
    ( void )lag;
    ( void )bal_values;
    ( void )p_values;

    return SIXTRL_NULLPTR;
}

SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{
    ( void )buffer;
    ( void )mpole;

    return SIXTRL_NULLPTR;
}

#endif /* !defined( _GPUCODE ) */
