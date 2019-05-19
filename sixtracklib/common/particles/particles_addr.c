#include "sixtracklib/common/particles/particles_addr.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"

NS(arch_status_t) NS(ParticlesAddr_prepare_buffer_based_on_particles_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT paddr_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer)
        *const SIXTRL_RESTRICT particles_buffer )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* obj_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles)* ptr_particles_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)* ptr_paddr_t;

    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( paddr_buffer != SIXTRL_NULLPTR ) &&
        ( particles_buffer != SIXTRL_NULLPTR ) )
    {
        buf_size_t const num_psets =
            NS(Particles_buffer_get_num_of_particle_blocks)( particles_buffer );

        buf_size_t num_psets_found = ( buf_size_t )0u;

        obj_iter_t it = NS(Buffer_get_const_objects_begin)(
            particles_buffer );

        obj_iter_t end = NS(Buffer_get_const_objects_begin)(
            particles_buffer );

        if( NS(Buffer_get_num_of_objects)( paddr_buffer ) > ( buf_size_t )0u )
        {
            NS(Buffer_clear)( paddr_buffer, true );
            NS(Buffer_reset)( paddr_buffer );
        }

        SIXTRL_ASSERT( NS(Buffer_get_num_of_objects)( paddr_buffer ) ==
            ( buf_size_t )0u );

        for( ; it != end ; ++it )
        {
            ptr_particles_t p = NS(BufferIndex_get_particles)( it );

            if( p != SIXTRL_NULLPTR )
            {
                ptr_paddr_t paddr = NS(ParticlesAddr_new)(
                    paddr_buffer, NS(Particles_get_num_of_particles)( p ) );

                if( paddr != SIXTRL_NULLPTR )
                {
                    ++num_psets_found;
                }
            }
            else
            {
                ptr_paddr_t paddr = NS(ParticlesAddr_new)(
                    paddr_buffer, ( buf_size_t )0u );
            }
        }

        status = ( ( num_psets_found == num_psets ) &&
                   ( NS(Buffer_get_num_of_objects)( particles_buffer ) ==
                     NS(Buffer_get_num_of_objects)( paddr_buffer ) ) )
            ? NS(ARCH_STATUS_SUCCESS) : NS(ARCH_STATUS_GENERAL_FAILURE);
    }

    return status;
}


NS(arch_status_t) NS(ParticlesAddr_buffer_store_addresses)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT paddr_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const index )
{
    SIXTRL_ASSERT( NS(Buffer_get_slot_size)( paddr_buffer ) ==
                   NS(Buffer_get_slot_size)( particles_buffer ) );

    return NS(Particles_managed_buffer_store_addresses)(
        ::NS(Buffer_get_data_begin)( paddr_buffer ),
        ::NS(Buffer_get_data_begin)( particles_buffer ),
        index, ::NS(Buffer_get_slot_size)( paddr_buffer ) );
}

NS(arch_status_t) NS(ParticlesAddr_buffer_store_all_addresses)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT paddr_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT particles_buffer )
{
    SIXTRL_ASSSERT( NS(Buffer_get_slot_size)( paddr_buffer ),
                    NS(Buffer_get_slot_size)( particles_buffer ) );

    return NS(Particles_managed_buffer_store_all_addresses)(
        NS(Buffer_get_data_begin)( paddr_buffer ),
        NS(Buffer_get_data_begin)( particles_buffer ),
        NS(Buffer_get_slot_size)( particles_buffer ) );
}

void NS(ParticlesAddr_buffer_remap_adresses)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const buffer_index,
    NS(buffer_addr_diff_t) const addr_offset )
{
    NS(ParticlesAddr_managed_buffer_remap_addresses)(
        NS(Buffer_get_data_begin)( buffer ), buffer_index, addr_offset,
        NS(Buffer_get_slot_size)( buffer ) );
}

void NS(ParticlesAddr_buffer_all_remap_adresses)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_addr_diff_t) const addr_offset )
{
    NS(ParticlesAddr_managed_buffer_all_remap_addresses)(
        NS(Buffer_get_data_begin)( buffer ), addr_offset,
        NS(Buffer_get_slot_size)( buffer ) );
}

/* end: sixtracklib/common/particles/particles_addr.c */
