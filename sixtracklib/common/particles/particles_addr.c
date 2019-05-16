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

/* end: sixtracklib/common/particles/particles_addr.c */
