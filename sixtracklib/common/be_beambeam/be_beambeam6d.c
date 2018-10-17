#include "sixtracklib/common/be_beambeam/be_beambeam6d.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"

#if !defined( _GPUCODE )

SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam6D)* NS(BeamBeam6D_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const data_size )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam6D)*   ptr_to_beam_beam_t;

    NS(buffer_size_t) const num_dataptrs = 1u;

    NS(buffer_size_t) const offsets[] =
    {
        offsetof( NS(BeamBeam6D), data )
    };

    NS(buffer_size_t) const sizes[] =
    {
        sizeof( SIXTRL_REAL_T  )
    };

    NS(buffer_size_t) const counts[] = { data_size };


    NS(BeamBeam6D) beam_beam;
    NS(BeamBeam6D_preset)( &beam_beam );

    beam_beam.size = data_size;

    return ( ptr_to_beam_beam_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &beam_beam, sizeof( beam_beam ),
            NS(OBJECT_TYPE_BEAM_BEAM_6D), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam6D)* NS(BeamBeam6D_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const  data_size,
    NS(beambeam6d_real_ptr_t) SIXTRL_RESTRICT input_data )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam6D)*   ptr_to_beam_beam_t;

    NS(buffer_size_t) const num_dataptrs = 1u;

    NS(buffer_size_t) const offsets[] =
    {
        offsetof( NS(BeamBeam6D), data )
    };

    NS(buffer_size_t) const sizes[] =
    {
        sizeof( SIXTRL_REAL_T  )
    };

    NS(buffer_size_t) const counts[] = { data_size };

    NS(BeamBeam6D) beam_beam;
    beam_beam.size = data_size;
    beam_beam.data   = input_data;

    return ( ptr_to_beam_beam_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &beam_beam, sizeof( beam_beam ),
            NS(OBJECT_TYPE_BEAM_BEAM_6D), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam6D)*
NS(BeamBeam6D_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam6D) *const SIXTRL_RESTRICT orig )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam6D)* ptr_to_beam_beam_t;

    NS(buffer_size_t) const num_dataptrs = 1u;

    NS(buffer_size_t) const offsets[] =
    {
        offsetof( NS(BeamBeam6D), data )
    };

    NS(buffer_size_t) const sizes[] =
    {
        sizeof( SIXTRL_REAL_T  )
    };

    NS(buffer_size_t) const counts[] =
    {
        NS(BeamBeam6D_get_data_size)( orig )
    };

    NS(BeamBeam6D) beam_beam;
    beam_beam.size = NS(BeamBeam6D_get_data_size)( orig );
    beam_beam.data   = orig->data;

    return ( ptr_to_beam_beam_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &beam_beam, sizeof( beam_beam ),
            NS(OBJECT_TYPE_BEAM_BEAM_6D), num_dataptrs, offsets, sizes, counts ) );
}

#endif /* !defined( _GPUCODE ) */

/* end: sixtracklib/common/be_beambeam/be_beambeam6d.c */
