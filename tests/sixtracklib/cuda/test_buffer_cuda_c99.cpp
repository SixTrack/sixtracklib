#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/path.h"
#include "sixtracklib/common/buffer.h"

#include "sixtracklib/cuda/impl/cuda_buffer_generic_obj_kernel.cuh"


TEST( C99_Cuda_BufferTests,
      InitWithGenericObjDataCopyToDeviceCopyBackCmpSingleThread )
{
    using buffer_t      = ::st_Buffer;
    using size_t        = ::st_buffer_size_t;

    buffer_t* orig_buffer = ::st_Buffer_new_from_file(
        ::st_PATH_TO_TEST_GENERIC_OBJ_BUFFER_DATA );
    ASSERT_TRUE( orig_buffer != nullptr );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( orig_buffer ) > size_t{ 0 } );

    size_t const orig_buffer_size = ::st_Buffer_get_size( orig_buffer );
    ASSERT_TRUE( orig_buffer_size > size_t{ 0 } );

    buffer_t* copy_buffer = ::st_Buffer_new( 4096u );
    ASSERT_TRUE( copy_buffer != nullptr );
    ASSERT_TRUE( copy_buffer != orig_buffer );
    ASSERT_TRUE( st_Buffer_get_data_begin_addr( orig_buffer ) !=
                 st_Buffer_get_data_begin_addr( copy_buffer ) );

    int success = ::st_Buffer_reserve( copy_buffer,
        ::st_Buffer_get_max_num_of_objects( orig_buffer ),
        ::st_Buffer_get_max_num_of_slots( orig_buffer ),
        ::st_Buffer_get_max_num_of_dataptrs( orig_buffer ),
        ::st_Buffer_get_max_num_of_garbage_ranges( orig_buffer ) );

    ASSERT_TRUE( success == 0 );

    unsigned char const* orig_buffer_begin =
        ::st_Buffer_get_const_data_begin( orig_buffer );

    ASSERT_TRUE( orig_buffer_begin != nullptr );

    unsigned char* copy_buffer_begin =
        ::st_Buffer_get_data_begin( copy_buffer );

    ASSERT_TRUE( copy_buffer_begin != nullptr );
    ASSERT_TRUE( copy_buffer_begin != orig_buffer_begin );

    /* --------------------------------------------------------------------- */

    int num_devices = 0;
    cudaError_t cu_err = cudaGetDeviceCount( &num_devices );
    ASSERT_TRUE( cu_err == cudaSuccess );

    if( num_devices > 0 )
    {
        int device_id = 0;

        for( ; device_id < num_devices ; ++device_id )
        {
            /* ---------------------------------------------------------------- */
            /* prepare copy buffer */

            ::st_Buffer_clear( copy_buffer, true );

            auto obj_it  = st_Buffer_get_const_objects_begin( orig_buffer );
            auto obj_end = st_Buffer_get_const_objects_end( orig_buffer );

            for( ; obj_it != obj_end ; ++obj_it )
            {
                ::st_GenericObj const* orig_obj = reinterpret_cast<
                    ::st_GenericObj const* >( static_cast< uintptr_t >(
                        ::st_Object_get_begin_addr( obj_it ) ) );

                ASSERT_TRUE( orig_obj != nullptr );
                ASSERT_TRUE( orig_obj->type_id ==
                             ::st_Object_get_type_id( obj_it ) );

                ::st_GenericObj* copy_obj = ::st_GenericObj_new( copy_buffer,
                    orig_obj->type_id, orig_obj->num_d, orig_obj->num_e );

                ASSERT_TRUE( copy_obj != nullptr );
                ASSERT_TRUE( orig_obj->type_id == copy_obj->type_id );
                ASSERT_TRUE( orig_obj->num_d   == copy_obj->num_d );
                ASSERT_TRUE( orig_obj->num_e   == copy_obj->num_e );
                ASSERT_TRUE( copy_obj->d != nullptr );
                ASSERT_TRUE( copy_obj->e != nullptr );
            }

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_buffer ) ==
                         ::st_Buffer_get_num_of_objects( orig_buffer ) );

            unsigned int const num_objects =
                ::st_Buffer_get_num_of_objects( copy_buffer );

            int32_t success_flag = int32_t{ 0 };

            ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_buffer ) );
            ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_buffer ) );

            ASSERT_TRUE( !::st_ManagedBuffer_needs_remapping(
                orig_buffer_begin, ::st_Buffer_get_slot_size( orig_buffer ) ) );

            ASSERT_TRUE( !::st_ManagedBuffer_needs_remapping(
                copy_buffer_begin, ::st_Buffer_get_slot_size( copy_buffer ) ) );

            /* ------------------------------------------------------------- */

            cudaDeviceProp properties;

            cu_err = cudaSetDevice( device_id );
            ASSERT_TRUE( cu_err == cudaSuccess );

            cu_err = cudaGetDeviceProperties( &properties, device_id );
            ASSERT_TRUE( cu_err == cudaSuccess );

            std::cout << "Device # " << std::setw( 3 )
                      << device_id   << "\r\n"
                      << "Name   : " << std::setw( 20 )
                      << properties.name << "\r\n" << std::endl;

            /* ------------------------------------------------------------- */

            int success = ::st_Run_test_buffer_generic_obj_kernel_on_cuda(
                dim3{ 1 }, dim3{ 1 }, orig_buffer_begin, copy_buffer_begin );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_buffer ) );
            ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_buffer ) ==
                         ::st_Buffer_get_num_of_objects( orig_buffer ) );

            obj_it      = ::st_Buffer_get_const_objects_begin( orig_buffer );
            obj_end     = ::st_Buffer_get_const_objects_end( orig_buffer );
            auto cmp_it = ::st_Buffer_get_const_objects_begin( copy_buffer );

            for( ; obj_it != obj_end ; ++obj_it, ++cmp_it )
            {
                ::st_GenericObj const* orig_obj = reinterpret_cast<
                    ::st_GenericObj const* >( static_cast< uintptr_t >(
                        ::st_Object_get_begin_addr( obj_it ) ) );

                ::st_GenericObj const* cmp_obj = reinterpret_cast<
                    ::st_GenericObj const* >( static_cast< uintptr_t >(
                        ::st_Object_get_begin_addr( cmp_it ) ) );

                ASSERT_TRUE( orig_obj != nullptr );
                ASSERT_TRUE( cmp_obj  != nullptr );
                ASSERT_TRUE( cmp_obj  != orig_obj );

                ASSERT_TRUE( orig_obj->type_id == cmp_obj->type_id );
                ASSERT_TRUE( orig_obj->num_d   == cmp_obj->num_d );
                ASSERT_TRUE( orig_obj->num_e   == cmp_obj->num_e );
                ASSERT_TRUE( orig_obj->a       == cmp_obj->a );

                ASSERT_TRUE( std::fabs( orig_obj->a - cmp_obj->a ) <
                    std::numeric_limits< double >::epsilon() );

                for( std::size_t ii = 0 ; ii < 4u ; ++ii )
                {
                    ASSERT_TRUE( std::fabs( orig_obj->c[ ii ] - cmp_obj->c[ ii ] ) <
                        std::numeric_limits< double >::epsilon() );
                }

                if( orig_obj->num_d > 0u )
                {
                    for( std::size_t ii = 0u ; ii < orig_obj->num_d ; ++ii )
                    {
                        ASSERT_TRUE( orig_obj->d[ ii ] == cmp_obj->d[ ii ] );
                    }
                }

                if( orig_obj->num_e > 0u )
                {
                    for( std::size_t ii = 0u ; ii < orig_obj->num_e ; ++ii )
                    {
                        ASSERT_TRUE( std::fabs( orig_obj->e[ ii ] - cmp_obj->e[ ii ] )
                            < std::numeric_limits< double >::epsilon() );
                    }
                }
            }

            /* ------------------------------------------------------------- */

            success = ::st_Run_test_buffer_generic_obj_kernel_on_cuda(
                dim3{ num_objects }, dim3{ 1 }, orig_buffer_begin,
                    copy_buffer_begin );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_buffer ) );
            ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_buffer ) ==
                         ::st_Buffer_get_num_of_objects( orig_buffer ) );

            obj_it  = ::st_Buffer_get_const_objects_begin( orig_buffer );
            obj_end = ::st_Buffer_get_const_objects_end( orig_buffer );
            cmp_it  = ::st_Buffer_get_const_objects_begin( copy_buffer );

            for( ; obj_it != obj_end ; ++obj_it, ++cmp_it )
            {
                ::st_GenericObj const* orig_obj = reinterpret_cast<
                    ::st_GenericObj const* >( static_cast< uintptr_t >(
                        ::st_Object_get_begin_addr( obj_it ) ) );

                ::st_GenericObj const* cmp_obj = reinterpret_cast<
                    ::st_GenericObj const* >( static_cast< uintptr_t >(
                        ::st_Object_get_begin_addr( cmp_it ) ) );

                ASSERT_TRUE( orig_obj != nullptr );
                ASSERT_TRUE( cmp_obj  != nullptr );
                ASSERT_TRUE( cmp_obj  != orig_obj );

                ASSERT_TRUE( orig_obj->type_id == cmp_obj->type_id );
                ASSERT_TRUE( orig_obj->num_d   == cmp_obj->num_d );
                ASSERT_TRUE( orig_obj->num_e   == cmp_obj->num_e );
                ASSERT_TRUE( orig_obj->a       == cmp_obj->a );

                ASSERT_TRUE( std::fabs( orig_obj->a - cmp_obj->a ) <
                    std::numeric_limits< double >::epsilon() );

                for( std::size_t ii = 0 ; ii < 4u ; ++ii )
                {
                    ASSERT_TRUE( std::fabs( orig_obj->c[ ii ] - cmp_obj->c[ ii ] ) <
                        std::numeric_limits< double >::epsilon() );
                }

                if( orig_obj->num_d > 0u )
                {
                    for( std::size_t ii = 0u ; ii < orig_obj->num_d ; ++ii )
                    {
                        ASSERT_TRUE( orig_obj->d[ ii ] == cmp_obj->d[ ii ] );
                    }
                }

                if( orig_obj->num_e > 0u )
                {
                    for( std::size_t ii = 0u ; ii < orig_obj->num_e ; ++ii )
                    {
                        ASSERT_TRUE( std::fabs( orig_obj->e[ ii ] - cmp_obj->e[ ii ] )
                            < std::numeric_limits< double >::epsilon() );
                    }
                }
            }

            /* ------------------------------------------------------------- */

            success = ::st_Run_test_buffer_generic_obj_kernel_on_cuda(
                dim3{ 128 }, dim3{ 1 }, orig_buffer_begin,
                    copy_buffer_begin );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_buffer ) );
            ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_buffer ) ==
                         ::st_Buffer_get_num_of_objects( orig_buffer ) );

            obj_it  = ::st_Buffer_get_const_objects_begin( orig_buffer );
            obj_end = ::st_Buffer_get_const_objects_end( orig_buffer );
            cmp_it  = ::st_Buffer_get_const_objects_begin( copy_buffer );

            for( ; obj_it != obj_end ; ++obj_it, ++cmp_it )
            {
                ::st_GenericObj const* orig_obj = reinterpret_cast<
                    ::st_GenericObj const* >( static_cast< uintptr_t >(
                        ::st_Object_get_begin_addr( obj_it ) ) );

                ::st_GenericObj const* cmp_obj = reinterpret_cast<
                    ::st_GenericObj const* >( static_cast< uintptr_t >(
                        ::st_Object_get_begin_addr( cmp_it ) ) );

                ASSERT_TRUE( orig_obj != nullptr );
                ASSERT_TRUE( cmp_obj  != nullptr );
                ASSERT_TRUE( cmp_obj  != orig_obj );

                ASSERT_TRUE( orig_obj->type_id == cmp_obj->type_id );
                ASSERT_TRUE( orig_obj->num_d   == cmp_obj->num_d );
                ASSERT_TRUE( orig_obj->num_e   == cmp_obj->num_e );
                ASSERT_TRUE( orig_obj->a       == cmp_obj->a );

                ASSERT_TRUE( std::fabs( orig_obj->a - cmp_obj->a ) <
                    std::numeric_limits< double >::epsilon() );

                for( std::size_t ii = 0 ; ii < 4u ; ++ii )
                {
                    ASSERT_TRUE( std::fabs( orig_obj->c[ ii ] - cmp_obj->c[ ii ] ) <
                        std::numeric_limits< double >::epsilon() );
                }

                if( orig_obj->num_d > 0u )
                {
                    for( std::size_t ii = 0u ; ii < orig_obj->num_d ; ++ii )
                    {
                        ASSERT_TRUE( orig_obj->d[ ii ] == cmp_obj->d[ ii ] );
                    }
                }

                if( orig_obj->num_e > 0u )
                {
                    for( std::size_t ii = 0u ; ii < orig_obj->num_e ; ++ii )
                    {
                        ASSERT_TRUE( std::fabs( orig_obj->e[ ii ] - cmp_obj->e[ ii ] )
                            < std::numeric_limits< double >::epsilon() );
                    }
                }
            }

            /* ------------------------------------------------------------- */

            success = ::st_Run_test_buffer_generic_obj_kernel_on_cuda(
                dim3{ 1 }, dim3{ num_objects }, orig_buffer_begin,
                    copy_buffer_begin );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_buffer ) );
            ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_buffer ) ==
                         ::st_Buffer_get_num_of_objects( orig_buffer ) );

            obj_it  = ::st_Buffer_get_const_objects_begin( orig_buffer );
            obj_end = ::st_Buffer_get_const_objects_end( orig_buffer );
            cmp_it  = ::st_Buffer_get_const_objects_begin( copy_buffer );

            for( ; obj_it != obj_end ; ++obj_it, ++cmp_it )
            {
                ::st_GenericObj const* orig_obj = reinterpret_cast<
                    ::st_GenericObj const* >( static_cast< uintptr_t >(
                        ::st_Object_get_begin_addr( obj_it ) ) );

                ::st_GenericObj const* cmp_obj = reinterpret_cast<
                    ::st_GenericObj const* >( static_cast< uintptr_t >(
                        ::st_Object_get_begin_addr( cmp_it ) ) );

                ASSERT_TRUE( orig_obj != nullptr );
                ASSERT_TRUE( cmp_obj  != nullptr );
                ASSERT_TRUE( cmp_obj  != orig_obj );

                ASSERT_TRUE( orig_obj->type_id == cmp_obj->type_id );
                ASSERT_TRUE( orig_obj->num_d   == cmp_obj->num_d );
                ASSERT_TRUE( orig_obj->num_e   == cmp_obj->num_e );
                ASSERT_TRUE( orig_obj->a       == cmp_obj->a );

                ASSERT_TRUE( std::fabs( orig_obj->a - cmp_obj->a ) <
                    std::numeric_limits< double >::epsilon() );

                for( std::size_t ii = 0 ; ii < 4u ; ++ii )
                {
                    ASSERT_TRUE( std::fabs( orig_obj->c[ ii ] - cmp_obj->c[ ii ] ) <
                        std::numeric_limits< double >::epsilon() );
                }

                if( orig_obj->num_d > 0u )
                {
                    for( std::size_t ii = 0u ; ii < orig_obj->num_d ; ++ii )
                    {
                        ASSERT_TRUE( orig_obj->d[ ii ] == cmp_obj->d[ ii ] );
                    }
                }

                if( orig_obj->num_e > 0u )
                {
                    for( std::size_t ii = 0u ; ii < orig_obj->num_e ; ++ii )
                    {
                        ASSERT_TRUE( std::fabs( orig_obj->e[ ii ] - cmp_obj->e[ ii ] )
                            < std::numeric_limits< double >::epsilon() );
                    }
                }
            }

            /* ------------------------------------------------------------- */

            success = ::st_Run_test_buffer_generic_obj_kernel_on_cuda(
                dim3{ 1 }, dim3{ 128 }, orig_buffer_begin,
                    copy_buffer_begin );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_buffer ) );
            ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_buffer ) ==
                         ::st_Buffer_get_num_of_objects( orig_buffer ) );

            obj_it  = ::st_Buffer_get_const_objects_begin( orig_buffer );
            obj_end = ::st_Buffer_get_const_objects_end( orig_buffer );
            cmp_it  = ::st_Buffer_get_const_objects_begin( copy_buffer );

            for( ; obj_it != obj_end ; ++obj_it, ++cmp_it )
            {
                ::st_GenericObj const* orig_obj = reinterpret_cast<
                    ::st_GenericObj const* >( static_cast< uintptr_t >(
                        ::st_Object_get_begin_addr( obj_it ) ) );

                ::st_GenericObj const* cmp_obj = reinterpret_cast<
                    ::st_GenericObj const* >( static_cast< uintptr_t >(
                        ::st_Object_get_begin_addr( cmp_it ) ) );

                ASSERT_TRUE( orig_obj != nullptr );
                ASSERT_TRUE( cmp_obj  != nullptr );
                ASSERT_TRUE( cmp_obj  != orig_obj );

                ASSERT_TRUE( orig_obj->type_id == cmp_obj->type_id );
                ASSERT_TRUE( orig_obj->num_d   == cmp_obj->num_d );
                ASSERT_TRUE( orig_obj->num_e   == cmp_obj->num_e );
                ASSERT_TRUE( orig_obj->a       == cmp_obj->a );

                ASSERT_TRUE( std::fabs( orig_obj->a - cmp_obj->a ) <
                    std::numeric_limits< double >::epsilon() );

                for( std::size_t ii = 0 ; ii < 4u ; ++ii )
                {
                    ASSERT_TRUE( std::fabs( orig_obj->c[ ii ] - cmp_obj->c[ ii ] ) <
                        std::numeric_limits< double >::epsilon() );
                }

                if( orig_obj->num_d > 0u )
                {
                    for( std::size_t ii = 0u ; ii < orig_obj->num_d ; ++ii )
                    {
                        ASSERT_TRUE( orig_obj->d[ ii ] == cmp_obj->d[ ii ] );
                    }
                }

                if( orig_obj->num_e > 0u )
                {
                    for( std::size_t ii = 0u ; ii < orig_obj->num_e ; ++ii )
                    {
                        ASSERT_TRUE( std::fabs( orig_obj->e[ ii ] - cmp_obj->e[ ii ] )
                            < std::numeric_limits< double >::epsilon() );
                    }
                }
            }

            /* ------------------------------------------------------------- */

            success = ::st_Run_test_buffer_generic_obj_kernel_on_cuda(
                dim3{ 128 }, dim3{ 128 }, orig_buffer_begin,
                    copy_buffer_begin );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_buffer ) );
            ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_buffer ) ==
                         ::st_Buffer_get_num_of_objects( orig_buffer ) );

            obj_it  = ::st_Buffer_get_const_objects_begin( orig_buffer );
            obj_end = ::st_Buffer_get_const_objects_end( orig_buffer );
            cmp_it  = ::st_Buffer_get_const_objects_begin( copy_buffer );

            for( ; obj_it != obj_end ; ++obj_it, ++cmp_it )
            {
                ::st_GenericObj const* orig_obj = reinterpret_cast<
                    ::st_GenericObj const* >( static_cast< uintptr_t >(
                        ::st_Object_get_begin_addr( obj_it ) ) );

                ::st_GenericObj const* cmp_obj = reinterpret_cast<
                    ::st_GenericObj const* >( static_cast< uintptr_t >(
                        ::st_Object_get_begin_addr( cmp_it ) ) );

                ASSERT_TRUE( orig_obj != nullptr );
                ASSERT_TRUE( cmp_obj  != nullptr );
                ASSERT_TRUE( cmp_obj  != orig_obj );

                ASSERT_TRUE( orig_obj->type_id == cmp_obj->type_id );
                ASSERT_TRUE( orig_obj->num_d   == cmp_obj->num_d );
                ASSERT_TRUE( orig_obj->num_e   == cmp_obj->num_e );
                ASSERT_TRUE( orig_obj->a       == cmp_obj->a );

                ASSERT_TRUE( std::fabs( orig_obj->a - cmp_obj->a ) <
                    std::numeric_limits< double >::epsilon() );

                for( std::size_t ii = 0 ; ii < 4u ; ++ii )
                {
                    ASSERT_TRUE( std::fabs( orig_obj->c[ ii ] - cmp_obj->c[ ii ] ) <
                        std::numeric_limits< double >::epsilon() );
                }

                if( orig_obj->num_d > 0u )
                {
                    for( std::size_t ii = 0u ; ii < orig_obj->num_d ; ++ii )
                    {
                        ASSERT_TRUE( orig_obj->d[ ii ] == cmp_obj->d[ ii ] );
                    }
                }

                if( orig_obj->num_e > 0u )
                {
                    for( std::size_t ii = 0u ; ii < orig_obj->num_e ; ++ii )
                    {
                        ASSERT_TRUE( std::fabs( orig_obj->e[ ii ] - cmp_obj->e[ ii ] )
                            < std::numeric_limits< double >::epsilon() );
                    }
                }
            }
        }
    }
    else
    {
        std::cout << "Unable to perform unit-test as no Cuda "
                  << "platforms have been found" << std::endl;
    }

    ::st_Buffer_delete( orig_buffer );
    ::st_Buffer_delete( copy_buffer );
}

/* end: sixtracklib/tests/sixtracklib/cuda/test_buffer_cuda_c99.cpp */
