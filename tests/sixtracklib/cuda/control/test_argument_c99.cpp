#include "sixtracklib/cuda/argument.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/common/definitions.h"
#include "sixtracklib/cuda/controller.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"

TEST( CXX_CudaArgumentTests, BasicUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using cuda_arg_t    = ::NS(CudaArgument);
    using size_t        = ::NS(ctrl_size_t);
    using status_t      = ::NS(ctrl_status_t);
    using cuda_ctrl_t   = ::NS(CudaController);
    using c_buffer_t    = ::NS(Buffer);
    using generic_obj_t = ::NS(GenericObj);
    using type_id_t     = ::NS(object_type_id_t);
    
    /* --------------------------------------------------------------------- */
    /* Prepare the cobject buffer's (both C++ and C99) */
    
    size_t constexpr num_d_values = size_t{ 10 };
    size_t constexpr num_e_values = size_t{ 10 };
    size_t constexpr num_obj      = size_t{ 10 };
    
    c_buffer_t* c_obj_buffer = ::NS(Buffer_new)( size_t{ 0 } );
        
    for( size_t ii = size_t{ 0 } ; ii < num_obj ; ++ii )
    {
        type_id_t const type_id = static_cast< type_id_t >( ii );
        
        generic_obj_t* ptr_c_obj = ::NS(GenericObj_new)( 
            c_obj_buffer, type_id, num_d_values, num_e_values );
        
        SIXTRL_ASSERT( ptr_c_obj != nullptr );
    }
    
    size_t const slot_size = ::NS(Buffer_get_slot_size)( c_obj_buffer );
    SIXTRL_ASSERT( slot_size > size_t{ 0 } );
    
    cuda_ctrl_t* cuda_controller = ::NS(CudaController_create)();
    
    cuda_arg_t* arg1 = ::NS(CudaArgument_new)( cuda_controller);
    ASSERT_TRUE( arg1 != nullptr );
    
    /* --------------------------------------------------------------------- */
    
    ::NS(Buffer_delete)( c_obj_buffer );
    c_obj_buffer = nullptr;
    
    ::NS(Controller_delete)( cuda_controller );
    cuda_controller = nullptr;
    
    ::NS(Argument_delete)( arg1 );
    arg1 = nullptr;
}

/* end: tests/sixtracklib/cuda/control/test_argument_cxx.cpp */

