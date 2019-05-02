#include "sixtracklib/cuda/context.hpp"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>
#include <cstdlib>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/cuda/internal/context_base.hpp"
#include "sixtracklib/cuda/argument.hpp"

namespace SIXTRL_CXX_NAMESPACE
{
    CudaController::CudaController( char const* config_str ) :
        SIXTRL_CXX_NAMESPACE::CudaControllerBase( config_str )
    {
        using success_flag_arg_t = CudaArgument;
        using success_flag_t     = CudaController::success_flag_t;
        using ptr_stored_arg_t   = CudaController::ptr_stored_base_argument_t;

        ptr_stored_arg_t success_flag_arg( new success_flag_arg_t(
            sizeof( success_flag_t ), this ) );

        this->doUpdateStoredSuccessFlagArgument(
            std::move( success_flag_arg ) );

        this->doSetReadyForSendFlag( true );
        this->doSetReadyForReceiveFlag( true );
        this->doSetReadyForRemapFlag( true );
    }

    CudaController::success_flag_t
    CudaController::doGetSuccessFlagValueFromArg() const
    {
        using success_flag_t     = CudaController::success_flag_t;
        using success_flag_arg_t = CudaController::success_flag_arg_t;
        using arg_buffer_t       = CudaController::cuda_const_arg_buffer_t;

        success_flag_t success_flag = success_flag_t{ 0 };

        if( ( this->hasSuccessFlagArgument() ) &&
            ( this->type() == this->ptrSuccessFlagArgument()->type() ) &&
            ( this->ptrSuccessFlagArgument()->hasArgumentBuffer() ) &&
            ( this->ptrSuccessFlagArgument()->usesRawArgument() ) &&
            ( this->ptrSuccessFlagArgument()->size() >=
                sizeof( success_flag_t ) ) )
        {
            /* WARNING: !! This cast relies on consistency of type() info !! */
            success_flag_arg_t const* ptr_success_flag_arg =
                static_cast< success_flag_arg_t const* >(
                    this->ptrSuccessFlagArgument() );

            if( ( ptr_success_flag_arg != nullptr ) &&
                ( ptr_success_flag_arg->hasCudaArgBuffer() ) )
            {
                arg_buffer_t arg_buffer =
                    ptr_success_flag_arg->cudaArgBuffer();

                /* On the way here, we've already passed the initial
                 * hasArgumentBuffer() test and the more narrower
                 * hasCudaArgBuffer() test, thus it should be safe to assume
                 * that actually arg_buffer != nullptr. Let's verify that*/
                SIXTRL_ASSERT( arg_buffer != nullptr );

                std::memcpy( &success_flag, arg_buffer,
                             sizeof( success_flag_t ) );
            }
        }

        return success_flag;
    }

    void CudaController::doSetSuccessFlagValueFromArg(
        CudaController::success_flag_t const success_flag )
    {
        using success_flag_t     = CudaController::success_flag_t;
        using success_flag_arg_t = CudaController::success_flag_arg_t;
        using arg_buffer_t       = CudaController::cuda_arg_buffer_t;

        if( ( this->hasSuccessFlagArgument() ) &&
            ( this->type() == this->ptrSuccessFlagArgument()->type() ) &&
            ( this->ptrSuccessFlagArgument()->hasArgumentBuffer() ) &&
            ( this->ptrSuccessFlagArgument()->usesRawArgument() ) &&
            ( this->ptrSuccessFlagArgument()->size() >=
                sizeof( success_flag_t ) ) )
        {
            /* WARNING: !! This cast relies on consistency of type() info !! */
            success_flag_arg_t* ptr_success_flag_arg =
                static_cast< success_flag_arg_t* >(
                    this->ptrSuccessFlagArgument() );

            if( ( ptr_success_flag_arg != nullptr ) &&
                ( ptr_success_flag_arg->hasCudaArgBuffer() ) )
            {
                arg_buffer_t arg_buffer =
                    ptr_success_flag_arg->cudaArgBuffer();

                /* On the way here, we've already passed the initial
                 * hasArgumentBuffer() test and the more narrower
                 * hasCudaArgBuffer() test, thus it should be safe to assume
                 * that actually arg_buffer != nullptr. Let's verify that*/
                SIXTRL_ASSERT( arg_buffer != nullptr );

                std::memcpy( arg_buffer, &success_flag,
                             sizeof( success_flag_t ) );
            }
        }

        return;
    }

    CudaController::success_flag_arg_t*
    CudaController::doGetPtrToDerivedSuccessFlagArg()
    {
        return nullptr;
    }

    CudaController::success_flag_arg_t const*
    CudaController::doGetPtrToDerivedSuccessFlagArg() const
    {
        return nullptr;
    }
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/internal/controller.cpp */
