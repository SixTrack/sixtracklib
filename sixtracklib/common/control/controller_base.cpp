#include "sixtracklib/common/control/controller_base.hpp"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/namespace.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/arch_info.hpp"
#include "sixtracklib/common/control/arch_base.hpp"
#include "sixtracklib/common/control/argument_base.hpp"
#include "sixtracklib/common/buffer/managed_buffer_remap.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/buffer.h"

namespace SIXTRL_CXX_NAMESPACE
{
    bool ControllerBase::usesNodes() const SIXTRL_NOEXCEPT
    {
        return this->m_uses_nodes;
    }

    void ControllerBase::clear()
    {
        this->doClear();
    }

    bool ControllerBase::readyForSend() const SIXTRL_NOEXCEPT
    {
        return this->m_ready_for_send;
    }

    bool ControllerBase::readyForReceive() const SIXTRL_NOEXCEPT
    {
        return this->m_ready_for_receive;
    }

    bool ControllerBase::readyForRemap() const SIXTRL_NOEXCEPT
    {
        return this->m_ready_for_remap;
    }

    ControllerBase::status_t ControllerBase::send(
        ControllerBase::ptr_arg_base_t SIXTRL_RESTRICT arg,
        const void *const SIXTRL_RESTRICT src_begin,
        ControllerBase::size_type const src_size )
    {
        using size_t     = ControllerBase::size_type;
        using status_t   = ControllerBase::status_t;

        status_t status = status_t{ -1 };

        if( ( arg != nullptr ) &&
            ( this->readyForSend() ) && ( this->readyForRemap() ) &&
            ( arg->usesRawArgument() ) &&
            ( arg->capacity() > size_t{ 0 } ) && ( src_begin != nullptr ) &&
            ( src_size > size_t{ 0 } ) && ( src_size <= arg->capacity() ) )
        {
            status = this->doSend( arg, src_begin, src_size );
        }

        return status;
    }

    ControllerBase::status_t ControllerBase::send(
        ControllerBase::ptr_arg_base_t SIXTRL_RESTRICT arg,
        const ControllerBase::c_buffer_t *const SIXTRL_RESTRICT source )
    {
        using size_t     = ControllerBase::size_type;
        using status_t   = ControllerBase::status_t;
        using data_ptr_t = void const*;

        status_t status = status_t{ -1 };

        data_ptr_t   src_begin = ::NS(Buffer_get_const_data_begin)( source );
        size_t const src_size  = ::NS(Buffer_get_size)( source );

        if( ( arg != nullptr ) &&
            ( this->readyForSend() ) && ( this->readyForRemap() ) &&
            ( arg->usesCObjectsBuffer() ) &&
            ( arg->capacity() > size_t{ 0 } ) && ( src_begin != nullptr ) &&
            ( src_size > size_t{ 0 } ) && ( src_size <= arg->capacity() ) )
        {
            status = this->doSend( arg, src_begin, src_size );

            if( status == status_t{ 0 } )
            {
                status = this->doRemapSentCObjectsBuffer( arg, src_size );
            }
        }

        return status;
    }

    ControllerBase::status_t ControllerBase::send(
        ControllerBase::ptr_arg_base_t SIXTRL_RESTRICT arg,
        ControllerBase::buffer_t const& SIXTRL_RESTRICT_REF source )
    {
        using size_t     = ControllerBase::size_type;
        using status_t   = ControllerBase::status_t;
        using data_ptr_t = void const*;

        status_t status = status_t{ -1 };

        data_ptr_t   src_begin = source.dataBegin< data_ptr_t >();
        size_t const src_size  = source.size();

        if( ( arg != nullptr ) &&
            ( this->readyForSend() ) && ( this->readyForRemap() ) &&
            ( arg->usesCObjectsBuffer() ) &&
            ( arg->capacity() > size_t{ 0 } ) && ( src_begin != nullptr ) &&
            ( src_size > size_t{ 0 } ) && ( src_size <= arg->capacity() ) )
        {
            status = this->doSend( arg, src_begin, src_size );

            if( status == status_t{ 0 } )
            {
                status = this->doRemapSentCObjectsBuffer( arg, src_size );
            }
        }

        return status;
    }

    ControllerBase::status_t ControllerBase::receive(
        void* SIXTRL_RESTRICT dest_begin,
        ControllerBase::size_type const dest_capacity,
        ControllerBase::ptr_arg_base_t SIXTRL_RESTRICT src_arg )
    {
        using size_t     = ControllerBase::size_type;
        using status_t   = ControllerBase::status_t;

        status_t status = status_t{ -1 };

        if( ( src_arg != nullptr ) && ( this->readyForReceive() ) &&
            ( src_arg->size() > size_t{ 0 } ) &&
            ( dest_capacity >= src_arg->size() ) && ( dest_begin != nullptr ) )
        {
            status = this->doReceive( dest_begin, dest_capacity, src_arg );

            if( ( status == status_t{ 0 } ) &&
                ( src_arg->usesCObjectsBuffer() ) )
            {
                unsigned char* managed_buffer_begin =
                    reinterpret_cast< unsigned char* >( dest_begin );

                size_t const slot_size = SIXTRL_BUFFER_DEFAULT_SLOT_SIZE;
                if( ::NS(ManagedBuffer_remap)( managed_buffer_begin,
                        slot_size ) != 0 )
                {
                    status = status_t{ -1 };
                }
            }
        }

        return status;

    }

    ControllerBase::status_t ControllerBase::receive(
        ControllerBase::c_buffer_t* SIXTRL_RESTRICT destination,
        ControllerBase::ptr_arg_base_t SIXTRL_RESTRICT src_arg )
    {
        using size_t     = ControllerBase::size_type;
        using status_t   = ControllerBase::status_t;
        using data_ptr_t = void*;

        status_t status = status_t{ -1 };
        data_ptr_t dest_begin = ::NS(Buffer_get_data_begin)( destination );
        size_t const dest_capacity = ::NS(Buffer_get_capacity)( destination );

        if( ( src_arg != nullptr ) && ( this->readyForReceive() ) &&
            ( src_arg->usesCObjectsBuffer() ) &&
            ( src_arg->size() > size_t{ 0 } ) &&
            ( dest_capacity >= src_arg->size() ) && ( dest_begin != nullptr ) )
        {
            status = this->doReceive( dest_begin, dest_capacity, src_arg );

            if( status == status_t{ 0 } )
            {
                status = ::NS(Buffer_remap)( destination );
            }
        }

        return status;
    }

    ControllerBase::status_t ControllerBase::receive(
        ControllerBase::buffer_t& SIXTRL_RESTRICT_REF destination,
        ControllerBase::ptr_arg_base_t SIXTRL_RESTRICT src_arg )
    {
        using size_t     = ControllerBase::size_type;
        using status_t   = ControllerBase::status_t;
        using data_ptr_t = void*;

        status_t status = status_t{ -1 };

        if( ( src_arg != nullptr ) && ( this->readyForReceive() ) &&
            ( src_arg->usesCObjectsBuffer() ) &&
            ( src_arg->size() > size_t{ 0 } ) &&
            ( destination.capacity() >= src_arg->size() ) &&
            ( destination.dataBegin< data_ptr_t >() != nullptr ) )
        {
            status = this->doReceive( destination.dataBegin< data_ptr_t >(),
                                      destination.capacity(), src_arg );

            if( status == status_t{ 0 } )
            {
                status = ( destination.remap() )
                    ? status_t{ 0 } : status_t{ -1 };
            }
        }

        return status;
    }

    ControllerBase::status_t ControllerBase::remapSentCObjectsBuffer(
        ControllerBase::ptr_arg_base_t SIXTRL_RESTRICT arg,
        ControllerBase::size_type const arg_size )
    {
        ControllerBase::status_t status = ControllerBase::status_t{ -1 };

        if( ( arg != nullptr ) && ( this->readyForRemap() ) )
        {
            status = this->doRemapSentCObjectsBuffer( arg, arg_size );
        }

        return status;
    }

    bool ControllerBase::hasSuccessFlagArgument() const SIXTRL_NOEXCEPT
    {
        return ( this->m_ptr_success_flag_arg.get() != nullptr );
    }

    ControllerBase::ptr_arg_base_t
    ControllerBase::ptrSuccessFlagArgument() SIXTRL_NOEXCEPT
    {
        using _this_t = ControllerBase;
        using ptr_t   = ControllerBase::ptr_arg_base_t;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).ptrSuccessFlagArgument() );
    }

    ControllerBase::success_flag_t ControllerBase::lastSuccessFlagValue() const
    {
        return this->doGetSuccessFlagValueFromArg();
    }

    ControllerBase::ptr_const_arg_base_t
    ControllerBase::ptrSuccessFlagArgument() const SIXTRL_NOEXCEPT
    {
        using flag_t = ControllerBase::success_flag_t;

        return ( ( this->m_ptr_success_flag_arg.get() != nullptr ) &&
                 ( this->m_ptr_success_flag_arg->usesRawArgument() ) &&
                 ( this->m_ptr_success_flag_arg->size() == sizeof( flag_t ) ) )
            ? this->m_ptr_success_flag_arg.get() : nullptr;
    }

    bool ControllerBase::isInDebugMode() const SIXTRL_NOEXCEPT
    {
        return this->m_debug_mode;
    }

    ControllerBase::~ControllerBase() SIXTRL_NOEXCEPT
    {

    }

    ControllerBase::ControllerBase(
        ControllerBase::arch_id_t const arch_id,
        const char *const SIXTRL_RESTRICT arch_str,
        const char *const SIXTRL_RESTRICT config_str ) :
        ArchBase( arch_id, arch_str, config_str ),
            m_ptr_success_flag_arg( nullptr ),
            m_uses_nodes( false ),
            m_ready_for_remap( false ),
            m_ready_for_send( false ),
            m_ready_for_receive( false ),
            m_debug_mode( false )
    {

    }

    void ControllerBase::doClear()
    {
        this->m_ready_for_receive = false;
        this->m_ready_for_send    = false;
        this->m_ready_for_remap   = false;

        return;
    }

    ControllerBase::status_t ControllerBase::doSend(
        ControllerBase::ptr_arg_base_t SIXTRL_RESTRICT,
        const void *const SIXTRL_RESTRICT, size_type const )
    {
        return ControllerBase::status_t{ -1 };
    }

    ControllerBase::status_t ControllerBase::doReceive( void* SIXTRL_RESTRICT,
        ControllerBase::size_type const,
        ControllerBase::ptr_arg_base_t SIXTRL_RESTRICT )
    {
        return ControllerBase::status_t{ -1 };
    }

    ControllerBase::status_t ControllerBase::doRemapSentCObjectsBuffer(
        ControllerBase::ptr_arg_base_t SIXTRL_RESTRICT,
        ControllerBase::size_type const )
    {
        return ControllerBase::status_t{ -1 };
    }

    ControllerBase::success_flag_t
    ControllerBase::doGetSuccessFlagValueFromArg() const
    {
        using  success_flag_t = ControllerBase::success_flag_t;
        return success_flag_t{ 0 };
    }

    void ControllerBase::doSetSuccessFlagValueFromArg(
        ControllerBase::success_flag_t const success_flag )
    {
        return;
    }

    void ControllerBase::doSetUsesNodesFlag( bool const flag ) SIXTRL_NOEXCEPT
    {
        this->m_uses_nodes = flag;
    }

    void ControllerBase::doSetReadyForSendFlag( bool const flag ) SIXTRL_NOEXCEPT
    {
        this->m_ready_for_send = flag;
    }

    void ControllerBase::doSetReadyForReceiveFlag(
        bool const flag ) SIXTRL_NOEXCEPT
    {
        this->m_ready_for_receive = flag;
    }

    void ControllerBase::doSetReadyForRemapFlag( bool const flag ) SIXTRL_NOEXCEPT
    {
        this->m_ready_for_remap = flag;
    }

    void ControllerBase::doSetDebugModeFlag( bool const flag ) SIXTRL_NOEXCEPT
    {
        this->m_debug_mode = flag;
    }

    void ControllerBase::doUpdateStoredSuccessFlagArgument(
        ControllerBase::ptr_stored_base_argument_t&&
            ptr_stored_arg ) SIXTRL_NOEXCEPT
    {
        this->m_ptr_success_flag_arg = std::move( ptr_stored_arg );
    }

}

#endif /* C++, host */

/* end: sixtracklib/common/control/controller_base.cpp */
