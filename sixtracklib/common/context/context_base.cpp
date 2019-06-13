#include "sixtracklib/common/context/context_base.hpp"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/namespace.h"
#include "sixtracklib/common/context/argument_base.hpp"
#include "sixtracklib/common/buffer/managed_buffer_remap.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/buffer.h"

namespace SIXTRL_CXX_NAMESPACE
{
    ContextBase::type_id_t ContextBase::type() const SIXTRL_NOEXCEPT
    {
        return this->m_type_id;
    }

    std::string const& ContextBase::typeStr() const SIXTRL_NOEXCEPT
    {
        return this->m_type_id_str;
    }

    char const* ContextBase::ptrTypeStr() const SIXTRL_NOEXCEPT
    {
        return this->m_type_id_str.c_str();
    }

    bool ContextBase::hasConfigStr() const SIXTRL_NOEXCEPT
    {
        return ( !this->m_config_str.empty() );
    }

    std::string const& ContextBase::configStr() const SIXTRL_NOEXCEPT
    {
        return this->m_config_str;
    }

    char const* ContextBase::ptrConfigStr() const SIXTRL_NOEXCEPT
    {
        return ( !this->m_config_str.empty() )
            ? this->m_config_str.c_str() : nullptr;
    }

    bool ContextBase::usesNodes() const SIXTRL_NOEXCEPT
    {
        return this->m_uses_nodes;
    }

    void ContextBase::clear()
    {
        this->doClear();
    }

    bool ContextBase::readyForSend() const SIXTRL_NOEXCEPT
    {
        return this->m_ready_for_send;
    }

    bool ContextBase::readyForReceive() const SIXTRL_NOEXCEPT
    {
        return this->m_ready_for_receive;
    }

    bool ContextBase::readyForRemap() const SIXTRL_NOEXCEPT
    {
        return this->m_ready_for_remap;
    }

    ContextBase::status_t ContextBase::send(
        ContextBase::ptr_arg_base_t SIXTRL_RESTRICT arg,
        const void *const SIXTRL_RESTRICT src_begin,
        ContextBase::size_type const src_size )
    {
        using size_t     = ContextBase::size_type;
        using status_t   = ContextBase::status_t;

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

    ContextBase::status_t ContextBase::send(
        ContextBase::ptr_arg_base_t SIXTRL_RESTRICT arg,
        const ContextBase::c_buffer_t *const SIXTRL_RESTRICT source )
    {
        using size_t     = ContextBase::size_type;
        using status_t   = ContextBase::status_t;
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

    ContextBase::status_t ContextBase::send(
        ContextBase::ptr_arg_base_t SIXTRL_RESTRICT arg,
        ContextBase::buffer_t const& SIXTRL_RESTRICT_REF source )
    {
        using size_t     = ContextBase::size_type;
        using status_t   = ContextBase::status_t;
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

    ContextBase::status_t ContextBase::receive(
        void* SIXTRL_RESTRICT dest_begin,
        ContextBase::size_type const dest_capacity,
        ContextBase::ptr_arg_base_t SIXTRL_RESTRICT src_arg )
    {
        using size_t     = ContextBase::size_type;
        using status_t   = ContextBase::status_t;

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

    ContextBase::status_t ContextBase::receive(
        ContextBase::c_buffer_t* SIXTRL_RESTRICT destination,
        ContextBase::ptr_arg_base_t SIXTRL_RESTRICT src_arg )
    {
        using size_t     = ContextBase::size_type;
        using status_t   = ContextBase::status_t;
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

    ContextBase::status_t ContextBase::receive(
        ContextBase::buffer_t& SIXTRL_RESTRICT_REF destination,
        ContextBase::ptr_arg_base_t SIXTRL_RESTRICT src_arg )
    {
        using size_t     = ContextBase::size_type;
        using status_t   = ContextBase::status_t;
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

    ContextBase::status_t ContextBase::remapSentCObjectsBuffer(
        ContextBase::ptr_arg_base_t SIXTRL_RESTRICT arg,
        ContextBase::size_type const arg_size )
    {
        ContextBase::status_t status = ContextBase::status_t{ -1 };

        if( ( arg != nullptr ) && ( this->readyForRemap() ) )
        {
            status = this->doRemapSentCObjectsBuffer( arg, arg_size );
        }

        return status;
    }

    bool ContextBase::hasSuccessFlagArgument() const SIXTRL_NOEXCEPT
    {
        return ( this->m_ptr_success_flag_arg.get() != nullptr );
    }

    ContextBase::ptr_arg_base_t
    ContextBase::ptrSuccessFlagArgument() SIXTRL_NOEXCEPT
    {
        using _this_t = ContextBase;
        using ptr_t   = ContextBase::ptr_arg_base_t;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).ptrSuccessFlagArgument() );
    }

    ContextBase::success_flag_t ContextBase::lastSuccessFlagValue() const
    {
        return this->doGetSuccessFlagValueFromArg();
    }

    ContextBase::ptr_const_arg_base_t
    ContextBase::ptrSuccessFlagArgument() const SIXTRL_NOEXCEPT
    {
        using flag_t = ContextBase::success_flag_t;

        return ( ( this->m_ptr_success_flag_arg.get() != nullptr ) &&
                 ( this->m_ptr_success_flag_arg->usesRawArgument() ) &&
                 ( this->m_ptr_success_flag_arg->size() == sizeof( flag_t ) ) )
            ? this->m_ptr_success_flag_arg.get() : nullptr;
    }

    bool ContextBase::isInDebugMode() const SIXTRL_NOEXCEPT
    {
        return this->m_debug_mode;
    }

    ContextBase::~ContextBase() SIXTRL_NOEXCEPT
    {

    }

    ContextBase::ContextBase(
        ContextBase::type_id_t const type_id,
        const char *const SIXTRL_RESTRICT type_id_str,
        const char *const SIXTRL_RESTRICT config_str ) :
            m_config_str(), m_type_id_str(), m_type_id( type_id ),
            m_ptr_success_flag_arg( nullptr ),
            m_uses_nodes( false ),
            m_ready_for_remap( false ),
            m_ready_for_send( false ),
            m_ready_for_receive( false ),
            m_debug_mode( false )
    {
        this->doSetTypeIdStr( type_id_str );
        this->doParseConfigStrBaseImpl( config_str );
    }

    void ContextBase::doClear()
    {
        this->m_ready_for_receive = false;
        this->m_ready_for_send    = false;
        this->m_ready_for_remap   = false;

        return;
    }

    void ContextBase::doParseConfigStr(
        const char *const SIXTRL_RESTRICT config_str )
    {
        this->doParseConfigStrBaseImpl( config_str );
    }

    ContextBase::status_t ContextBase::doSend(
        ContextBase::ptr_arg_base_t SIXTRL_RESTRICT,
        const void *const SIXTRL_RESTRICT, size_type const )
    {
        return ContextBase::status_t{ -1 };
    }

    ContextBase::status_t ContextBase::doReceive( void* SIXTRL_RESTRICT,
        ContextBase::size_type const,
        ContextBase::ptr_arg_base_t SIXTRL_RESTRICT )
    {
        return ContextBase::status_t{ -1 };
    }

    ContextBase::status_t ContextBase::doRemapSentCObjectsBuffer(
        ContextBase::ptr_arg_base_t SIXTRL_RESTRICT,
        ContextBase::size_type const )
    {
        return ContextBase::status_t{ -1 };
    }

    ContextBase::success_flag_t ContextBase::doGetSuccessFlagValueFromArg() const
    {
        using  success_flag_t = ContextBase::success_flag_t;
        return success_flag_t{ 0 };
    }

    void ContextBase::doSetSuccessFlagValueFromArg(
        ContextBase::success_flag_t const success_flag )
    {
        return;
    }

    void ContextBase::doSetTypeId(
        ContextBase::type_id_t const type_id ) SIXTRL_NOEXCEPT
    {
        this->m_type_id = type_id;
    }

    void ContextBase::doSetTypeIdStr(
        const char *const SIXTRL_RESTRICT type_id_str ) SIXTRL_NOEXCEPT
    {
        if( ( type_id_str != nullptr ) &&
            ( std::strlen( type_id_str ) > std::size_t{ 0 } ) )
        {
            this->m_type_id_str = std::string{ type_id_str };
        }
        else
        {
            this->m_type_id_str.clear();
        }
    }

    void ContextBase::doSetUsesNodesFlag( bool const flag ) SIXTRL_NOEXCEPT
    {
        this->m_uses_nodes = flag;
    }

    void ContextBase::doSetReadyForSendFlag( bool const flag ) SIXTRL_NOEXCEPT
    {
        this->m_ready_for_send = flag;
    }

    void ContextBase::doSetReadyForReceiveFlag(
        bool const flag ) SIXTRL_NOEXCEPT
    {
        this->m_ready_for_receive = flag;
    }

    void ContextBase::doSetReadyForRemapFlag( bool const flag ) SIXTRL_NOEXCEPT
    {
        this->m_ready_for_remap = flag;
    }

    void ContextBase::doSetDebugModeFlag( bool const flag ) SIXTRL_NOEXCEPT
    {
        this->m_debug_mode = flag;
    }

    void ContextBase::doUpdateStoredSuccessFlagArgument(
        ContextBase::ptr_stored_base_argument_t&&
            ptr_stored_arg ) SIXTRL_NOEXCEPT
    {
        this->m_ptr_success_flag_arg = std::move( ptr_stored_arg );
    }

    void ContextBase::doParseConfigStrBaseImpl(
            const char *const SIXTRL_RESTRICT config_str )
    {
        if( ( config_str != nullptr ) &&
            ( std::strlen( config_str ) > std::size_t{ 0 } ) )
        {
            this->m_config_str = std::string{ config_str };
        }
        else if( !this->m_config_str.empty() )
        {
            this->m_config_str.clear();
        }

        return;
    }
}

#endif /* C++, host */

/* end: sixtracklib/common/context/context_base.cpp */
