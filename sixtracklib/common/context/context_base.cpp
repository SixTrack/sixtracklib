#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/context/context_base.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdint>
    #include <cstdlib>
    #include <limits>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/generated/namespace.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

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

    void ContextBase::doClear()
    {
        return;
    }

    ContextBase::ContextBase( const char *const SIXTRL_RESTRICT type_str,
        ContextBase::type_id_t const type_id ) :
            m_config_str(), m_type_id_str(), m_type_id( type_id ),
            m_uses_nodes( false )
    {
        if( type_str != nullptr )
        {
            this->m_type_id_str = std::string( type_str );
        }
    }

    void ContextBase::doParseConfigStr(
        const char *const SIXTRL_RESTRICT config_str )
    {
        if( config_str != nullptr )
        {
            this->m_config_str = std::string( config_str );
        }
    }

    void ContextBase::doSetUsesNodesFlag( bool const flag ) SIXTRL_NOEXCEPT
    {
        this->m_uses_nodes = flag;
    }
}

/* end: sixtracklib/common/context/context_base.cpp */
