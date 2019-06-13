#ifndef SIXTRACKL_COMMON_PARTICLES_PARTICLES_ADDR_CXX_HEADER_HPP_
#define SIXTRACKL_COMMON_PARTICLES_PARTICLES_ADDR_CXX_HEADER_HPP_

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/particles/particles_addr.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdlib>
    #include <utility>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.hpp"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    struct ParticlesAddr : public ::NS(ParticlesAddr)
    {
        using c_api_t        = ::NS(ParticlesAddr);
        using buffer_t       = SIXTRL_CXX_NAMESPACE::Buffer;
        using c_buffer_t     = buffer_t::c_api_t;
        using address_t      = buffer_t::address_t;
        using size_type      = buffer_t::size_type;
        using type_id_t      = buffer_t::type_id_t;
        using num_particles_t = ::NS(particle_num_elements_t);

        SIXTRL_FN ParticlesAddr() = default;
        SIXTRL_FN ParticlesAddr( ParticlesAddr const& other ) = default;
        SIXTRL_FN ParticlesAddr( ParticlesAddr&& other ) = default;

        SIXTRL_FN ParticlesAddr& operator=( ParticlesAddr const& rhs ) = default;
        SIXTRL_FN ParticlesAddr& operator=( ParticlesAddr&& rhs ) = default;

        SIXTRL_FN ~ParticlesAddr() = default;

        SIXTRL_STATIC SIXTRL_FN SIXTRL_PARTICLE_ARGPTR_DEC
        ParticlesAddr const* FromBuffer(
            Buffer const& SIXTRL_RESTRICT_REF buffer,
            size_type const index ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN SIXTRL_PARTICLE_ARGPTR_DEC
        ParticlesAddr const* FromBuffer(
            const ::NS(Buffer) *const SIXTRL_RESTRICT buffer,
            size_type const index ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN SIXTRL_PARTICLE_ARGPTR_DEC
        ParticlesAddr* FromBuffer( Buffer& SIXTRL_RESTRICT_REF buffer,
            size_type const index ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN SIXTRL_PARTICLE_ARGPTR_DEC
        ParticlesAddr* FromBuffer( ::NS(Buffer)* SIXTRL_RESTRICT buffer,
            size_type const index ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr const*
        FromBufferObject( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object)
            *const SIXTRL_RESTRICT info ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr*
        FromBufferObject( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
                SIXTRL_RESTRICT info ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_STATIC SIXTRL_FN bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            num_particles_t const num_particles,
            SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer,
            num_particles_t const num_particles =
                num_particles_t{ 0 } );

        SIXTRL_STATIC SIXTRL_FN SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr*
        AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            num_particles_t const num_particles,
            address_t const q0_addr,    address_t  const mass0_addr,
            address_t const beta0_addr, address_t  const gamma0_addr,
            address_t const p0c_addr,   address_t  const s_addr,
            address_t const x_addr,     address_t  const y_addr,
            address_t const px_addr,    address_t  const py_addr,
            address_t const zeta_addr,  address_t  const psigma_addr,
            address_t const delta_addr, address_t  const rpp_addr,
            address_t const rvv_addr,   address_t  const chi_addr,
            address_t const charge_ratio_addr,
            address_t const particle_id_addr,
            address_t const at_element_id_addr,
            address_t const at_turn_addr, address_t const state_addr );

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId()      const SIXTRL_NOEXCEPT;

        SIXTRL_FN c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN c_api_t* getCApiPtr() SIXTRL_NOEXCEPT;

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN num_particles_t getNumOfParticles() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getQ0Addr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getMass0Addr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getBeta0Addr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getGamma0Addr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getP0cAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getSAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getXAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getYAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getPxAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getPyAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getZetaAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getPsigmaAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getDeltaAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getRppAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getRvvAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getChiAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getChargeRatioAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getParticleIdAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getAtElementIdAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getAtTurnAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getStateAddr() const SIXTRL_NOEXCEPT;

        SIXTRL_FN void setNumOfParticles(
            num_particles_t const num_elements ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setQ0Addr( address_t const addr ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setMass0Addr( address_t const addr ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setBeta0Addr( address_t const addr ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setGamma0Addr( address_t const addr ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setP0cAddr( address_t const addr ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setSAddr( address_t const addr ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setXAddr( address_t const addr ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setYAddr( address_t const addr ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setPxAddr( address_t const addr ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setPyAddr( address_t const addr ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setZetaAddr( address_t const addr ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setPsigmaAddr( address_t const addr ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setDeltaAddr( address_t const addr ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setRppAddr( address_t const addr ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setRvvAddr( address_t const addr ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setChiAddr( address_t const addr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setChargeRatioAddr(
            address_t const addr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setParticleIdAddr(
            address_t const addr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setAtElementIdAddr(
            address_t const addr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setAtTurnAddr( address_t const addr ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setStateAddr(address_t const addr ) SIXTRL_NOEXCEPT;
    };

    template<> struct ObjectTypeTraits< ::NS(ParticlesAddr) >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return ::NS(OBJECT_TYPE_PARTICLES_ADDR);
        }
    };

    template<> struct ObjectTypeTraits< SIXTRL_CXX_NAMESPACE::ParticlesAddr >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return ::NS(OBJECT_TYPE_PARTICLES_ADDR);
        }
    };

    SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* ParticlesAddr_new(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        ParticlesAddr::num_particles_t const num_particles );

    SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* ParticlesAddr_new(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT buffer,
        ParticlesAddr::num_particles_t const num_particles );

    SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* ParticlesAddr_add(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        ParticlesAddr::num_particles_t const num_particles,
        ParticlesAddr::address_t const q0_addr,
        ParticlesAddr::address_t const mass0_addr,
        ParticlesAddr::address_t const beta0_addr,
        ParticlesAddr::address_t const gamma0_addr,
        ParticlesAddr::address_t const p0c_addr,
        ParticlesAddr::address_t const s_addr,
        ParticlesAddr::address_t const x_addr,
        ParticlesAddr::address_t const y_addr,
        ParticlesAddr::address_t const px_addr,
        ParticlesAddr::address_t const py_addr,
        ParticlesAddr::address_t const zeta_addr,
        ParticlesAddr::address_t const psigma_addr,
        ParticlesAddr::address_t const delta_addr,
        ParticlesAddr::address_t const rpp_addr,
        ParticlesAddr::address_t const rvv_addr,
        ParticlesAddr::address_t const chi_addr,
        ParticlesAddr::address_t const charge_ratio_addr,
        ParticlesAddr::address_t const particle_id_addr,
        ParticlesAddr::address_t const at_element_id_addr,
        ParticlesAddr::address_t const at_turn_addr,
        ParticlesAddr::address_t const state_addr  );

    SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* ParticlesAddr_add(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT buffer,
        ParticlesAddr::num_particles_t const num_particles,
        ParticlesAddr::address_t const q0_addr,
        ParticlesAddr::address_t const mass0_addr,
        ParticlesAddr::address_t const beta0_addr,
        ParticlesAddr::address_t const gamma0_addr,
        ParticlesAddr::address_t const p0c_addr,
        ParticlesAddr::address_t const s_addr,
        ParticlesAddr::address_t const x_addr,
        ParticlesAddr::address_t const y_addr,
        ParticlesAddr::address_t const px_addr,
        ParticlesAddr::address_t const py_addr,
        ParticlesAddr::address_t const zeta_addr,
        ParticlesAddr::address_t const psigma_addr,
        ParticlesAddr::address_t const delta_addr,
        ParticlesAddr::address_t const rpp_addr,
        ParticlesAddr::address_t const rvv_addr,
        ParticlesAddr::address_t const chi_addr,
        ParticlesAddr::address_t const charge_ratio_addr,
        ParticlesAddr::address_t const particle_id_addr,
        ParticlesAddr::address_t const at_element_id_addr,
        ParticlesAddr::address_t const at_turn_addr,
        ParticlesAddr::address_t const state_addr  );

    SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* ParticlesAddr_add_copy(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        ParticlesAddr const& SIXTRL_RESTRICT_REF orig );

    SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* ParticlesAddr_add_copy(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT buffer,
        ParticlesAddr const& SIXTRL_RESTRICT_REF orig );
}

#endif /* C++, Host */


/* ************************************************************************* *
 * *** Implementation of inline and template member functions          ***** *
 * ************************************************************************* */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
     SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr const*
    ParticlesAddr::FromBuffer( Buffer const& SIXTRL_RESTRICT_REF buffer,
        ParticlesAddr::size_type const index ) SIXTRL_NOEXCEPT
    {
        return ParticlesAddr::FromBufferObject( buffer[ index ] );
    }

    SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr const*
    ParticlesAddr::FromBuffer(
        const ::NS(Buffer) *const SIXTRL_RESTRICT buffer,
        ParticlesAddr::size_type const index ) SIXTRL_NOEXCEPT
    {
        return ParticlesAddr::FromBufferObject(
            ::NS(Buffer_get_const_object)( buffer, index ) );
    }

    SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr*
    ParticlesAddr::FromBuffer( Buffer& SIXTRL_RESTRICT_REF buffer,
        ParticlesAddr::size_type const index ) SIXTRL_NOEXCEPT
    {
        return ParticlesAddr::FromBufferObject( buffer[ index ] );
    }

    SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr*
    ParticlesAddr::FromBuffer( ::NS(Buffer)* SIXTRL_RESTRICT buffer,
        ParticlesAddr::size_type const index ) SIXTRL_NOEXCEPT
    {
        return ParticlesAddr::FromBufferObject(
            ::NS(Buffer_get_object)( buffer, index ) );
    }

    SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr const*
    ParticlesAddr::FromBufferObject( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
        NS(Object) *const SIXTRL_RESTRICT be_info ) SIXTRL_NOEXCEPT
    {
        using belement_t = ParticlesAddr;
        using ptr_beam_elem_t = SIXTRL_BUFFER_OBJ_ARGPTR_DEC belement_t const*;

        if( ( be_info != nullptr ) &&
            ( ::NS(Object_get_type_id)( be_info ) ==
              ::NS(OBJECT_TYPE_BEAM_MONITOR) ) &&
            ( ::NS(Object_get_size)( be_info ) >= sizeof( ParticlesAddr ) ) )
        {
            return reinterpret_cast< ptr_beam_elem_t >(
                static_cast< uintptr_t >( ::NS(Object_get_begin_addr)(
                    be_info ) ) );
        }

        return nullptr;
    }

    SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr*
    ParticlesAddr::FromBufferObject( SIXTRL_BUFFER_OBJ_ARGPTR_DEC ::NS(Object)*
        SIXTRL_RESTRICT be_info ) SIXTRL_NOEXCEPT
    {
        using  _this_t        = ParticlesAddr;
        using beam_element_t  = _this_t;
        using ptr_beam_elem_t = SIXTRL_BUFFER_OBJ_DATAPTR_DEC beam_element_t*;

        using object_t        = ::NS(Object);
        using ptr_const_obj_t = SIXTRL_BUFFER_OBJ_ARGPTR_DEC object_t const*;

        ptr_const_obj_t const_be_info = be_info;

        return const_cast< ptr_beam_elem_t >(
            _this_t::FromBufferObject( const_be_info ) );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE bool ParticlesAddr::CanAddToBuffer(
        ParticlesAddr::buffer_t& SIXTRL_RESTRICT_REF buffer,
        ParticlesAddr::num_particles_t const num_particles,
        SIXTRL_BUFFER_ARGPTR_DEC ParticlesAddr::size_type*
            SIXTRL_RESTRICT ptr_requ_objects,
        SIXTRL_BUFFER_ARGPTR_DEC ParticlesAddr::size_type*
            SIXTRL_RESTRICT ptr_requ_slots,
        SIXTRL_BUFFER_ARGPTR_DEC ParticlesAddr::size_type*
            SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT
    {
        return ::NS(ParticlesAddr_can_be_added)( buffer.getCApiPtr(),
            num_particles, ptr_requ_objects, ptr_requ_slots, ptr_requ_dataptrs );
    }

    SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr*
    ParticlesAddr::CreateNewOnBuffer(
        ParticlesAddr::buffer_t& SIXTRL_RESTRICT_REF buffer,
        ParticlesAddr::num_particles_t const num_particles )
    {
        using ptr_t = SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr*;
        return static_cast< ptr_t >( ::NS(ParticlesAddr_new)(
            buffer.getCApiPtr(), num_particles ) );
    }

    SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr*
    ParticlesAddr::AddToBuffer(
        ParticlesAddr::buffer_t& SIXTRL_RESTRICT_REF buffer,
        ParticlesAddr::num_particles_t const num_particles,
        ParticlesAddr::address_t const q0_addr,
        ParticlesAddr::address_t const mass0_addr,
        ParticlesAddr::address_t const beta0_addr,
        ParticlesAddr::address_t const gamma0_addr,
        ParticlesAddr::address_t const p0c_addr,
        ParticlesAddr::address_t const s_addr,
        ParticlesAddr::address_t const x_addr,
        ParticlesAddr::address_t const y_addr,
        ParticlesAddr::address_t const px_addr,
        ParticlesAddr::address_t const py_addr,
        ParticlesAddr::address_t const zeta_addr,
        ParticlesAddr::address_t const psigma_addr,
        ParticlesAddr::address_t const delta_addr,
        ParticlesAddr::address_t const rpp_addr,
        ParticlesAddr::address_t const rvv_addr,
        ParticlesAddr::address_t const chi_addr,
        ParticlesAddr::address_t const charge_ratio_addr,
        ParticlesAddr::address_t const particle_id_addr,
        ParticlesAddr::address_t const at_element_id_addr,
        ParticlesAddr::address_t const at_turn_addr,
        ParticlesAddr::address_t const state_addr  )
    {
        using ptr_t = SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr*;

        return static_cast< ptr_t >( ::NS(ParticlesAddr_add)(
            buffer.getCApiPtr(), num_particles, q0_addr, mass0_addr,
            beta0_addr, gamma0_addr, p0c_addr, s_addr, x_addr, y_addr,
            px_addr, py_addr, zeta_addr, psigma_addr, delta_addr, rpp_addr,
            rvv_addr, chi_addr, charge_ratio_addr, particle_id_addr,
            at_element_id_addr, at_turn_addr, state_addr ) );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE ParticlesAddr::type_id_t
    ParticlesAddr::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_PARTICLES_ADDR);
    }

    SIXTRL_INLINE ParticlesAddr::c_api_t const*
    ParticlesAddr::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        using c_api_t = ParticlesAddr::c_api_t;
        return reinterpret_cast< SIXTRL_PARTICLE_ARGPTR_DEC c_api_t const* >( this );
    }

    SIXTRL_INLINE ParticlesAddr::c_api_t*
    ParticlesAddr::getCApiPtr() SIXTRL_NOEXCEPT
    {
        using c_api_t = ParticlesAddr::c_api_t;
        return const_cast< SIXTRL_PARTICLE_ARGPTR_DEC c_api_t* >(
            static_cast< ParticlesAddr const& >( *this ).getCApiPtr() );
    }

    SIXTRL_INLINE void ParticlesAddr::preset() SIXTRL_NOEXCEPT
    {
        ::NS(ParticlesAddr_preset)( this->getCApiPtr() );
        return;
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE ParticlesAddr::num_particles_t
    ParticlesAddr::getNumOfParticles() const SIXTRL_NOEXCEPT
    { return this->num_particles; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getQ0Addr() const SIXTRL_NOEXCEPT
    { return this->q0_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getMass0Addr() const SIXTRL_NOEXCEPT
    { return this->mass0_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getBeta0Addr() const SIXTRL_NOEXCEPT
    { return this->beta0_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getGamma0Addr() const SIXTRL_NOEXCEPT
    { return this->gamma0_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getP0cAddr() const SIXTRL_NOEXCEPT
    { return this->p0c_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getSAddr() const SIXTRL_NOEXCEPT
    { return this->s_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getXAddr() const SIXTRL_NOEXCEPT
    { return this->x_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getYAddr() const SIXTRL_NOEXCEPT
    { return this->y_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getPxAddr() const SIXTRL_NOEXCEPT
    { return this->px_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getPyAddr() const SIXTRL_NOEXCEPT
    { return this->py_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getZetaAddr() const SIXTRL_NOEXCEPT
    { return this->zeta_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getPsigmaAddr() const SIXTRL_NOEXCEPT
    { return this->psigma_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getDeltaAddr() const SIXTRL_NOEXCEPT
    { return this->delta_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getRppAddr() const SIXTRL_NOEXCEPT
    { return this->rpp_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getRvvAddr() const SIXTRL_NOEXCEPT
    { return this->rvv_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getChiAddr() const SIXTRL_NOEXCEPT
    { return this->chi_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getChargeRatioAddr() const SIXTRL_NOEXCEPT
    { return this->charge_ratio_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getParticleIdAddr() const SIXTRL_NOEXCEPT
    { return this->particle_id_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getAtElementIdAddr() const SIXTRL_NOEXCEPT
    { return this->at_element_id_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getAtTurnAddr() const SIXTRL_NOEXCEPT
    { return this->at_turn_addr; }

    SIXTRL_INLINE ParticlesAddr::address_t
    ParticlesAddr::getStateAddr() const SIXTRL_NOEXCEPT
    { return this->state_addr; }

    SIXTRL_INLINE void ParticlesAddr::setNumOfParticles(
        ParticlesAddr::num_particles_t const num_elements ) SIXTRL_NOEXCEPT
    { this->num_particles = num_elements; }

    SIXTRL_INLINE void ParticlesAddr::setQ0Addr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->q0_addr = addr; }

    SIXTRL_INLINE void ParticlesAddr::setMass0Addr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->mass0_addr  = addr; }

    SIXTRL_INLINE void ParticlesAddr::setBeta0Addr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->beta0_addr  = addr; }

    SIXTRL_INLINE void ParticlesAddr::setGamma0Addr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->gamma0_addr  = addr; }

    SIXTRL_INLINE void ParticlesAddr::setP0cAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->p0c_addr  = addr; }

    SIXTRL_INLINE void ParticlesAddr::setSAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->s_addr = addr; }

    SIXTRL_INLINE void ParticlesAddr::setXAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->x_addr = addr; }

    SIXTRL_INLINE void ParticlesAddr::setYAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->y_addr = addr; }

    SIXTRL_INLINE void ParticlesAddr::setPxAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->px_addr = addr; }

    SIXTRL_INLINE void ParticlesAddr::setPyAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->py_addr = addr; }

    SIXTRL_INLINE void ParticlesAddr::setZetaAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->zeta_addr = addr; }

    SIXTRL_INLINE void ParticlesAddr::setPsigmaAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->psigma_addr = addr; }

    SIXTRL_INLINE void ParticlesAddr::setDeltaAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->delta_addr = addr; }

    SIXTRL_INLINE void ParticlesAddr::setRppAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->rpp_addr = addr; }

    SIXTRL_INLINE void ParticlesAddr::setRvvAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->rvv_addr = addr; }

    SIXTRL_INLINE void ParticlesAddr::setChiAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->chi_addr = addr; }

    SIXTRL_INLINE void ParticlesAddr::setChargeRatioAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->charge_ratio_addr = addr; }

    SIXTRL_INLINE void ParticlesAddr::setParticleIdAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->particle_id_addr = addr; }

    SIXTRL_INLINE void ParticlesAddr::setAtElementIdAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->at_element_id_addr = addr; }

    SIXTRL_INLINE void ParticlesAddr::setAtTurnAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->at_turn_addr = addr; }

    SIXTRL_INLINE void ParticlesAddr::setStateAddr(
        ParticlesAddr::address_t const addr ) SIXTRL_NOEXCEPT
    { this->state_addr = addr; }

    /* ===================================================================== */

    SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* ParticlesAddr_new(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        ParticlesAddr::num_particles_t const num_particles )
    {
        return static_cast< SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* >(
            ::NS(ParticlesAddr_new)( buffer.getCApiPtr(), num_particles ) );
    }

    SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* ParticlesAddr_new(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT buffer,
        ParticlesAddr::num_particles_t const num_particles )
    {
        return static_cast< SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* >(
            ::NS(ParticlesAddr_new)( buffer, num_particles ) );
    }

    SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* ParticlesAddr_add(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        ParticlesAddr::num_particles_t const num_particles,
        ParticlesAddr::address_t const q0_addr,
        ParticlesAddr::address_t const mass0_addr,
        ParticlesAddr::address_t const beta0_addr,
        ParticlesAddr::address_t const gamma0_addr,
        ParticlesAddr::address_t const p0c_addr,
        ParticlesAddr::address_t const s_addr,
        ParticlesAddr::address_t const x_addr,
        ParticlesAddr::address_t const y_addr,
        ParticlesAddr::address_t const px_addr,
        ParticlesAddr::address_t const py_addr,
        ParticlesAddr::address_t const zeta_addr,
        ParticlesAddr::address_t const psigma_addr,
        ParticlesAddr::address_t const delta_addr,
        ParticlesAddr::address_t const rpp_addr,
        ParticlesAddr::address_t const rvv_addr,
        ParticlesAddr::address_t const chi_addr,
        ParticlesAddr::address_t const charge_ratio_addr,
        ParticlesAddr::address_t const particle_id_addr,
        ParticlesAddr::address_t const at_element_id_addr,
        ParticlesAddr::address_t const at_turn_addr,
        ParticlesAddr::address_t const state_addr )
    {
        return static_cast< SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* >(
            ::NS(ParticlesAddr_add)( buffer.getCApiPtr(), num_particles,
            q0_addr, mass0_addr, beta0_addr, gamma0_addr, p0c_addr,
            s_addr, x_addr, y_addr, px_addr, py_addr, zeta_addr,
            psigma_addr, delta_addr, rpp_addr, rvv_addr, chi_addr,
            charge_ratio_addr,
            particle_id_addr, at_element_id_addr, at_turn_addr, state_addr ) );
    }

    SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* ParticlesAddr_add(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT buffer,
        ParticlesAddr::num_particles_t const num_particles,
        ParticlesAddr::address_t const q0_addr,
        ParticlesAddr::address_t const mass0_addr,
        ParticlesAddr::address_t const beta0_addr,
        ParticlesAddr::address_t const gamma0_addr,
        ParticlesAddr::address_t const p0c_addr,
        ParticlesAddr::address_t const s_addr,
        ParticlesAddr::address_t const x_addr,
        ParticlesAddr::address_t const y_addr,
        ParticlesAddr::address_t const px_addr,
        ParticlesAddr::address_t const py_addr,
        ParticlesAddr::address_t const zeta_addr,
        ParticlesAddr::address_t const psigma_addr,
        ParticlesAddr::address_t const delta_addr,
        ParticlesAddr::address_t const rpp_addr,
        ParticlesAddr::address_t const rvv_addr,
        ParticlesAddr::address_t const chi_addr,
        ParticlesAddr::address_t const charge_ratio_addr,
        ParticlesAddr::address_t const particle_id_addr,
        ParticlesAddr::address_t const at_element_id_addr,
        ParticlesAddr::address_t const at_turn_addr,
        ParticlesAddr::address_t const state_addr )
    {
        return static_cast< SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* >(
            ::NS(ParticlesAddr_add)( buffer, num_particles,
            q0_addr, mass0_addr, beta0_addr, gamma0_addr, p0c_addr,
            s_addr, x_addr, y_addr, px_addr, py_addr, zeta_addr,
            psigma_addr, delta_addr, rpp_addr, rvv_addr, chi_addr,
            charge_ratio_addr,
            particle_id_addr, at_element_id_addr, at_turn_addr, state_addr ) );
    }

    SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* ParticlesAddr_add_copy(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        ParticlesAddr const& SIXTRL_RESTRICT_REF orig )
    {
        return static_cast< SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* >(
            ::NS(ParticlesAddr_add_copy)(
                buffer.getCApiPtr(), orig.getCApiPtr() ) );
    }

    SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr*
    ParticlesAddr_add_copy(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT buffer,
        ParticlesAddr const& SIXTRL_RESTRICT_REF orig )
    {
        return static_cast< SIXTRL_PARTICLE_ARGPTR_DEC ParticlesAddr* >(
            ::NS(ParticlesAddr_add_copy)( buffer, orig.getCApiPtr() ) );
    }
}

#endif /* C++ */

#endif /* SIXTRACKL_COMMON_PARTICLES_PARTICLES_ADDR_CXX_HEADER_HPP_ */
/* end: sixtracklib/common/particles/particles_addr.hpp */