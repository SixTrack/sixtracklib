#ifndef SIXTRACKLIB_COMMON_TRACK_TRACK_JOB_BUFFER_STORE_H__
#define SIXTRACKLIB_COMMON_TRACK_TRACK_JOB_BUFFER_STORE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <memory>
    #else /* !defined( __cplusplus ) */
        #include <stdbool.h>
        #include <stddef.h>
        #include <stdint.h>
        #include <stdlib.h>
    #endif /* !defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/buffer/assign_address_item.hpp"
    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

typedef struct NS(TrackJobDestSrcBufferIds)
{
    NS(buffer_size_t)   dest_buffer_id;
    NS(buffer_size_t)   src_buffer_id;
}
NS(TrackJobDestSrcBufferIds);

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#if defined( __cplusplus )
namespace SIXTRL_CXX_NAMESPACE
{
    struct TrackJobDestSrcBufferIdsLessCmp
    {
        bool operator()(
            ::NS(TrackJobDestSrcBufferIds) const& SIXTRL_RESTRICT_REF lhs,
            ::NS(TrackJobDestSrcBufferIds) const& SIXTRL_RESTRICT_REF rhs
        ) const SIXTRL_NOEXCEPT
        {
            return ( ( lhs.dest_buffer_id < rhs.dest_buffer_id ) ||
                     ( ( lhs.dest_buffer_id == rhs.dest_buffer_id ) &&
                       ( lhs.src_buffer_id < rhs.src_buffer_id ) ) );
        }
    };

    /* ********************************************************************* */

    class TrackJobBufferStore
    {
        public:

        using buffer_t    = SIXTRL_CXX_NAMESPACE::Buffer;
        using c_buffer_t  = ::NS(Buffer);
        using size_type   = buffer_t::size_type;
        using flags_t     = buffer_t::flags_t;

        static size_type constexpr DEFAULT_BUFFER_CAPACITY =
            buffer_t::DEFAULT_BUFFER_CAPACITY;

        static flags_t constexpr DEFAULT_DATASTORE_FLAGS =
            buffer_t::DEFAULT_DATASTORE_FLAGS;

        SIXTRL_HOST_FN explicit TrackJobBufferStore(
            size_type const buffer_capacity = DEFAULT_BUFFER_CAPACITY,
            flags_t const buffer_flags = DEFAULT_DATASTORE_FLAGS );

        SIXTRL_HOST_FN explicit TrackJobBufferStore(
            buffer_t* SIXTRL_RESTRICT cxx_buffer,
            bool const take_ownership = false );

        SIXTRL_HOST_FN explicit TrackJobBufferStore(
            c_buffer_t* SIXTRL_RESTRICT c99_buffer,
            bool const take_ownership = false,
            bool const delete_ptr_after_move = true );

        SIXTRL_HOST_FN explicit TrackJobBufferStore(
            std::unique_ptr< buffer_t >&& stored_ptr_buffer );

        SIXTRL_HOST_FN explicit TrackJobBufferStore( buffer_t&& cxx_buffer );

        SIXTRL_HOST_FN TrackJobBufferStore( TrackJobBufferStore const& other );
        SIXTRL_HOST_FN TrackJobBufferStore(
            TrackJobBufferStore&& other ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN TrackJobBufferStore& operator=(
            TrackJobBufferStore const& rhs );

        SIXTRL_HOST_FN TrackJobBufferStore& operator=(
            TrackJobBufferStore&& other ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN ~TrackJobBufferStore() = default;

        SIXTRL_HOST_FN bool active() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool owns_buffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t const* ptr_buffer() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN c_buffer_t* ptr_buffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t const* ptr_cxx_buffer() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN buffer_t* ptr_cxx_buffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void clear() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void reset(
            buffer_t::size_type const buffer_capacity,
            buffer_t::flags_t const buffer_flags = DEFAULT_DATASTORE_FLAGS );

        SIXTRL_HOST_FN void reset(
            buffer_t* SIXTRL_RESTRICT cxx_buffer,
            bool const take_ownership = false );

        SIXTRL_HOST_FN void reset(
            c_buffer_t* SIXTRL_RESTRICT c99_buffer,
            bool const take_ownership = false,
            bool const delete_ptr_after_move = true );

        SIXTRL_HOST_FN void reset(
            std::unique_ptr< buffer_t >&& stored_ptr_buffer );

        SIXTRL_HOST_FN void reset( buffer_t&& cxx_buffer );

        private:

        buffer_t*   m_ptr_cxx_buffer;
        c_buffer_t* m_ptr_c99_buffer;
        std::unique_ptr< buffer_t > m_own_buffer;
    };
}
#endif /* C++ */

#endif /* SIXTRACKLIB_COMMON_TRACK_TRACK_JOB_BUFFER_STORE_H__ */
