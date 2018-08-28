#ifndef CXX_SIXTRACKLIB_COMMON_BUFFER_HPP__
#define CXX_SIXTRACKLIB_COMMON_BUFFER_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdint>
    #include <cstdlib>
    #include <limits>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

namespace SIXTRL_NAMESPACE
{
    struct Buffer : public ::NS(Buffer)
    {
        using size_type     = ::NS(buffer_size_t);
        using flags_t       = ::NS(buffer_flags_t);
        using address_t     = ::NS(buffer_addr_t);
        using addr_diff_t   = ::NS(buffer_addr_diff_t);
        using object_t      = ::NS(Object);
        using c_api_t       = ::NS(Buffer);
        using type_id_t     = ::NS(object_type_id_t);

        SIXTRL_FN Buffer() = default;

        SIXTRL_FN Buffer( Buffer const& other ) = default;
        SIXTRL_FN Buffer( Buffer&& other ) = default;

        SIXTRL_FN Buffer& operator=( Buffer const& rhs ) = default;
        SIXTRL_FN Buffer& operator=( Buffer&& rhs ) = default;

        SIXTRL_FN ~Buffer() = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN SIXTRL_STATIC Buffer* Make(
            size_type buffer_capacity,
            flags_t const buffer_flags = flags_t{} );

        template< typename Ptr >
        SIXTRL_FN SIXTRL_STATIC Buffer* Make(
            Ptr SIXTRL_RESTRICT data_buffer_begin,
            size_type const max_data_buffer_size );

        SIXTRL_FN SIXTRL_STATIC Buffer* Make(
            size_type max_num_objects,
            size_type max_num_slots,
            size_type max_num_dataptrs,
            size_type max_num_garbage_ranges,
            flags_t const buffer_flags = flags_t{} );

        /* ----------------------------------------------------------------- */

        SIXTRL_FN c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN c_api_t*       getCApiPtr() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN void preset()   SIXTRL_NOEXCEPT;
        SIXTRL_FN void clear()    SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN flags_t   getFlags()               const SIXTRL_NOEXCEPT;

        SIXTRL_FN bool hasDataStore()                const SIXTRL_NOEXCEPT;
        SIXTRL_FN bool usesDataStore()               const SIXTRL_NOEXCEPT;
        SIXTRL_FN bool ownsDataStore()               const SIXTRL_NOEXCEPT;

        SIXTRL_FN bool allowModifyContents()         const SIXTRL_NOEXCEPT;
        SIXTRL_FN bool allowClear()                  const SIXTRL_NOEXCEPT;
        SIXTRL_FN bool allowAppend()                 const SIXTRL_NOEXCEPT;
        SIXTRL_FN bool allowDelete()                 const SIXTRL_NOEXCEPT;
        SIXTRL_FN bool allowRemap()                  const SIXTRL_NOEXCEPT;
        SIXTRL_FN bool allowResize()                 const SIXTRL_NOEXCEPT;

        SIXTRL_FN bool usesMempoolDataStore()        const SIXTRL_NOEXCEPT;
        SIXTRL_FN bool usesSpecialOpenCLDataStore()  const SIXTRL_NOEXCEPT;
        SIXTRL_FN bool usesSpecialCudaDataStore()    const SIXTRL_NOEXCEPT;

        SIXTRL_FN flags_t getDataStoreSpecialFlags() const SIXTRL_NOEXCEPT;
        SIXTRL_FN void    setDataStoreSpecialFlags( flags_t flags ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN address_t getDataStoreBeginAddr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getDataBeginAddr()      const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getDataEndAddr()        const SIXTRL_NOEXCEPT;

        template< typename CPtr > SIXTRL_FN CPtr dataBegin() const SIXTRL_NOEXCEPT;
        template< typename CPtr > SIXTRL_FN CPtr dataEnd()   const SIXTRL_NOEXCEPT;

        template< typename Ptr >  SIXTRL_FN Ptr dataBegin() SIXTRL_NOEXCEPT;
        template< typename Ptr >  SIXTRL_FN Ptr dataEnd()   SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN size_type size()                  const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type capacity()              const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type headerSize()            const SIXTRL_NOEXCEPT;

        SIXTRL_FN size_type getSize()               const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getCapacity()           const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getHeaderSize()         const SIXTRL_NOEXCEPT;

        SIXTRL_FN size_type getSlotSize()           const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getHeaderSize()         const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getSectionHeaderSize()  const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN size_type getNumSlots()           const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getMaxNumSlots()        const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getSlotsSize()          const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getMaxSlotsSize()       const SIXTRL_NOEXCEPT;

        SIXTRL_FN size_type getNumObjects()         const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getMaxNumObjects()      const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getObjectsSize()        const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getMaxObjectsSize()     const SIXTRL_NOEXCEPT;

        SIXTRL_FN size_type getNumDataptrs()        const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getMaxNumDataptrs()     const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getDataptrsSize()       const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getMaxDataptrsSize()    const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN address_t getIndexBeginAddr()     const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getIndexEndAddr()       const SIXTRL_NOEXCEPT;

        template< typename CPtr > SIXTRL_FN CPtr indexBegin() const SIXTRL_NOEXCEPT;
        template< typename CPtr > SIXTRL_FN CPtr indexEnd()   const SIXTRL_NOEXCEPT;

        template< typename Ptr >  SIXTRL_FN Ptr  indexBegin() SIXTRL_NOEXCEPT;
        template< typename Ptr >  SIXTRL_FN Ptr  indexEnd()   SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN bool needsRemapping() const SIXTRL_NOEXCEPT;
        SIXTRL_FN bool remap() SIXTRL_NOEXCEPT;

        SIXTRL_FN bool reserve( size_type const new_max_num_objects );

        SIXTRL_FN bool reserve( size_type const new_max_num_objects,
                                size_type const new_max_num_slots );

        SIXTRL_FN bool reserve( size_type const new_max_num_objects,
                                size_type const new_max_num_slots,
                                size_type const new_max_num_dataptrs );

        SIXTRL_FN bool reserve( size_type const new_max_num_objects,
                                size_type const new_max_num_slots,
                                size_type const new_max_num_dataptrs,
                                size_type const new_max_num_garbage_ranges );

        /* ----------------------------------------------------------------- */

        template< class T, typename... Args  >
        SIXTRL_FN bool canAdd( Args&&... args ) const SIXTRL_NOEXCEPT
        {
            return T::CanAddToBuffer(
                this->getCApiPtr(), std::forward< Args >( args )... );
        }

        template< class T, typename... Args >
        SIXTRL_FN T* createNew( Args&&... args )
        {
            return T::CreateNewOnBuffer(
                this->getCApiPtr(), std::forward< Args >( args )... );
        }

        template< class T, typename... Args >
        SIXTRL_FN T* add( Args&&... args )
        {
            return T::AddToBuffer(
                this->getCApiPtr(), std::forward< Args >( args )... );
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        template< class T, typename Ptr >
        SIXTRL_FN bool canAddObject(
            T const& SIXTRL_RESTRICT_REF obj, size_type num_dataptrs,
            Ptr SIXTRL_RESTRICT sizes, Ptr SIXTRL_RESTRICT counts,
            size_type& SIXTRL_RESTRICT_REF num_objects,
            size_type& SIXTRL_RESTRICT_REF num_slots,
            size_type& SIXTRL_RESTRICT_REF num_dataptrs ) const SIXTRL_NOEXCEPT;

        template< class T, typename Ptr >
        SIXTRL_FN object_t* addObject(
            T const& SIXTRL_RESTRICT_REF obj,
            type_id_t const type_id,
            size_type num_dataptrs,
            Ptr SIXTRL_RESTRICT offsets,
            Ptr SIXTRL_RESTRICT sizes,
            Ptr SIXTRL_RESTRICT counts );
    };
}

/* ************************************************************************* *
 * **** Inline method implementation                                         *
 * ************************************************************************* */

namespace SIXTRL_NAMESPACE
{
    SIXTRL_INLINE Buffer* Buffer::Make(
        Buffer::size_type const buffer_capacity,
        Buffer::flags_t const buffer_flags )
    {

    }

    template< typename Ptr >
    SIXTRL_INLINE Buffer* Buffer::Make(
        Ptr SIXTRL_RESTRICT data_buffer_begin,
        Buffer::size_type const max_data_buffer_size )
    {

    }

    SIXTRL_INLINE Buffer* Buffer::Make(
        Buffer::size_type const max_num_objects,
        Buffer::size_type const max_num_slots,
        Buffer::size_type const max_num_dataptrs,
        Buffer::size_type const max_num_garbage_ranges,
        Buffer::flags_t   const buffer_flags  )
    {

    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE Buffer::c_api_t const* Buffer::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        return static_cast< Buffer::c_api_t const* >( this );
    }

    SIXTRL_INLINE Buffer::c_api_t* Buffer::getCApiPtr() SIXTRL_NOEXCEPT
    {
        return static_cast< Buffer::c_api_t* >( this );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE void Buffer::preset() SIXTRL_NOEXCEPT
    {
        NS(Buffer_preset)( this->getCApiPtr() );
        return;
    }

    SIXTRL_INLINE void Buffer::clear() SIXTRL_NOEXCEPT
    {
        NS(Buffer_clear)( this->getCApiPtr() );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE Buffer::flags_t Buffer::getFlags() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_flags)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool Buffer::hasDataStore() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_has_datastore)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool Buffer::usesDataStore() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_uses_datastore)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool Buffer::ownsDataStore() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_owns_datastore)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool Buffer::allowModifyContents() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_allow_modify_datastore_contents)(
            this->getCApiPtr() );
    }

    SIXTRL_INLINE bool Buffer::allowClear() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_allow_clear)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool Buffer::allowAppend() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_allow_append_objects)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool Buffer::allowDelete() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_allow_delete_objects)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool Buffer::allowRemap() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_allow_remapping)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool Buffer::allowResize() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_allow_resize)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool Buffer::usesMempoolDataStore() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_uses_mempool_datastore)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool Buffer::usesSpecialOpenCLDataStore() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_uses_special_opencl_datastore)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool Buffer::usesSpecialCudaDataStore() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_uses_special_cuda_datastore)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::flags_t
    Buffer::getDataStoreSpecialFlags() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_datastore_special_flags)( this->getCApiPtr() );
    }

    SIXTRL_INLINE void Buffer::setDataStoreSpecialFlags(
        Buffer::flags_t const flags ) SIXTRL_NOEXCEPT
    {
         NS(Buffer_set_datastore_special_flags)( this->getCApiPtr(), flags );
         return;
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE Buffer::address_t
    Buffer::getDataStoreBeginAddr() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_datastore_begin_addr)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::address_t
    Buffer::getDataBeginAddr() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_data_begin_addr)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::address_t
    Buffer::getDataEndAddr() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_data_end_addr)( this->getCApiPtr() );
    }

    template< typename CPtr >
    SIXTRL_INLINE CPtr Buffer::dataBegin() const SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< CPtr >( static_cast< uintptr_t >(
            NS(Buffer_get_data_begin_addr)( this->getCApiPtr() ) ) );
    }

    template< typename CPtr >
    SIXTRL_INLINE CPtr Buffer::dataEnd() const SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< CPtr >( static_cast< uintptr_t >(
            NS(Buffer_get_data_end_addr)( this->getCApiPtr() ) ) );
    }

    template< typename Ptr >
    SIXTRL_INLINE Ptr Buffer::dataBegin() SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< Ptr >( static_cast< uintptr_t >(
            NS(Buffer_get_data_begin_addr)( this->getCApiPtr() ) ) );
    }

    template< typename Ptr >
    SIXTRL_INLINE Ptr Buffer::dataEnd() SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< CPtr >( static_cast< uintptr_t >(
            NS(Buffer_get_data_end_addr)( this->getCApiPtr() ) ) );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE Buffer::size_type
    Buffer::size() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_size)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::size_type
    Buffer::capacity() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_capacity)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::size_type
    Buffer::headerSize() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_header_size)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::size_type getSize() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_size)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::size_type
    Buffer::getCapacity() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_capacity)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::size_type
    Buffer::getHeaderSize() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_size)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::size_type
    Buffer::getSlotSize() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_slot_size)( this->getCApiPtr() );
    }


    SIXTRL_INLINE Buffer::size_type
    Buffer::getHeaderSize() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_header_size)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::size_type
    Buffer::getSectionHeaderSize() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_section_header_size)( this->getCApiPtr() );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE Buffer::size_type
    Buffer::getNumSlots() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_num_of_slots)( this->getCApiPtr() );

    }

    SIXTRL_INLINE Buffer::size_type
    Buffer::getMaxNumSlots() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_max_num_of_slots)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::size_type
    Buffer::getSlotsSize() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_slots_size)( this->getCApiPtr() );
    }



    SIXTRL_INLINE Buffer::size_type
    Buffer::getNumObjects() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_num_of_objects)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::size_type
    Buffer::getMaxNumObjects() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_max_num_of_objects)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::size_type
    Buffer::getObjectsSize() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_objects_size)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::size_type
    Buffer::getNumDataptrs() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_num_of_dataptrs)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::size_type
    Buffer::getMaxNumDataptrs() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_max_num_of_dataptrs)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::size_type
    Buffer::getDataptrsSize() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_dataptrs_size)( this->getCApiPtr() );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE Buffer::address_t
    Buffer::getIndexBeginAddr() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_objects_begin_addr)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::address_t
    Buffer::getIndexEndAddr() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_objects_end_addr)( this->getCApiPtr() );
    }

    template< typename CPtr >
    SIXTRL_INLINE CPtr Buffer::indexBegin() const SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< CPtr >( static_cast< uintptr_t >(
            NS(Buffer_get_objects_begin_addr)( this->getCApiPtr() ) ) );
    }

    template< typename CPtr >
    SIXTRL_INLINE CPtr Buffer::indexEnd() const SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< CPtr >( static_cast< uintptr_t >(
            NS(Buffer_get_objects_end_addr)( this->getCApiPtr() ) ) );
    }

    template< typename Ptr >
    SIXTRL_INLINE Ptr Buffer::indexBegin() SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< Ptr >( static_cast< uintptr_t >(
            NS(Buffer_get_objects_begin_addr)( this->getCApiPtr() ) ) );
    }

    template< typename Ptr >
    SIXTRL_INLINE Ptr Buffer::indexEnd() SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< Ptr >( static_cast< uintptr_t >(
            NS(Buffer_get_objects_end_addr)( this->getCApiPtr() ) ) );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE bool Buffer::needsRemapping() const SIXTRL_NOEXCEPT
    {
        return NS(BufferMem_needs_remapping)(
            reinterpret_cast< SIXTRL_ARGPTR_DEC unsigned char const* >(
                static_cast< uintptr_t >( NS(Buffer_get_data_begin_addr)(
                    this->getCApiPtr() ) ) ),
            NS(Buffer_get_slot_size)( this->getCApiPtr() ) );
    }

    SIXTRL_INLINE bool Buffer::remap() SIXTRL_NOEXCEPT
    {
        return ( 0 == NS(Buffer_remap)( this->getCApiPtr() ) );
    }

    SIXTRL_INLINE bool Buffer::reserve(
        Buffer::Buffer::size_type const new_max_num_objects )
    {
        return ( 0 == NS(Buffer_reserve)( this->getCApiPtr(),
            new_max_num_objects,
            NS(Buffer_get_max_num_of_slots)( this->getCApiPtr() ),
            NS(Buffer_get_max_num_of_dataptrs)( this->getCApiPtr() ),
            NS(Buffer_get_num_of_garbage_ranges)( this->getCApiPtr() ) ) );

    }

    SIXTRL_INLINE bool Buffer::reserve(
        Buffer::size_type const new_max_num_objects,
        Buffer::size_type const new_max_num_slots )
    {
        return ( 0 == NS(Buffer_reserve)( this->getCApiPtr(),
            new_max_num_objects, new_max_num_slots,
            NS(Buffer_get_max_num_of_dataptrs)( this->getCApiPtr() ),
            NS(Buffer_get_num_of_garbage_ranges)( this->getCApiPtr() ) ) );
    }

    SIXTRL_INLINE bool Buffer::reserve(
        Buffer::size_type const new_max_num_objects,
        Buffer::size_type const new_max_num_slots,
        Buffer::size_type const new_max_num_dataptrs )
    {
        return ( 0 == NS(Buffer_reserve)( this->getCApiPtr(),
            new_max_num_objects, new_max_num_slots, new_max_num_dataptrs,
            NS(Buffer_get_num_of_garbage_ranges)( this->getCApiPtr() ) ) );
    }

    SIXTRL_INLINE bool Buffer::reserve(
        Buffer::size_type const new_max_num_objects,
        Buffer::size_type const new_max_num_slots,
        Buffer::size_type const new_max_num_dataptrs,
        Buffer::size_type const new_max_num_garbage_ranges )
    {
        return ( 0 == NS(Buffer_reserve)( this->getCApiPtr(),
            new_max_num_objects, new_max_num_slots, new_max_num_dataptrs,
                new_max_num_garbage_ranges ) );
    }

    /* ----------------------------------------------------------------- */

    template< class T, typename... Args  >
    SIXTRL_INLINE bool Buffer::canAdd( Args&&... args ) const SIXTRL_NOEXCEPT
    {
        return T::CanAddToBuffer(
            this->getCApiPtr(), std::forward< Args >( args )... );
    }

    template< class T, typename... Args >
    SIXTRL_INLINE T* Buffer::createNew( Args&&... args )
    {
        return T::CreateNewOnBuffer(
            this->getCApiPtr(), std::forward< Args >( args )... );
    }

    template< class T, typename... Args >
    SIXTRL_INLINE T* Buffer::add( Args&&... args )
    {
        return T::AddToBuffer(
            this->getCApiPtr(), std::forward< Args >( args )... );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< class T, typename Ptr >
    SIXTRL_INLINE bool Buffer::canAddObject(
        T const& SIXTRL_RESTRICT_REF obj,
        Buffer::size_type const num_dataptrs,
        Ptr SIXTRL_RESTRICT sizes,
        Ptr SIXTRL_RESTRICT counts,
        Buffer::size_type& SIXTRL_RESTRICT_REF num_objects,
        Buffer::size_type& SIXTRL_RESTRICT_REF num_slots,
        Buffer::size_type& SIXTRL_RESTRICT_REF num_dataptrs ) const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_can_add_object)(
            this->getCApiPtr(), sizeof( T ), num_dataptrs, sizes, counts,
                &num_objects, &num_slots, &num_dataptrs );
    }

    template< class T, typename Ptr >
    SIXTRL_INLINE Buffer::object_t* Buffer::addObject(
        T const& SIXTRL_RESTRICT_REF obj, type_id_t const type_id,
        Buffer::size_type const num_dataptrs, Ptr SIXTRL_RESTRICT offsets,
        Ptr SIXTRL_RESTRICT sizes, Ptr SIXTRL_RESTRICT counts )
    {
        return NS(Buffer_add_object)( this->getCApiPtr(), &obj, sizeof( T ),
            type_id, num_dataptrs, offsets, sizes, counts );
    }

}

#endif /* defined( __cplusplus ) */

#endif /* CXX_SIXTRACKLIB_COMMON_BUFFER_HPP__ */

/* end: sixtracklib/common/buffer.hpp */
