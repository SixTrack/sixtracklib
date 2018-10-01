#ifndef CXX_SIXTRACKLIB_COMMON_BUFFER_HPP__
#define CXX_SIXTRACKLIB_COMMON_BUFFER_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <limits>
        #include <utility>
    #endif /* defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

namespace SIXTRL_NAMESPACE
{
    class Buffer : public ::NS(Buffer)
    {
        public:

        using size_type     = ::NS(buffer_size_t);
        using flags_t       = ::NS(buffer_flags_t);
        using address_t     = ::NS(buffer_addr_t);
        using addr_diff_t   = ::NS(buffer_addr_diff_t);
        using object_t      = ::NS(Object);
        using c_api_t       = ::NS(Buffer);
        using type_id_t     = ::NS(object_type_id_t);

        SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST size_type
            DEFAULT_BUFFER_CAPACITY = size_type{ 256 };

        SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST flags_t
            DEFAULT_DATASTORE_FLAGS = ::NS(BUFFER_DATASTORE_MEMPOOL);

        SIXTRL_FN SIXTRL_STATIC size_type CalculateBufferSize(
            size_type max_num_objects, size_type max_num_slots,
            size_type max_num_dataptrs, size_type max_num_garbage_ranges,
            size_type slot_size = ::NS(BUFFER_DEFAULT_SLOT_SIZE)
        ) SIXTRL_NOEXCEPT;


        SIXTRL_FN explicit Buffer(
            size_type const buffer_capacity = DEFAULT_BUFFER_CAPACITY,
            flags_t const buffer_flags = DEFAULT_DATASTORE_FLAGS );

        SIXTRL_FN Buffer(
            size_type max_num_objects,
            size_type max_num_slots,
            size_type max_num_dataptrs,
            size_type max_num_garbage_ranges,
            flags_t buffer_flags = DEFAULT_DATASTORE_FLAGS );

        template< typename Ptr >
        SIXTRL_FN Buffer( Ptr SIXTRL_RESTRICT data_buffer_begin,
                          size_type max_buffer_size ) SIXTRL_NOEXCEPT;

        template< typename Ptr >
        SIXTRL_FN Buffer(
            Ptr SIXTRL_RESTRICT begin,
            size_type max_num_objects,
            size_type max_num_slots,
            size_type max_num_dataptrs,
            size_type max_num_garbage_ranges,
            size_type buffer_capacity ) SIXTRL_NOEXCEPT;

        /* TODO: Implement Copy & Move semantics for Buffer */
        SIXTRL_FN Buffer( Buffer const& other ) = delete;
        SIXTRL_FN Buffer( Buffer&& other ) = delete;

        SIXTRL_FN Buffer& operator=( Buffer const& rhs ) = delete;
        SIXTRL_FN Buffer& operator=( Buffer&& rhs ) = delete;

        SIXTRL_FN virtual ~Buffer();

        /* ----------------------------------------------------------------- */

        SIXTRL_FN c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN c_api_t*       getCApiPtr() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN void preset()   SIXTRL_NOEXCEPT;
        SIXTRL_FN void clear( bool set_data_to_zero = false ) SIXTRL_NOEXCEPT;

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

        SIXTRL_FN size_type getSlotSize()           const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getHeaderSize()         const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getSectionHeaderSize()  const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN size_type getNumSlots()           const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getMaxNumSlots()        const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getSlotsSize()          const SIXTRL_NOEXCEPT;

        SIXTRL_FN size_type getNumObjects()         const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getMaxNumObjects()      const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getObjectsSize()        const SIXTRL_NOEXCEPT;

        SIXTRL_FN size_type getNumDataptrs()        const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getMaxNumDataptrs()     const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getDataptrsSize()       const SIXTRL_NOEXCEPT;

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
                *this->getCApiPtr(), std::forward< Args >( args )... );
        }

        template< class T, typename... Args >
        SIXTRL_FN T* add( Args&&... args )
        {
            return T::AddToBuffer(
                *this->getCApiPtr(), std::forward< Args >( args )... );
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        template< class T, typename CPtr, typename Ptr >
        SIXTRL_FN bool canAddObject(
            T const& SIXTRL_RESTRICT_REF obj,
            size_type num_dataptrs,
            CPtr SIXTRL_RESTRICT sizes,
            CPtr SIXTRL_RESTRICT counts,
            Ptr  SIXTRL_RESTRICT ptr_requ_num_objects,
            Ptr  SIXTRL_RESTRICT ptr_requ_num_slots,
            Ptr  SIXTRL_RESTRICT ptr_requ_num_dataptrs ) const SIXTRL_NOEXCEPT;

        template< class T, typename Ptr >
        SIXTRL_FN object_t* addObject(
            T const& SIXTRL_RESTRICT_REF obj,
            type_id_t const type_id,
            size_type num_dataptrs,
            Ptr SIXTRL_RESTRICT offsets,
            Ptr SIXTRL_RESTRICT sizes,
            Ptr SIXTRL_RESTRICT counts );

        private:

        bool allocateDataStore(
            size_type const buffer_capacity, flags_t const flags );
    };
}

/* ************************************************************************* *
 * **** Inline method implementation                                         *
 * ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/mem_pool.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_NAMESPACE
{
    SIXTRL_INLINE Buffer::size_type Buffer::CalculateBufferSize(
        Buffer::size_type const max_num_objects,
        Buffer::size_type const max_num_slots,
        Buffer::size_type const max_num_dataptrs,
        Buffer::size_type const max_num_garbage_ranges,
        Buffer::size_type const slot_size ) SIXTRL_NOEXCEPT
    {
        return NS(ManagedBuffer_calculate_buffer_length)( nullptr,
            max_num_objects, max_num_slots, max_num_dataptrs,
                max_num_garbage_ranges, slot_size );
    }

    SIXTRL_INLINE Buffer::Buffer(
        Buffer::size_type const buffer_capacity,
        Buffer::flags_t const buffer_flags ) :
        ::NS(Buffer)()
    {
        using c_api_t = Buffer::c_api_t;

        c_api_t* _buffer = ::NS(Buffer_preset)( this->getCApiPtr() );

        if( !this->allocateDataStore( buffer_capacity, buffer_flags ) )
        {
            ::NS(Buffer_free)( _buffer );
        }
    }

    SIXTRL_INLINE Buffer::Buffer(
        Buffer::size_type const max_num_objects,
        Buffer::size_type const max_num_slots,
        Buffer::size_type const max_num_dataptrs,
        Buffer::size_type const max_num_garbage_ranges,
        Buffer::flags_t   const buffer_flags  ) :
        ::NS(Buffer)()
    {
        using size_t  = Buffer::size_type;
        using c_api_t = Buffer::c_api_t;

        c_api_t* _buffer = ::NS(Buffer_preset)( this->getCApiPtr() );
        size_t const slot_size = ::NS(Buffer_get_slot_size)( _buffer );

        size_t const requ_buffer_capacity =
            ::NS(ManagedBuffer_calculate_buffer_length)( nullptr,
                max_num_objects, max_num_slots, max_num_dataptrs,
                    max_num_garbage_ranges, slot_size );

        if( ( this->allocateDataStore(
                requ_buffer_capacity, buffer_flags ) ) &&
            ( ( max_num_objects  > size_type{ 0 } ) ||
              ( max_num_slots    > size_type{ 0 } ) ||
              ( max_num_dataptrs > size_type{ 0 } ) ||
              ( max_num_garbage_ranges > size_type{ 0 } ) ) )
        {
            this->reserve( max_num_objects, max_num_slots,
                       max_num_dataptrs, max_num_garbage_ranges );
        }
    }

    template< typename Ptr >
    SIXTRL_INLINE Buffer::Buffer(
        Ptr SIXTRL_RESTRICT data_buffer_begin,
        Buffer::size_type max_buffer_size ) SIXTRL_NOEXCEPT :
        ::NS(Buffer)()
    {
        using c_api_t       = Buffer::c_api_t;
        using ptr_to_raw_t  = SIXTRL_ARGPTR_DEC unsigned char*;

        c_api_t* _buffer = ::NS(Buffer_preset)( this->getCApiPtr() );

        if( 0 != ::NS(Buffer_init)( _buffer, reinterpret_cast< ptr_to_raw_t >(
            data_buffer_begin ), max_buffer_size ) )
        {
            ::NS(Buffer_free)( _buffer );
        }
    }

    template< typename Ptr >
    SIXTRL_INLINE Buffer::Buffer(
        Ptr SIXTRL_RESTRICT data_buffer_begin,
        Buffer::size_type const max_num_objects,
        Buffer::size_type const max_num_slots,
        Buffer::size_type const max_num_dataptrs,
        Buffer::size_type const max_num_garbage_ranges,
        Buffer::size_type const buffer_capacity ) SIXTRL_NOEXCEPT :
        ::NS(Buffer)()
    {
        using c_api_t       = Buffer::c_api_t;
        using size_t        = Buffer::size_type;
        using ptr_to_raw_t  = SIXTRL_ARGPTR_DEC unsigned char*;

        c_api_t* _buffer = ::NS(Buffer_preset)( this->getCApiPtr() );

        ptr_to_raw_t begin =
            reinterpret_cast< ptr_to_raw_t >( data_buffer_begin );

        size_t buffer_size = Buffer::CalculateBufferSize( max_num_objects,
            max_num_slots, max_num_dataptrs, max_num_garbage_ranges );

        int success = ::NS(Buffer_init)( _buffer, begin, buffer_size );

        if( 0 == success )
        {
            success = ::NS(Buffer_reserve)( _buffer, max_num_objects,
                max_num_slots, max_num_dataptrs, max_num_garbage_ranges );
        }

        if( 0 == success )
        {
            ::NS(Buffer_free)( _buffer );
        }
    }

    SIXTRL_INLINE Buffer::~Buffer()
    {
        NS(Buffer_free)( this->getCApiPtr() );
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

    SIXTRL_INLINE void Buffer::clear(
        bool const set_data_to_zero) SIXTRL_NOEXCEPT
    {
        ::NS(Buffer_clear)( this->getCApiPtr(), set_data_to_zero );
        return;
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
        return reinterpret_cast< Ptr >( static_cast< uintptr_t >(
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

    SIXTRL_INLINE Buffer::size_type Buffer::getSize() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_size)( this->getCApiPtr() );
    }

    SIXTRL_INLINE Buffer::size_type
    Buffer::getCapacity() const SIXTRL_NOEXCEPT
    {
        return NS(Buffer_get_capacity)( this->getCApiPtr() );
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
        return NS(ManagedBuffer_needs_remapping)(
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

    template< class T, typename CPtr, typename Ptr >
    SIXTRL_INLINE bool Buffer::canAddObject(
        T const& SIXTRL_RESTRICT_REF obj,
        Buffer::size_type const num_dataptrs,
        CPtr SIXTRL_RESTRICT sizes,
        CPtr SIXTRL_RESTRICT counts,
        Ptr  SIXTRL_RESTRICT ptr_requ_num_objects,
        Ptr  SIXTRL_RESTRICT ptr_requ_num_slots,
        Ptr  SIXTRL_RESTRICT ptr_requ_num_dataptrs ) const SIXTRL_NOEXCEPT
    {
        using size_t            = Buffer::size_type;
        using ptr_const_size_t  = SIXTRL_ARGPTR_DEC size_t const*;
        using ptr_size_t        = SIXTRL_ARGPTR_DEC size_t*;

        return ::NS(Buffer_can_add_object)(
            this->getCApiPtr(), sizeof( T ), num_dataptrs,
            reinterpret_cast< ptr_const_size_t >( sizes  ),
            reinterpret_cast< ptr_const_size_t >( counts ),
            reinterpret_cast< ptr_size_t >( ptr_requ_num_objects ),
            reinterpret_cast< ptr_size_t >( ptr_requ_num_slots ),
            reinterpret_cast< ptr_size_t >( ptr_requ_num_dataptrs ) );
    }

    template< class T, typename Ptr >
    SIXTRL_INLINE Buffer::object_t* Buffer::addObject(
        T const& SIXTRL_RESTRICT_REF obj,
        type_id_t const type_id,
        Buffer::size_type const num_dataptrs,
        Ptr SIXTRL_RESTRICT offsets,
        Ptr SIXTRL_RESTRICT sizes,
        Ptr SIXTRL_RESTRICT counts )
    {
        using size_t            = Buffer::size_type;
        using ptr_const_size_t  = SIXTRL_ARGPTR_DEC size_t const*;

        return ::NS(Buffer_add_object)(
            this->getCApiPtr(), &obj, sizeof( T ),
            type_id, num_dataptrs,
            reinterpret_cast< ptr_const_size_t >( offsets ),
            reinterpret_cast< ptr_const_size_t >( sizes   ),
            reinterpret_cast< ptr_const_size_t >( counts  ) );
    }

    SIXTRL_INLINE bool Buffer::allocateDataStore(
        Buffer::size_type const buffer_capacity, Buffer::flags_t const flags )
    {
        using _this_t    = Buffer;
        using c_api_t    = _this_t::c_api_t;
        using size_t     = _this_t::size_type;
        using address_t  = _this_t::address_t;
        using raw_t      = unsigned char;

        int success      = -1;

        c_api_t* _buffer = this->getCApiPtr();
        size_t const  slot_size = ::NS(Buffer_get_slot_size)( _buffer );

        size_t const min_length = ::NS(ManagedBuffer_calculate_buffer_length)(
                nullptr, 0u, 0u, 0u, 0u, slot_size );

        if( min_length <= buffer_capacity )
        {
            if( ( flags & ::NS(BUFFER_DATASTORE_MEMPOOL) ) ==
                          ::NS(BUFFER_DATASTORE_MEMPOOL) )
            {
                #if !defined( _GPUCODE )

                using mem_pool_t        = ::NS(MemPool);
                using ptr_to_mem_pool_t = mem_pool_t*;

                ptr_to_mem_pool_t mem_pool = ::NS(MemPool_preset)(
                    reinterpret_cast< ptr_to_mem_pool_t >( ::malloc(
                        sizeof( mem_pool_t ) ) ) );

                ::NS(MemPool_set_chunk_size)( mem_pool, slot_size );

                if( ::NS(MemPool_reserve_aligned)(
                        mem_pool, buffer_capacity, slot_size ) )
                {
                    ::NS(AllocResult) result = ::NS(MemPool_append_aligned)(
                        mem_pool, buffer_capacity, slot_size );

                    if( ::NS(AllocResult_valid)( &result ) )
                    {
                        raw_t const z = ( raw_t )0u;

                        SIXTRACKLIB_SET_VALUES( raw_t,
                            ::NS(AllocResult_get_pointer)( &result ),
                            ::NS(AllocResult_get_length)(  &result ), z );

                        success = ::NS(Buffer_init_on_flat_memory)( _buffer,
                            ::NS(AllocResult_get_pointer)( &result ),
                            ::NS(AllocResult_get_length)(  &result ) );

                        if( success == 0 )
                        {
                            _buffer->datastore_addr = static_cast< address_t >(
                                reinterpret_cast< uintptr_t >( mem_pool ) );

                            _buffer->datastore_flags = flags |
                                ::NS(BUFFER_USES_DATASTORE) |
                                ::NS(BUFFER_OWNS_DATASTORE) |
                                ::NS(BUFFER_DATASTORE_MEMPOOL) |
                                ::NS(BUFFER_DATASTORE_ALLOW_APPENDS) |
                                ::NS(BUFFER_DATASTORE_ALLOW_CLEAR)   |
                                ::NS(BUFFER_DATASTORE_ALLOW_REMAPPING) |
                                ::NS(BUFFER_DATASTORE_ALLOW_RESIZE);
                        }
                    }
                }

                #endif /* defined( _GPUCODE ) */
            }
        }

        if( success != 0 )
        {
            if( ( flags & ::NS(BUFFER_DATASTORE_MEMPOOL) ) ==
                          ::NS(BUFFER_DATASTORE_MEMPOOL) )
            {
                #if !defined(_GPUCODE )

                using mem_pool_t     = ::NS(MemPool);
                using ptr_mem_pool_t = mem_pool_t*;

                ptr_mem_pool_t pool = reinterpret_cast< ptr_mem_pool_t >(
                    static_cast< uintptr_t >(
                        ::NS(Buffer_get_datastore_begin_addr)( _buffer ) ) );

                ::NS(MemPool_free)( pool );
                ::free( pool );

                _buffer->datastore_addr = address_t{ 0u };
                #endif /* !defined( _GPUCODE ) */
            }
        }

        return ( success == 0 );
    }
}

#endif /* defined( __cplusplus ) */

#endif /* CXX_SIXTRACKLIB_COMMON_BUFFER_HPP__ */

/* end: sixtracklib/common/buffer.hpp */
