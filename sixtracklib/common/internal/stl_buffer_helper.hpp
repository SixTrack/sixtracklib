#ifndef SIXTRACKLIB_COMMON_INTERNAL_STL_BUFFER_HELPER_CXX_HPP__
#define SIXTRACKLIB_COMMON_INTERNAL_STL_BUFFER_HELPER_CXX_HPP__

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <algorithm>
    #include <cstddef>
    #include <cstdlib>
    #include <iterator>
    #include <map>
    #include <unordered_map>
    #include <vector>
    #include <utility>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename Iter, typename T >
    SIXTRL_STATIC SIXTRL_HOST_FN bool Vector_sorted_has_value(
        Iter begin, Iter end, T const& SIXTRL_RESTRICT_REF value,
        bool const check_sorting = false );

    template< typename T, class Allocator >
    SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_CXX_NAMESPACE::arch_status_t
    Vector_add_sorted( std::vector< T, Allocator >& SIXTRL_RESTRICT_REF vector,
        T const& SIXTRL_RESTRICT_REF value_to_insert,
        T const& SIXTRL_RESTRICT_REF save_value_to_compare_against = T{},
        bool const keep_ordered = true );

    template< typename T, class Allocator, class IncrementPred >
    SIXTRL_STATIC SIXTRL_HOST_FN void Vector_add_next_sorted(
        std::vector< T, Allocator >& SIXTRL_RESTRICT_REF vector,
        IncrementPred increment_fn );

    template< typename T, class Allocator >
    SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_CXX_NAMESPACE::arch_status_t
    Vector_sorted_remove_key(
        std::vector< T, Allocator >& SIXTRL_RESTRICT_REF vector,
        T const& SIXTRL_RESTRICT_REF key );

    /* --------------------------------------------------------------------- */

    template< typename Key, typename Value, class Cmp, class Allocator >
    SIXTRL_STATIC SIXTRL_HOST_FN bool Map_has_key(
        std::map< Key, Value, Cmp, Allocator > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key );

    template< typename Key, typename Value, class Cmp, class Allocator >
    SIXTRL_STATIC SIXTRL_HOST_FN bool Map_has_value_for_key(
        std::map< Key, Value, Cmp, Allocator > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        Value const& SIXTRL_RESTRICT_REF value );

    template< typename Key, typename Value, class Cmp, class Allocator >
    SIXTRL_STATIC SIXTRL_HOST_FN Value const& Map_get_value_or_default_for_key(
        std::map< Key, Value, Cmp, Allocator > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        Value const& SIXTRL_RESTRICT_REF default_value = Value{} );


    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc>
    SIXTRL_STATIC SIXTRL_HOST_FN bool Map_ordered_vec_has_value(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc > const&
            SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        T const& SIXTRL_RESTRICT_REF value );

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc>
    SIXTRL_STATIC SIXTRL_HOST_FN bool Map_ordered_vec_empty(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc > const&
            SIXTRL_RESTRICT_REF map, Key const& SIXTRL_RESTRICT_REF key );

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc>
    SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_CXX_NAMESPACE::arch_size_t
    Map_ordered_vec_size( std::map< Key, std::vector< T, VecAlloc >,
            Cmp, Alloc > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key );

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc>
    SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_CXX_NAMESPACE::arch_status_t
    Map_remove_if_ordered_vec_empty( std::map< Key, std::vector< T, VecAlloc >,
            Cmp, Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key );

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc>
    SIXTRL_CXX_NAMESPACE::arch_status_t Map_remove_value_from_ordered_vec(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc >&
            SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        T const& SIXTRL_RESTRICT_REF value,
        bool const remove_entry_if_ordered_vec_is_empty = false );

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc>
    SIXTRL_CXX_NAMESPACE::arch_status_t Map_ordered_vec_insert_value(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc >&
            SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        T const& SIXTRL_RESTRICT_REF value,
        T const& SIXTRL_RESTRICT_REF save_value_to_compare_against = T{},
        bool const keep_ordered = true );

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc>
    SIXTRL_STATIC SIXTRL_HOST_FN T const* Map_ordered_vec_begin(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc > const&
            SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key );

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc>
    SIXTRL_STATIC SIXTRL_HOST_FN T const* Map_ordered_vec_end(
        std::map< Key, std::vector< T, VecAlloc >,
            Cmp, Alloc > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key );

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc>
    SIXTRL_STATIC SIXTRL_HOST_FN T* Map_ordered_vec_begin(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc >&
            SIXTRL_RESTRICT_REF map, Key const& SIXTRL_RESTRICT_REF key );

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc>
    SIXTRL_STATIC SIXTRL_HOST_FN T* Map_ordered_vec_end(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc >&
            SIXTRL_RESTRICT_REF map, Key const& SIXTRL_RESTRICT_REF key );

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc>
    SIXTRL_STATIC SIXTRL_HOST_FN std::size_t Map_ordered_vec_get_value_index(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc > const&
            SIXTRL_RESTRICT_REF map, Key const& SIXTRL_RESTRICT_REF key,
                T const& SIXTRL_RESTRICT_REF value );

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc>
    SIXTRL_STATIC SIXTRL_HOST_FN T* Map_ordered_vec_get_ptr_value(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc >&
            SIXTRL_RESTRICT_REF map, Key const& SIXTRL_RESTRICT_REF key,
                T const& SIXTRL_RESTRICT_REF value );

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc>
    SIXTRL_STATIC SIXTRL_HOST_FN T const* Map_ordered_vec_get_ptr_const_value(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc > const&
            SIXTRL_RESTRICT_REF map, Key const& SIXTRL_RESTRICT_REF key,
                T const& SIXTRL_RESTRICT_REF value );

    /* --------------------------------------------------------------------- */

    template< typename Key, typename Value, class Hash,
              class KeyEqual, class Alloc >
    SIXTRL_STATIC SIXTRL_HOST_FN bool Map_has_key(
        std::unordered_map< Key, Value, Hash, KeyEqual, Alloc > const&
            SIXTRL_RESTRICT_REF map, Key const& SIXTRL_RESTRICT_REF key );

    template< typename Key, typename Value, class Hash,
              class KeyEqual, class Alloc >
    SIXTRL_STATIC SIXTRL_HOST_FN bool Map_has_value_for_key(
        std::unordered_map< Key, Value, Hash, KeyEqual, Alloc >  const&
            SIXTRL_RESTRICT_REF map, Key const& SIXTRL_RESTRICT_REF key,
        Value const& SIXTRL_RESTRICT_REF value );

    template< typename Key, typename Value, class Hash,
              class KeyEqual, class Alloc >
    SIXTRL_STATIC SIXTRL_HOST_FN Value const& Map_get_value_or_default_for_key(
        std::unordered_map< Key, Value, Hash, KeyEqual, Alloc >  const&
            SIXTRL_RESTRICT_REF map, Key const& SIXTRL_RESTRICT_REF key,
        Value const& SIXTRL_RESTRICT_REF default_value = Value{} );

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_STATIC SIXTRL_HOST_FN bool Map_ordered_vec_has_value(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash, KeyEqual,
            Alloc > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        T const& SIXTRL_RESTRICT_REF value );

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_STATIC SIXTRL_HOST_FN bool Map_ordered_vec_empty(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash, KeyEqual,
            Alloc > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key );

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_CXX_NAMESPACE::arch_status_t
    Map_remove_if_ordered_vec_empty(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash, KeyEqual,
            Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key );

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_CXX_NAMESPACE::arch_status_t
    Map_remove_value_from_ordered_vec( std::unordered_map< Key, std::vector<
        T, VecAlloc >, Hash, KeyEqual, Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        T const& SIXTRL_RESTRICT_REF value,
        bool remove_entry_if_ordered_vec_is_empty = false );

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_CXX_NAMESPACE::arch_status_t
    Map_ordered_vec_insert_value(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash, KeyEqual,
            Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        T const& SIXTRL_RESTRICT_REF value,
        T const& SIXTRL_RESTRICT_REF save_value_to_compare_against = T{},
        bool const keep_ordered = true );

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_STATIC SIXTRL_HOST_FN T const* Map_ordered_vec_begin(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash, KeyEqual,
            Alloc > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key );

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_STATIC SIXTRL_HOST_FN T const* Map_ordered_vec_end(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash, KeyEqual,
            Alloc > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key );

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_STATIC SIXTRL_HOST_FN T* Map_ordered_vec_begin(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash, KeyEqual,
            Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key );

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_STATIC SIXTRL_HOST_FN T* Map_ordered_vec_end(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash,
            KeyEqual, Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key );

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_STATIC SIXTRL_HOST_FN std::size_t Map_ordered_vec_get_value_index(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash,
            KeyEqual, Alloc > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        T const& SIXTRL_RESTRICT_REF value );

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_STATIC SIXTRL_HOST_FN T* Map_ordered_vec_get_ptr_value(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash,
            KeyEqual, Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        T const& SIXTRL_RESTRICT_REF value );

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_STATIC SIXTRL_HOST_FN T const* Map_ordered_vec_get_ptr_const_value(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash,
            KeyEqual, Alloc > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        T const& SIXTRL_RESTRICT_REF value );
}

#endif /* SIXTRACKLIB_COMMON_INTERNAL_STL_BUFFER_HELPER_CXX_HPP__ */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename Iter, typename T >
    SIXTRL_INLINE bool Vector_sorted_has_value(
        Iter begin, Iter end, T const& SIXTRL_RESTRICT_REF value,
        bool const check_sorting )
    {
        bool has_value = false;

        if( ( std::distance( begin, end ) > std::ptrdiff_t{ 0 } ) &&
            ( ( !check_sorting ) || (  std::is_sorted( begin, end ) ) ) )
        {
            has_value = std::binary_search( begin, end, value );
        }

        return has_value;
    }

    template< typename T, class Allocator >
    SIXTRL_INLINE SIXTRL_CXX_NAMESPACE::arch_status_t Vector_add_sorted(
        std::vector< T, Allocator >& SIXTRL_RESTRICT_REF vector,
        T const& SIXTRL_RESTRICT_REF value_to_insert,
        T const& SIXTRL_RESTRICT_REF save_value_to_compare_against,
        bool const keep_ordered )
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        using vec_t  = std::vector< T, Allocator >;
        using value_type = typename vec_t::value_type;

        st::arch_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        SIXTRL_ASSERT( std::is_sorted( vector.begin(), vector.end() ) );

        if( !std::binary_search( vector.begin(), vector.end(),
                value_to_insert ) )
        {
            value_type const prev_back = ( !vector.empty() )
                ? vector.back() : save_value_to_compare_against;

            vector.push_back( value_to_insert );
            if( ( keep_ordered ) && ( prev_back > value_to_insert ) )
            {
                std::sort( vector.begin(), vector.end() );
            }

            SIXTRL_ASSERT( ( !keep_ordered ) ||
                ( std::is_sorted( vector.begin(), vector.end() ) ) );

            status = st::ARCH_STATUS_SUCCESS;
        }

        return status;
    }

    template< typename T, class Allocator, class IncrementPred >
    void Vector_add_next_sorted(
        std::vector< T, Allocator >& SIXTRL_RESTRICT_REF vector,
        IncrementPred increment_fn )
    {
        if( !vector.empty() )
        {
            SIXTRL_ASSERT( std::is_sorted( vector.begin(), vector.end() ) );
            vector.emplace_back( increment_fn( vector.back() ) );
        }
        else
        {
            vector.emplace_back( increment_fn( T{} ) );
        }
    }

    template< typename T, class Allocator >
    SIXTRL_CXX_NAMESPACE::arch_status_t Vector_sorted_remove_key(
        std::vector< T, Allocator >& SIXTRL_RESTRICT_REF vector,
        T const& SIXTRL_RESTRICT_REF key )
    {
        SIXTRL_CXX_NAMESPACE::arch_status_t status =
            SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;

        if( ( !vector.empty() ) &&
            ( std::is_sorted( vector.begin(), vector.end() ) ) )
        {
            auto it = std::lower_bound( vector.begin(), vector.end(),
                key );

            if( it != vector.end() )
            {
                vector.erase( it );
                status = SIXTRL_CXX_NAMESPACE::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    /* --------------------------------------------------------------------- */


    template< typename Key, typename Value, class Cmp, class Allocator >
    SIXTRL_INLINE bool Map_has_key(
        std::map< Key, Value, Cmp, Allocator > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key )
    {
        return ( map.find( key ) != map.end() );
    }

    template< typename Key, typename Value, class Cmp, class Allocator >
    SIXTRL_INLINE bool Map_has_value_for_key(
        std::map< Key, Value, Cmp, Allocator > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        Value const& SIXTRL_RESTRICT_REF value )
    {
        auto it = map.find( key );
        return ( ( it != map.end() ) && ( it->second == value ) );
    }

    template< typename Key, typename Value, class Cmp, class Allocator >
    SIXTRL_INLINE Value const& Map_get_value_or_default_for_key(
        std::map< Key, Value, Cmp, Allocator > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        Value const& SIXTRL_RESTRICT_REF default_value  )
    {
        auto it = map.find( key );
        return ( it != map.end() ) ? it->second : default_value;
    }

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc >
    SIXTRL_INLINE bool Map_ordered_vec_has_value(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc > const&
            SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        T const& SIXTRL_RESTRICT_REF value )
    {
        bool has_value = false;
        auto it = map.find( key );
        if( it != map.end() )
        {
            SIXTRL_ASSERT( std::is_sorted(
                it->second.begin(), it->second.end() ) );

            has_value = std::binary_search( it->second.begin(),
                it->second.end(), value );
        }

        return has_value;
    }

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc >
    SIXTRL_INLINE  bool Map_ordered_vec_empty(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc > const&
            SIXTRL_RESTRICT_REF map, Key const& SIXTRL_RESTRICT_REF key )
    {
        auto it = map.find( key );
        return ( ( it != map.end() ) && ( it->second.empty() ) );
    }

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc >
    SIXTRL_INLINE SIXTRL_CXX_NAMESPACE::arch_size_t Map_ordered_vec_size(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc > const&
            SIXTRL_RESTRICT_REF map, Key const& SIXTRL_RESTRICT_REF key )
    {
        auto it = map.find( key );
        return ( it != map.end() )
            ? it->second.size() : SIXTRL_CXX_NAMESPACE::arch_size_t{ 0 };
    }

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc >
    SIXTRL_INLINE SIXTRL_CXX_NAMESPACE::arch_status_t
    Map_remove_if_ordered_vec_empty( std::map< Key, std::vector< T, VecAlloc >,
            Cmp, Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key )
    {
        namespace st = SIXTRL_CXX_NAMESPACE;

        st::arch_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        auto it = map.find( key );

        if( it != map.end() )
        {
            SIXTRL_ASSERT( std::is_sorted(
                it->second.begin(), it->seconed.end() ) );

            if( it->second.empty() )
            {
                map.erase( it );
            }

            status = st::ARCH_STATUS_SUCCESS;
        }

        return status;
    }

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc >
    SIXTRL_INLINE SIXTRL_CXX_NAMESPACE::arch_status_t
    Map_remove_value_from_ordered_vec( std::map< Key, std::vector< T, VecAlloc >,
            Cmp, Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        T const& SIXTRL_RESTRICT_REF value,
        bool remove_entry_if_ordered_vec_is_empty )
    {
        namespace  st = SIXTRL_CXX_NAMESPACE;
        st::arch_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        auto it = map.find( key );

        if( it != map.end() )
        {
            SIXTRL_ASSERT( std::is_sorted( it->second.begin(),
                it->second.end() ) );

            auto val_it = std::lower_bound( it->second.begin(),
                it->second.end(), value );

            if( val_it != it->second.end() )
            {
                it->second.erase( val_it );

                if( ( remove_entry_if_ordered_vec_is_empty ) &&
                    ( it->second.empty() ) )
                {
                    map.erase( it );
                }

                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc >
    SIXTRL_CXX_NAMESPACE::arch_status_t Map_ordered_vec_insert_value(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc >&
            SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        T const& SIXTRL_RESTRICT_REF value,
        T const& SIXTRL_RESTRICT_REF save_value_to_cmp_against,
        bool const keep_ordered )
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        auto it = map.find( key );
        return ( it != map.end() )
            ? st::Vector_add_sorted(
                it->second, value, save_value_to_cmp_against, keep_ordered )
            : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc >
    SIXTRL_INLINE T const* Map_ordered_vec_begin(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc >
            const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key )
    {
        T const* ptr = nullptr;

        auto it = map.find( key );
        if( ( it != map.end() ) && ( !it->second.empty() ) )
        {
            ptr = it->second.data();
        }

        return ptr;
    }

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc >
    SIXTRL_INLINE T const* Map_ordered_vec_end( std::map< Key, std::vector< T,
            VecAlloc >, Cmp, Alloc > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key )
    {
        T const* ptr = nullptr;

        auto it = map.find( key );
        if( ( it != map.end() ) && ( !it->second.empty() ) )
        {
            ptr = it->second.data();
            SIXTRL_ASSERT( ptr != nullptr );
            std::advance( ptr, it->second.size() );
        }

        return ptr;
    }

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc >
    SIXTRL_INLINE T* Map_ordered_vec_begin( std::map< Key, std::vector< T,
            VecAlloc >, Cmp, Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key )
    {
        using vec_t = std::vector< T, VecAlloc >;
        using map_t = std::map< Key, vec_t, Cmp, Alloc >;

        return const_cast< T* >( SIXTRL_CXX_NAMESPACE::Map_ordered_vec_begin(
            static_cast< map_t const& >( map ), key ) );
    }

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc >
    SIXTRL_INLINE T* Map_ordered_vec_end( std::map< Key, std::vector< T,
            VecAlloc >, Cmp, Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key )
    {
        using vec_t = std::vector< T, VecAlloc >;
        using map_t = std::map< Key, vec_t, Cmp, Alloc >;

        return const_cast< T* >( SIXTRL_CXX_NAMESPACE::Map_ordered_vec_end(
            static_cast< map_t const& >( map ), key ) );
    }

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc>
    SIXTRL_STATIC SIXTRL_HOST_FN std::size_t Map_ordered_vec_get_value_index(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc > const&
            SIXTRL_RESTRICT_REF map, Key const& SIXTRL_RESTRICT_REF key,
                T const& SIXTRL_RESTRICT_REF value )
    {
        auto it = map.find( key );
        if( ( it != map.end() ) && ( !it->second.empty() ) )
        {
            SIXTRL_ASSERT( std::is_sorted(
                it->second.begin(), it->second.end() ) );

            auto val_it = std::lower_bound(
                it->second.begin(), it->second.end(), value );

            if( val_it != it->second.end() )
            {
                return static_cast< std::size_t >( std::distance(
                    it->second.begin(), val_it ) );
            }

            return it->second.size();
        }

        return std::numeric_limits< std::size_t >::max();
    }

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc>
    SIXTRL_STATIC SIXTRL_HOST_FN T* Map_ordered_vec_get_ptr_value(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc >&
            SIXTRL_RESTRICT_REF map, Key const& SIXTRL_RESTRICT_REF key,
                T const& SIXTRL_RESTRICT_REF value )
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        return const_cast< T* >( st::Map_ordered_vec_get_ptr_const_value(
            map, key, value ) );
    }

    template< typename Key, typename T, class Cmp, class Alloc, class VecAlloc>
    SIXTRL_STATIC SIXTRL_HOST_FN T const* Map_ordered_vec_get_ptr_const_value(
        std::map< Key, std::vector< T, VecAlloc >, Cmp, Alloc > const&
            SIXTRL_RESTRICT_REF map, Key const& SIXTRL_RESTRICT_REF key,
                T const& SIXTRL_RESTRICT_REF value )
    {
        auto it = map.find( key );

        if( ( it != map.end() ) && ( !it->second.empty() ) )
        {
            SIXTRL_ASSERT( std::is_sorted(
                it->second.begin(), it->second.end() ) );

            auto val_it = std::lower_bound(
                it->second.begin(), it->second.end(), value );

            if( val_it != it->second.end() )
            {
                return std::addressof( *val_it );
            }
        }

        return nullptr;
    }

    /* --------------------------------------------------------------------- */

    template< typename Key, typename Value, class Hash,
              class KeyEqual, class Alloc >
    SIXTRL_INLINE bool Map_has_key(
        std::unordered_map< Key, Value, Hash, KeyEqual, Alloc > const&
            SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key )
    {
        return ( map.find( key ) != map.end() );
    }

    template< typename Key, typename Value, class Hash,
              class KeyEqual, class Alloc >
    SIXTRL_INLINE bool Map_has_value_for_key(
        std::unordered_map< Key, Value, Hash, KeyEqual, Alloc >  const&
            SIXTRL_RESTRICT_REF map, Key const& SIXTRL_RESTRICT_REF key,
        Value const& SIXTRL_RESTRICT_REF value )
    {
        auto it = map.find( key );
        return ( ( it != map.end() ) && ( it->second == value ) );
    }

    template< typename Key, typename Value, class Hash,
              class KeyEqual, class Alloc >
    SIXTRL_INLINE Value const& Map_get_value_or_default_for_key(
        std::unordered_map< Key, Value, Hash, KeyEqual, Alloc >  const&
            SIXTRL_RESTRICT_REF map, Key const& SIXTRL_RESTRICT_REF key,
        Value const& SIXTRL_RESTRICT_REF default_value )
    {
        auto it = map.find( key );
        return ( it != map.end() ) ? it->second : default_value;
    }

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_INLINE bool Map_ordered_vec_has_value(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash, KeyEqual,
            Alloc > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        T const& SIXTRL_RESTRICT_REF value )
    {
        bool has_value = false;

        auto it = map.find( key );

        if( it != map.end() )
        {
            SIXTRL_ASSERT( std::is_sorted(
                it->second.begin(), it->second.end() ) );

            has_value = std::binary_search( it->second.begin(),
                it->second.end(), value );
        }

        return has_value;
    }

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_INLINE SIXTRL_CXX_NAMESPACE::arch_status_t
    Map_remove_if_ordered_vec_empty( std::unordered_map< Key, std::vector< T,
            VecAlloc >, Hash, KeyEqual, Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key )
    {
        namespace st = SIXTRL_CXX_NAMESPACE;

        st::arch_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        auto it = map.find( key );

        if( it != map.end() )
        {
            SIXTRL_ASSERT( std::is_sorted(
                it->second.begin(), it->second.end() ) );

            if( it->second.empty() )
            {
                map.erase( key );
            }

            status = st::ARCH_STATUS_SUCCESS;
        }

        return status;
    }

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_INLINE bool Map_ordered_vec_empty(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash, KeyEqual,
            Alloc > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key )
    {
        auto it = map.find( key );
        return( ( it != map.end() ) && ( it->second.empty() ) );
    }

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_INLINE SIXTRL_CXX_NAMESPACE::arch_size_t Map_ordered_vec_size(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash, KeyEqual,
            Alloc > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key )
    {
        auto it = map.find( key );
        return ( it != map.end() )
            ? it->second.size() : SIXTRL_CXX_NAMESPACE::arch_size_t{ 0 };
    }


    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_INLINE SIXTRL_CXX_NAMESPACE::arch_status_t
    Map_remove_value_from_ordered_vec( std::unordered_map< Key, std::vector< T,
            VecAlloc >, Hash, KeyEqual, Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        T const& SIXTRL_RESTRICT_REF value,
        bool remove_entry_if_ordered_vec_is_empty  )
    {
        namespace  st = SIXTRL_CXX_NAMESPACE;
        st::arch_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        auto it = map.find( key );

        if( it != map.end() )
        {
            SIXTRL_ASSERT( std::is_sorted( it->second.begin(),
                it->second.end() ) );

            auto val_it = std::lower_bound( it->second.begin(),
                it->second.end(), value );

            if( val_it != it->second.end() )
            {
                it->second.erase( val_it );

                if( ( remove_entry_if_ordered_vec_is_empty ) &&
                    ( it->second.empty() ) )
                {
                    map.erase( it );
                }

                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_INLINE SIXTRL_CXX_NAMESPACE::arch_status_t
    Map_ordered_vec_insert_value(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash, KeyEqual,
            Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key,
        T const& SIXTRL_RESTRICT_REF value,
        T const& SIXTRL_RESTRICT_REF save_value_to_cmp_against,
        bool const keep_ordered )
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        using vec_t = std::vector< T, VecAlloc >;
        using map_t = std::unordered_map< Key, vec_t, Hash, KeyEqual, Alloc >;

        typename map_t::iterator it = map.find( key );
        return ( it != map.end() )
            ? st::Vector_add_sorted(
                it->second, value, save_value_to_cmp_against, keep_ordered )
            : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_INLINE T const* Map_ordered_vec_begin(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash, KeyEqual,
            Alloc > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key )
    {
        T const* ptr = nullptr;

        auto it = map.find( key );
        if( ( it != map.end() ) && ( !it->second.empty() ) )
        {
            ptr = it->second.data();
        }

        return ptr;
    }

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_INLINE T const* Map_ordered_vec_end(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash, KeyEqual,
            Alloc > const& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key )
    {
        T const* ptr = nullptr;

        auto it = map.find( key );
        if( ( it != map.end() ) && ( !it->second.empty() ) )
        {
            ptr = it->second.data();
            SIXTRL_ASSERT( ptr != nullptr );
            std::advance( ptr, it->second.size() );
        }

        return ptr;
    }

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_INLINE T* Map_ordered_vec_begin( std::unordered_map< Key, std::vector<
        T, VecAlloc >, Hash, KeyEqual, Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key )
    {
        using vec_t = std::vector< T, VecAlloc >;
        using map_t = std::unordered_map< Key, vec_t, Hash, KeyEqual, Alloc >;

        return const_cast< T* >( SIXTRL_CXX_NAMESPACE::Map_ordered_vec_begin(
            static_cast< map_t const& >( map ), key ) );
    }

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_INLINE T* Map_ordered_vec_end( std::unordered_map< Key, std::vector<
        T, VecAlloc >, Hash, KeyEqual, Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key )
    {
        using vec_t = std::vector< T, VecAlloc >;
        using map_t = std::unordered_map< Key, vec_t, Hash, KeyEqual, Alloc >;

        return const_cast< T* >( SIXTRL_CXX_NAMESPACE::Map_ordered_vec_end(
            static_cast< map_t const& >( map ), key ) );
    }

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_STATIC SIXTRL_HOST_FN std::size_t Map_ordered_vec_get_value_index(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash, KeyEqual,
            Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key, T const& SIXTRL_RESTRICT_REF value )
    {
        auto it = map.find( key );
        if( ( it != map.end() ) && ( !it->second.empty() ) )
        {
            SIXTRL_ASSERT( std::is_sorted(
                it->second.begin(), it->second.end() ) );

            auto val_it = std::lower_bound(
                it->second.begin(), it->second.end(), value );

            if( val_it != it->second.end() )
            {
                return static_cast< std::size_t >( std::distance(
                    it->second.begin(), val_it ) );
            }

            return it->second.size();
        }

        return std::numeric_limits< std::size_t >::max();
    }

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_STATIC SIXTRL_HOST_FN T* Map_ordered_vec_get_ptr_value(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash, KeyEqual,
            Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key, T const& SIXTRL_RESTRICT_REF value )
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        return const_cast< T* >( st::Map_ordered_vec_get_ptr_const_value(
            map, key, value ) );
    }

    template< typename Key, typename T, class Hash,
              class KeyEqual, class Alloc, class VecAlloc >
    SIXTRL_STATIC SIXTRL_HOST_FN T const* Map_ordered_vec_get_ptr_const_value(
        std::unordered_map< Key, std::vector< T, VecAlloc >, Hash, KeyEqual,
            Alloc >& SIXTRL_RESTRICT_REF map,
        Key const& SIXTRL_RESTRICT_REF key, T const& SIXTRL_RESTRICT_REF value )
    {
        auto it = map.find( key );
        if( ( it != map.end() ) && ( !it->second.empty() ) )
        {
            SIXTRL_ASSERT( std::is_sorted(
                it->second.begin(), it->second.end() ) );

            auto val_it = std::lower_bound(
                it->second.begin(), it->second.end(), value );

            if( val_it != it->second.end() )
            {
                return std::addressof( *val_it );
            }
        }

        return nullptr;
    }
}

#endif /* c++, Host */

#endif /* SIXTRACKLIB_COMMON_INTERNAL_STL_BUFFER_HELPER_CXX_HPP__ */

/* end: sixtracklib/common/internal/stl_buffer_helper.hpp */
