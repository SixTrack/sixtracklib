#include "sixtracklib/common/block_info.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/namespace_begin.h"
#include "sixtracklib/common/impl/block_info_impl.h"


extern int NS(Block_write_to_binary_file)( FILE* fp,
    NS(BlockType) const type_id,
    NS(block_num_elements_t) const num_elements,
    NS(block_size_t) const num_attributes,
    void** SIXTRL_RESTRICT ptr_attr_begins,
    const NS(block_size_t) *const SIXTRL_RESTRICT attr_sizes, 
    const NS(block_size_t) *const SIXTRL_RESTRICT attr_counts );


extern int NS(Block_peak_at_next_block_in_binary_file)( FILE* fp, 
    SIXTRL_UINT64_T* SIXTRL_RESTRICT binary_length,
    SIXTRL_INT64_T*  SIXTRL_RESTRICT success_flag,
    NS(BlockType)*   SIXTRL_RESTRICT type_id,
    NS(block_num_elements_t)* SIXTRL_RESTRICT num_elements,
    NS(block_size_t)* SIXTRL_RESTRICT num_attributes, 
    NS(block_size_t)* SIXTRL_RESTRICT attr_sizes, 
    NS(block_size_t)* SIXTRL_RESTRICT attr_counts, 
    NS(block_size_t) const max_num_attributes );


extern int NS(Block_read_structure_from_binary_file)( FILE* fp,
    SIXTRL_UINT64_T* SIXTRL_RESTRICT binary_length,
    SIXTRL_INT64_T*  SIXTRL_RESTRICT success_flag,    
    NS(BlockType)* SIXTRL_RESTRICT type_id,
    NS(block_num_elements_t)* SIXTRL_RESTRICT num_elements,
    NS(block_size_t)* SIXTRL_RESTRICT num_attributes,
    void** SIXTRL_RESTRICT ptr_attr_begins,
    NS(block_size_t)* SIXTRL_RESTRICT attr_sizes,
    NS(block_size_t)* SIXTRL_RESTRICT attr_counts );

/* ------------------------------------------------------------------------- */

int NS(Block_write_to_binary_file)( 
    FILE* fp, 
    NS(BlockType) const type_id, 
    NS(block_num_elements_t) const num_elements, 
    NS(block_size_t) const num_attributes, 
    void** SIXTRL_RESTRICT ptr_attr_begins,
    const NS(block_size_t) *const SIXTRL_RESTRICT attr_sizes, 
    const NS(block_size_t) *const SIXTRL_RESTRICT attr_counts )
{
    int success = -1;
    
    SIXTRL_STATIC SIXTRL_SIZE_T   const ONE       = ( SIXTRL_SIZE_T )1u;
    SIXTRL_STATIC SIXTRL_SIZE_T   const U64_SIZE  = sizeof( SIXTRL_UINT64_T );
    SIXTRL_STATIC SIXTRL_SIZE_T   const I64_SIZE  = sizeof( SIXTRL_INT64_T );
    SIXTRL_STATIC SIXTRL_SIZE_T   const ZERO      = ( NS(block_size_t ) )0u;    
    SIXTRL_STATIC SIXTRL_UINT64_T const U64_ZERO  = ( SIXTRL_UINT64_T )0u;
    SIXTRL_STATIC SIXTRL_INT64_T  const NOT_VALID = ( SIXTRL_INT64_T )-1;
    
    if( ( fp != 0 ) && ( type_id != NS(BLOCK_TYPE_INVALID ) ) && 
        ( num_elements   > ( NS(block_num_elements_t) )0u ) && 
        ( num_attributes > ZERO ) && ( ptr_attr_begins != 0 ) && 
        ( attr_sizes != 0 ) && ( attr_counts != 0 ) )
    {
        long int const begin_fpos = ftell( fp );
            
        SIXTRL_UINT64_T binary_length = U64_ZERO;
        SIXTRL_UINT64_T type_id_num   = ( SIXTRL_UINT64_T )type_id;
        SIXTRL_INT64_T  success_flag  = NOT_VALID;
        SIXTRL_UINT64_T num_elem_num  = ( SIXTRL_UINT64_T )num_elements;
        SIXTRL_UINT64_T num_attr_num  = ( SIXTRL_UINT64_T )num_attributes;
        
        if( ( begin_fpos != -1 ) &&
            ( ONE == fwrite( &binary_length, U64_SIZE, ONE, fp ) ) &&
            ( ONE == fwrite( &success_flag,  I64_SIZE, ONE, fp ) ) &&
            ( ONE == fwrite( &type_id_num,   U64_SIZE, ONE, fp ) ) &&
            ( ONE == fwrite( &num_elem_num,  U64_SIZE, ONE, fp ) ) &&
            ( ONE == fwrite( &num_attr_num,  U64_SIZE, ONE, fp ) ) )
        {            
            SIXTRL_UINT64_T  offsets_vector[ 64 ];
            SIXTRL_UINT64_T* offsets = &offsets_vector[ 0 ];
            
            success = 0;
                        
            if( num_attributes > 64 )
            {
                offsets = ( SIXTRL_UINT64_T* )malloc( 
                    sizeof( SIXTRL_UINT64_T ) * num_attributes );
                
                if( offsets == 0 ) success = -1;
            }
            
            if( success == 0 )
            {
                NS(block_size_t) jj = ZERO;
                
                for( ; jj < num_attributes ; ++jj )
                {
                    success = -1;
                    
                    if( ( attr_sizes[ jj ]  > ZERO ) && 
                        ( attr_counts[ jj ] > ZERO ) )
                    {
                        SIXTRL_UINT64_T const attr_size  = attr_sizes[ jj ];
                        SIXTRL_UINT64_T const attr_count = attr_counts[ jj ];
                        
                        if( ( ONE == fwrite( &attr_size, U64_SIZE, ONE, fp ) ) &&
                            ( ONE == fwrite( &attr_count, U64_SIZE, ONE, fp ) ) )
                        {
                            offsets[ jj ] = attr_sizes[ jj ] * attr_counts[ jj ];
                            success = 0;
                        }
                    }
                    
                    if( success != 0 ) break;
                }
            }
            
            if( success == 0 )
            {
                NS(block_num_elements_t) ii = 0;
                
                for( ; ii < num_elements ; ++ii )
                {
                    NS(block_size_t) jj = ZERO;
                    
                    for( jj = ZERO ; jj < num_attributes ; ++jj )
                    {
                        unsigned char* attr_begin = 
                            ( unsigned char* )ptr_attr_begins[ jj ];
                        
                        SIXTRL_ASSERT( attr_begin != 0 );
                            
                        if( ( attr_counts[ jj ] == ZERO ) ||
                            ( attr_counts[ jj ] != fwrite( 
                                attr_begin + ( offsets[ jj ] * ii ), 
                                attr_sizes[ jj ], attr_counts[ jj ], fp ) ) )
                        {
                            success = -1;
                            break;
                        }
                    }
                    
                    if( success != 0 ) break;
                }
            }
        
            if( begin_fpos != -1 )
            {
                long int const end_fpos = ftell( fp );
                SIXTRL_ASSERT( ( end_fpos >= begin_fpos ) && ( fp != 0 ) );
                
                binary_length = ( SIXTRL_UINT64_T )( end_fpos - begin_fpos );
                success_flag  = ( SIXTRL_INT64_T  )success;
                
                if( ( fseek( fp, begin_fpos, SEEK_SET ) != 0 ) ||
                    ( fwrite( &binary_length, U64_SIZE, ONE, fp ) != ONE ) ||
                    ( fwrite( &success_flag,  U64_SIZE, ONE, fp ) != ONE ) ||
                    ( fseek( fp, end_fpos, SEEK_SET ) != 0 ) )
                {
                    fseek( fp, begin_fpos, SEEK_SET );
                    success = -1;
                }
            }
            
            if( offsets != &offsets_vector[ 0 ] )
            {
                free( offsets );
                offsets = 0;
            }
        }
    }
    
    return success;
}

int NS(Block_peak_at_next_block_in_binary_file)( FILE* fp, 
    SIXTRL_UINT64_T* SIXTRL_RESTRICT binary_length,
    SIXTRL_INT64_T*  SIXTRL_RESTRICT success_flag,
    NS(BlockType)*   SIXTRL_RESTRICT type_id,
    NS(block_num_elements_t)* SIXTRL_RESTRICT num_elements,
    NS(block_size_t)* SIXTRL_RESTRICT num_attributes, 
    NS(block_size_t)* SIXTRL_RESTRICT attr_sizes, 
    NS(block_size_t)* SIXTRL_RESTRICT attr_counts, 
    NS(block_size_t) const max_num_attributes )
{
    int success = -1;
    
    SIXTRL_STATIC SIXTRL_SIZE_T   const ONE       = ( SIXTRL_SIZE_T )1u;
    SIXTRL_STATIC SIXTRL_SIZE_T   const U64_SIZE  = sizeof( SIXTRL_UINT64_T );
    SIXTRL_STATIC SIXTRL_SIZE_T   const I64_SIZE  = sizeof( SIXTRL_INT64_T );
    SIXTRL_STATIC SIXTRL_SIZE_T   const ZERO      = ( NS(block_size_t ) )0u;    
    SIXTRL_STATIC SIXTRL_UINT64_T const U64_ZERO  = ( SIXTRL_UINT64_T )0u;
    SIXTRL_STATIC SIXTRL_INT64_T  const NOT_VALID = ( SIXTRL_INT64_T )-1;
    
    if( ( fp != 0 ) && ( max_num_attributes > ZERO ) && ( type_id != 0 ) &&
        ( num_elements != 0 ) && ( num_attributes != 0 ) && 
        ( attr_sizes != 0 ) && ( attr_counts != 0 ) )
    {
        long int const begin_fpos = ftell( fp );
        
        SIXTRL_UINT64_T temp_binary_length = ZERO;
        SIXTRL_UINT64_T temp_type_id_num   = ZERO;
        SIXTRL_UINT64_T temp_num_elem_num  = ZERO;
        SIXTRL_UINT64_T temp_num_attr_num  = ZERO;
        SIXTRL_INT64_T  temp_success_flag  = NOT_VALID;
        
        if( ( begin_fpos != -1 ) &&
            ( ONE == fread( &temp_binary_length, U64_SIZE, ONE, fp ) ) &&
            ( ONE == fread( &temp_success_flag,  I64_SIZE, ONE, fp ) ) &&
            ( ONE == fread( &temp_type_id_num,   U64_SIZE, ONE, fp ) ) &&
            ( ONE == fread( &temp_num_elem_num,  U64_SIZE, ONE, fp ) ) &&
            ( ONE == fread( &temp_num_attr_num,  U64_SIZE, ONE, fp ) ) &&
            ( temp_num_attr_num <= max_num_attributes ) )
        {
            SIXTRL_UINT64_T ii = U64_ZERO;
            
            for( ; ii < temp_num_attr_num ; ++ii )
            {
                attr_sizes[ ii ]  = U64_ZERO;
                attr_counts[ ii ] = U64_ZERO;
            }
            
            success = 0;
            
            for( ii = U64_ZERO ; ii < temp_num_attr_num ; ++ii )
            {
                if( !( ( ONE == fread( &attr_sizes[ ii ],  
                            U64_SIZE, ONE, fp ) ) &&
                       ( ONE == fread( &attr_counts[ ii ], 
                            U64_SIZE, ONE, fp ) ) &&
                       ( attr_sizes[ ii ]  > U64_ZERO ) && 
                       ( attr_counts[ ii ] > U64_ZERO ) ) )
                {
                    success = -1;
                    break;                                
                }                
            }
        }
        
        if( success == 0 )
        {
            *type_id = NS(BlockType_from_number)( temp_type_id_num );
            *num_elements   = temp_num_elem_num;
            *num_attributes = temp_num_attr_num;
            
            if( success_flag != 0 )
            {
                *success_flag = temp_success_flag;
            }
            
            if( binary_length != 0 )
            {
                *binary_length = temp_binary_length;
            }
        }
        else
        {
            *type_id = NS(BLOCK_TYPE_INVALID);
            *num_elements   = U64_ZERO;
            *num_attributes = U64_ZERO;
        }
        
        if( begin_fpos != -1 )
        {
            fseek( fp, begin_fpos, SEEK_SET );            
        }
    }
    
    return success;
}

int NS(Block_read_structure_from_file)( FILE* fp,
    SIXTRL_UINT64_T* SIXTRL_RESTRICT binary_length,
    SIXTRL_INT64_T*  SIXTRL_RESTRICT success_flag,    
    NS(BlockType)* SIXTRL_RESTRICT type_id,
    NS(block_num_elements_t)* SIXTRL_RESTRICT num_elements,
    NS(block_size_t)* SIXTRL_RESTRICT num_attributes,
    void** SIXTRL_RESTRICT ptr_attr_begins,
    NS(block_size_t)* SIXTRL_RESTRICT attr_sizes,
    NS(block_size_t)* SIXTRL_RESTRICT attr_counts )
{
    int success = -1;
    
    SIXTRL_STATIC SIXTRL_SIZE_T   const ONE       = ( SIXTRL_SIZE_T )1u;
    SIXTRL_STATIC SIXTRL_SIZE_T   const U64_SIZE  = sizeof( SIXTRL_UINT64_T );
    SIXTRL_STATIC SIXTRL_SIZE_T   const I64_SIZE  = sizeof( SIXTRL_INT64_T );
    SIXTRL_STATIC SIXTRL_SIZE_T   const ZERO      = ( NS(block_size_t ) )0u;    
    SIXTRL_STATIC SIXTRL_UINT64_T const U64_ZERO  = ( SIXTRL_UINT64_T )0u;
    SIXTRL_STATIC SIXTRL_INT64_T  const NOT_VALID = ( SIXTRL_INT64_T )-1;
    
    if( ( fp != 0 ) && ( type_id != 0 ) &&
        ( num_elements != 0 ) && ( num_attributes != 0 ) && 
        ( attr_sizes != 0 ) && ( attr_counts != 0 ) )
    {
        long int const begin_fpos = ftell( fp );
        
        SIXTRL_UINT64_T temp_binary_length = ZERO;
        SIXTRL_UINT64_T temp_type_id_num   = ZERO;
        SIXTRL_UINT64_T temp_num_elem_num  = ZERO;
        SIXTRL_UINT64_T temp_num_attr_num  = ZERO;
        SIXTRL_INT64_T  temp_success_flag  = NOT_VALID;
        
        if( ( begin_fpos != -1 ) &&
            ( ONE == fread( &temp_binary_length, U64_SIZE, ONE, fp ) ) &&
            ( ONE == fread( &temp_success_flag,  I64_SIZE, ONE, fp ) ) &&
            ( temp_success_flag == ( SIXTRL_INT64_T )0 ) &&
            ( ONE == fread( &temp_type_id_num,   U64_SIZE, ONE, fp ) ) &&
            ( temp_type_id_num == 
                NS(BlockType_to_number)( NS(BLOCK_TYPE_INVALID) ) ) &&
            ( ONE == fread( &temp_num_elem_num,  U64_SIZE, ONE, fp ) ) &&
            ( ONE == fread( &temp_num_attr_num,  U64_SIZE, ONE, fp ) ) )
        {
            SIXTRL_UINT64_T  offsets_vector[ 64 ];
            SIXTRL_UINT64_T* offsets = &offsets_vector[ 0 ];        
            
            SIXTRL_UINT64_T ii = U64_ZERO;
            
            if( *num_attributes > 64 )
            {
                offsets = ( SIXTRL_UINT64_T* )malloc( 
                    sizeof( SIXTRL_UINT64_T ) * ( *num_attributes ) );
                
                if( offsets == 0 ) return success;
            }
            
            for( ; ii < temp_num_attr_num ; ++ii )
            {
                attr_sizes[ ii ]  = U64_ZERO;
                attr_counts[ ii ] = U64_ZERO;
            }
            
            success = 0;
            
            for( ii = U64_ZERO ; ii < temp_num_attr_num ; ++ii )
            {
                if( !( ( ONE == fread( &attr_sizes[ ii ],  
                            U64_SIZE, ONE, fp ) ) &&
                       ( ONE == fread( &attr_counts[ ii ], 
                            U64_SIZE, ONE, fp ) ) &&
                       ( attr_sizes[ ii ]  > U64_ZERO ) && 
                       ( attr_counts[ ii ] > U64_ZERO ) ) )
                {
                    success = -1;
                    break;                                
                }
            }
            
            if( success == 0 )
            {
                for( ii = U64_ZERO ; ii < temp_num_attr_num ; ++ii )
                {
                    NS(block_size_t) jj = ZERO;
                    NS(block_size_t) const num_attrs = *num_attributes;
                    
                    for( jj = ZERO ; jj < num_attrs ; ++jj )
                    {
                        unsigned char* attr_begin = 
                            ( unsigned char* )ptr_attr_begins[ jj ];
                        
                        SIXTRL_ASSERT( attr_begin != 0 );
                            
                        if( ( attr_counts[ jj ] == ZERO ) ||
                            ( attr_counts[ jj ] != fwrite( 
                                attr_begin + ( offsets[ jj ] * ii ), 
                                attr_sizes[ jj ], attr_counts[ jj ], fp ) ) )
                        {
                            success = -1;
                            break;
                        }
                    }
                    
                    if( success != 0 ) break;
                }
            }
            
            if( offsets != &offsets_vector[ 0 ] )
            {
                free( offsets );
                offsets = 0;
            }
        }
        
        if( success == 0 )
        {
            *type_id = NS(BlockType_from_number)( temp_type_id_num );
            *num_elements   = temp_num_elem_num;
            *num_attributes = temp_num_attr_num;
            
            if( success_flag != 0 )
            {
                *success_flag = temp_success_flag;
            }
            
            if( binary_length != 0 )
            {
                *binary_length = temp_binary_length;
            }
        }
        else
        {
            *type_id = NS(BLOCK_TYPE_INVALID);
            *num_elements   = U64_ZERO;
            *num_attributes = U64_ZERO;
        }
        
        if( begin_fpos != -1 )
        {
            fseek( fp, begin_fpos, SEEK_SET );            
        }
    }
    
    return success;
}


int NS(Block_read_structure_from_binary_file)( FILE* fp,
    SIXTRL_UINT64_T* SIXTRL_RESTRICT binary_length,
    SIXTRL_INT64_T*  SIXTRL_RESTRICT success_flag,    
    NS(BlockType)* SIXTRL_RESTRICT type_id,
    NS(block_num_elements_t)* SIXTRL_RESTRICT num_elements,
    NS(block_size_t)* SIXTRL_RESTRICT num_attributes,
    void** SIXTRL_RESTRICT ptr_attr_begins,
    NS(block_size_t)* SIXTRL_RESTRICT attr_sizes,
    NS(block_size_t)* SIXTRL_RESTRICT attr_counts )
{
    int success = -1;
    
    SIXTRL_STATIC SIXTRL_SIZE_T   const ONE       = ( SIXTRL_SIZE_T )1u;
    SIXTRL_STATIC SIXTRL_SIZE_T   const U64_SIZE  = sizeof( SIXTRL_UINT64_T );
    SIXTRL_STATIC SIXTRL_SIZE_T   const I64_SIZE  = sizeof( SIXTRL_INT64_T );
    SIXTRL_STATIC SIXTRL_SIZE_T   const ZERO      = ( NS(block_size_t ) )0u;    
    SIXTRL_STATIC SIXTRL_UINT64_T const U64_ZERO  = ( SIXTRL_UINT64_T )0u;
    SIXTRL_STATIC SIXTRL_INT64_T  const NOT_VALID = ( SIXTRL_INT64_T )-1;
    
    if( ( fp != 0 ) && ( type_id != 0 ) && 
        ( num_elements != 0 ) && ( num_attributes != 0 ) && 
        ( attr_sizes != 0 ) && ( attr_counts != 0 ) )
    {
        long int const begin_fpos = ftell( fp );
            
        SIXTRL_UINT64_T temp_type_id_num   = ( SIXTRL_UINT64_T 
            )NS(BlockType_to_number)( NS(BLOCK_TYPE_INVALID) );
        
        SIXTRL_UINT64_T temp_binary_length = U64_ZERO;
        SIXTRL_INT64_T  temp_success_flag  = NOT_VALID;
        SIXTRL_UINT64_T num_elem_num       = U64_ZERO;
        SIXTRL_UINT64_T num_attr_num       = U64_ZERO;
        
        if( ( begin_fpos != -1 ) &&
            ( ONE == fread( &temp_binary_length, U64_SIZE, ONE, fp ) ) &&
            ( ONE == fread( &temp_success_flag,  I64_SIZE, ONE, fp ) ) &&
            ( temp_success_flag == ( SIXTRL_INT64_T )0 ) &&
            ( ONE == fread( &temp_type_id_num,   U64_SIZE, ONE, fp ) ) &&
            ( temp_type_id_num == 
                NS(BlockType_to_number)( NS(BLOCK_TYPE_INVALID) ) ) &&
            ( ONE == fread( &num_elem_num,  U64_SIZE, ONE, fp ) ) &&
            ( ONE == fread( &num_attr_num,  U64_SIZE, ONE, fp ) ) )
        {            
            SIXTRL_UINT64_T  offsets_vector[ 64 ];
            SIXTRL_UINT64_T* offsets = &offsets_vector[ 0 ];
            
            success = 0;
                        
            if( num_attr_num > 64 )
            {
                offsets = ( SIXTRL_UINT64_T* )malloc( 
                    sizeof( SIXTRL_UINT64_T ) * num_attr_num );
                
                if( offsets == 0 ) success = -1;
            }
            
            if( success == 0 )
            {
                SIXTRL_UINT64_T ii = U64_ZERO;
                
                for( ; ii < num_attr_num ; ++ii )
                {
                    if( ( ONE == fread( &attr_sizes[ ii ],  
                                        U64_SIZE, ONE, fp )) &&
                        ( ONE == fread( &attr_counts[ ii ], 
                                        U64_SIZE, ONE, fp )) &&
                        ( attr_sizes[ ii ]  > U64_ZERO ) && 
                        ( attr_counts[ ii ] > U64_ZERO ) )
                    {
                        offsets[ ii ] = attr_sizes[ ii ] * attr_counts[ ii ];
                    }
                    else
                    {
                        success = -1;
                        break;                                
                    }
                }
            }
            
            if( success == 0 )
            {
                SIXTRL_UINT64_T ii = U64_ZERO;
                
                for( ; ii < num_elem_num ; ++ii )
                {
                    SIXTRL_UINT64_T jj = U64_ZERO;
                    
                    for( ; jj < num_attr_num ; ++jj )
                    {
                        unsigned char* attr_begin = 
                            ( unsigned char* )ptr_attr_begins[ jj ];
                        
                        SIXTRL_ASSERT( attr_begin != 0 );
                            
                        if( ( attr_counts[ jj ] == ZERO ) ||
                            ( attr_counts[ jj ] != fread( 
                                attr_begin + ( offsets[ jj ] * ii ), 
                                attr_sizes[ jj ], attr_counts[ jj ], fp ) ) )
                        {
                            success = -1;
                            break;
                        }
                    }
                    
                    if( success != 0 ) break;
                }
            }
        
            if( success == 0 )
            {
                *num_elements    = num_elem_num;
                *num_attributes  = num_attr_num;
                *type_id = NS(BlockType_from_number)( temp_type_id_num );
                
                if( success_flag != 0 )
                {
                    *success_flag = temp_success_flag;                    
                }
                
                if( binary_length != 0 )
                {
                    *binary_length = temp_binary_length;
                }
            }
            else
            {
                *num_elements    = ZERO;
                *num_attributes  = ZERO;
                *type_id = NS(BLOCK_TYPE_INVALID);
                
                if( success_flag != 0 )
                {
                    *success_flag = temp_success_flag;                    
                }
                
                if( binary_length != 0 )
                {
                    *binary_length = temp_binary_length;
                }
            }
        
            if( begin_fpos != -1 )
            {
                long int const end_fpos = begin_fpos + temp_binary_length;
                fseek( fp, end_fpos, SEEK_SET );
            }
            
            if( offsets != &offsets_vector[ 0 ] )
            {
                free( offsets );
                offsets = 0;
            }
        }
    }
    
    return success;
}
