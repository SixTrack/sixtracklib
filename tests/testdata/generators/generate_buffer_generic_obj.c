#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <stdio.h>
    #include <limits.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/sixtracklib.h"
    #include "sixtracklib/testlib.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

int main()
{
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(object_type_id_t)    type_id_t;
    typedef NS(GenericObj)          gen_obj_t;

    NS(Buffer)* buffer = NS(Buffer_new)( ( buf_size_t )( 1u << 20u ) );


    type_id_t  const obj1_type_id      = ( type_id_t )1;
    buf_size_t const obj1_num_d_values = 4;
    buf_size_t const obj1_num_e_values = 4;
    int32_t    const obj1_a_value      = 20;
    double     const obj1_b_value      = 21.0;
    double     const obj1_c_value[]    = { 22.0, 23.0, 24.0, 25.0 };
    uint8_t          obj1_d_value[]    = { 26, 27, 28, 29 };
    double           obj1_e_value[]    = { 30.0, 31.0, 32.0, 33.0 };

    gen_obj_t* ptr_obj1 = NS(GenericObj_add)(
        buffer, obj1_type_id, obj1_num_d_values, obj1_num_e_values,
        obj1_a_value, obj1_b_value, &obj1_c_value[ 0 ],
        &obj1_d_value[ 0 ], &obj1_e_value[ 0 ] );

    type_id_t  const obj2_type_id      = ( type_id_t )2;
    buf_size_t const obj2_num_d_values = 4;
    buf_size_t const obj2_num_e_values = 4;
    int32_t    const obj2_a_value      = 40;
    double     const obj2_b_value      = 41.0;
    double     const obj2_c_value[]    = { 42.0, 43.0, 44.0, 45.0 };
    uint8_t          obj2_d_value[]    = { 46, 47, 48, 49 };
    double           obj2_e_value[]    = { 50.0, 51.0, 52.0, 53.0 };

    gen_obj_t* ptr_obj2 = NS(GenericObj_add)(
        buffer, obj2_type_id, obj2_num_d_values, obj2_num_e_values,
        obj2_a_value, obj2_b_value,
        &obj2_c_value[ 0 ], &obj2_d_value[ 0 ], &obj2_e_value[ 0 ] );

    type_id_t  const obj3_type_id      = ( type_id_t )3;
    buf_size_t const obj3_num_d_values = 4;
    buf_size_t const obj3_num_e_values = 4;
    int32_t    const obj3_a_value      = 60;
    double     const obj3_b_value      = 61.0;
    double     const obj3_c_value[]    = { 62.0, 63.0, 64.0, 65.0 };
    uint8_t          obj3_d_value[]    = { 66, 67, 68, 69 };
    double           obj3_e_value[]    = { 60.0, 61.0, 62.0, 63.0 };

    gen_obj_t* ptr_obj3 = NS(GenericObj_add)(
        buffer, obj3_type_id, obj3_num_d_values, obj3_num_e_values,
        obj3_a_value, obj3_b_value,
        &obj3_c_value[ 0 ], &obj3_d_value[ 0 ], &obj3_e_value[ 0 ] );

    type_id_t  const obj4_type_id   = ( type_id_t )4;
    buf_size_t const obj4_num_d_values = 6;
    buf_size_t const obj4_num_e_values = 6;
    int32_t    const obj4_a_value   = 80;
    double     const obj4_b_value   = 81.0;
    double     const obj4_c_value[] = { 82.0, 83.0, 84.0, 85.0 };
    uint8_t          obj4_d_value[] = { 86, 87, 88, 89, 90, 91 };
    double           obj4_e_value[] = { 92.0, 93.0, 94.0, 95.0, 96.0, 97.0 };

    gen_obj_t* ptr_obj4 = NS(GenericObj_add)(
        buffer, obj4_type_id, obj4_num_d_values, obj4_num_e_values,
        obj4_a_value, obj4_b_value,
        &obj4_c_value[ 0 ], &obj4_d_value[ 0 ], &obj4_e_value[ 0 ] );

    type_id_t  const obj5_type_id   = ( type_id_t )4;
    buf_size_t const obj5_num_d_values = 100;
    buf_size_t const obj5_num_e_values = 256;

    gen_obj_t* ptr_obj5 = NS(GenericObj_new)(
        buffer, obj5_type_id, obj5_num_d_values, obj5_num_e_values );

    FILE* fp = fopen( NS(PATH_TO_TEST_GENERIC_OBJ_BUFFER_DATA), "wb" );

    if( fp != SIXTRL_NULLPTR )
    {
        buf_size_t const cnt = fwrite( ( unsigned char const* )(
            uintptr_t )NS(Buffer_get_data_begin_addr)( buffer ),
            NS(Buffer_get_size)( buffer ), ( buf_size_t )1u, fp );

        SIXTRL_ASSERT( cnt == ( buf_size_t )1u );
        fclose( fp );
        fp = SIXTRL_NULLPTR;
    }

    NS(Buffer_delete)( buffer );
    return 0;
}

/* end: tests/testdata/generators/generate_buffer_generic_obj.c */
