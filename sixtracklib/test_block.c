#include <stdio.h>
#include "block.h"
#include "block_handling.c"


int main(int argc, char **argv) {
    CLGLOBAL block_t *block;
    block = block_initialize(2);
    uint64_t offsets[3];
    uint64_t nel = 0;
    double knl[3] = { 1.0, 3.0, 5.0};
    double ksl[3] = { 2.0, 4.0, 6.0};
    block_add_multipole(block, offsets, nel++, knl, 3, ksl, 3, 0, 0, 0);
    block_add_drift(block, offsets, nel++, 56.0);
    block_add_drift(block, offsets, nel++, 5.0);
    block_add_block(block, offsets, nel);
    for(int i = 0; i < block->last; i++) {
        printf("%lu, ", block->data[i].u64);
    }
    printf("\n");
    block_clean(block);
}
