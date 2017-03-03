#include <stdlib.h>
#include <math.h>

CLGLOBAL block_t* block_initialize(unsigned int size) {
    CLGLOBAL block_t *block = malloc(sizeof(CLGLOBAL block_t*));
    if(!size) {
        size = 512;
    }
    CLGLOBAL value_t *data = malloc(sizeof(CLGLOBAL value_t) * size);
    if(!data) {
        return NULL;
    }
    block->size = sizeof(CLGLOBAL value_t) * size;
    block->last = 0;
    block->data = data;
    return block;
}

void block_reshape(CLGLOBAL block_t *block, unsigned int n) {
    if(block->last+(sizeof(CLGLOBAL value_t)*n) >= block->size) {
        CLGLOBAL value_t *ndata = realloc(block->data, (block->size+(sizeof(CLGLOBAL value_t)*n))*2);
        if(!ndata) {
            return;
        }
        block->size = (block->size+(sizeof(CLGLOBAL value_t)*n))*2;
        block-> data = ndata;
    }
}

void block_clean(CLGLOBAL block_t *block) {
    free(block->data);
    free(block);
}

CLGLOBAL block_t* block_add_drift(CLGLOBAL block_t *block, uint64_t *offsets, uint64_t nel, double length) {
    block_reshape(block, 2);
    offsets[nel] = block->last;
    block->data[block->last++].u64 = DriftID;
    block->data[block->last++].f64 = length;
    return block;
}

CLGLOBAL block_t* block_add_cavity(CLGLOBAL block_t *block, uint64_t *offsets, uint64_t nel, double volt, double freq, double lag) {
    block_reshape(block, 4);
    offsets[nel] = block->last;
    block->data[block->last++].u64 = CavityID;
    block->data[block->last++].f64 = volt;
    block->data[block->last++].f64 = freq;
    block->data[block->last++].f64 = lag/180.0 * M_PI;
    return block;
}

CLGLOBAL block_t* block_add_align(CLGLOBAL block_t *block, uint64_t *offsets, uint64_t nel, double tilt, double dx, double dy) {
    block_reshape(block, 5);
    offsets[nel] = block->last;
    block->data[block->last++].u64 = AlignID;
    block->data[block->last++].f64 = cos(tilt/180.0 * M_PI);
    block->data[block->last++].f64 = sin(tilt/180.0 * M_PI);
    block->data[block->last++].f64 = dx;
    block->data[block->last++].f64 = dy;
    return block;
}

CLGLOBAL block_t* block_add_multipole(CLGLOBAL block_t *block, uint64_t *offsets, uint64_t nel, double *knl, unsigned int knl_len, double *ksl, unsigned int ksl_len, double length, double hxl, double hyl) {
    double bal[(knl_len >= ksl_len ? 2*knl_len : 2*ksl_len)];
    int i = 0;
    for(; i < knl_len || i < ksl_len; i++) {
        if(i < knl_len) {
            bal[2*i] = knl[i];
        } else {
            bal[2*i] = 0;
        }
        if(i < ksl_len) {
            bal[2*i+1] = ksl[i];
        } else {
            bal[2*i+1] = 0;
        }
    }
    uint64_t order = i-1;
    for(int j = 0, fact = 1; j < i; fact *= ++j) {
        bal[2*j] /= fact;
        bal[2*j+1] /= fact;
    }
    block_reshape(block, 5+(2*i));
    offsets[nel] = block->last;
    block->data[block->last++].u64 = MultipoleID;
    block->data[block->last++].u64 = order;
    block->data[block->last++].f64 = length;
    block->data[block->last++].f64 = hxl;
    block->data[block->last++].f64 = hyl;
    for(int j = 0; j < 2*i; j++) {
        block->data[block->last++].f64 = bal[j];
    }
    return block;
}

CLGLOBAL block_t* block_add_block(CLGLOBAL block_t *block, uint64_t *offsets, uint64_t nel) {
    block_reshape(block, nel+2);
    block->data[block->last++].u64 = BlockID;
    block->data[block->last++].u64 = nel;
    for(uint64_t i = 0; i < nel; i++) {
        block->data[block->last++].u64 = offsets[i];
    }
    return block;
}

