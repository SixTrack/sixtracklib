#!/usr/bin/env python
# -*- coding: utf-8 -*-
import test_data_generation

if __name__ == '__main__':
    test_data_generation.generate_testdata(pyst_example='lhc_no_bb',
                                           pysixtrack_line_from_pickle=False)
