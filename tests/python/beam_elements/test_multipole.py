#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cobjects import CBuffer
import numpy as np
from scipy.special import factorial
import sixtracklib as st
import pysixtrack as pyst

if __name__ == '__main__':
    num_tests = 64
    pyst_line = []
    in_knl_data = []
    in_ksl_data = []
    in_bal_data = []

    for test_idx in range(0, num_tests):
        order = np.random.randint(0, 10)
        in_knl = np.random.uniform(0.0, 100.0, order + 1)
        in_ksl = np.random.uniform(0.0, 100.0, order + 1)

        bal_length = 2 * order + 2
        in_bal = np.zeros(bal_length)
        for ii in range(0, len(in_ksl)):
            in_bal[2 * ii] = in_knl[ii] / factorial(ii, exact=True)
            in_bal[2 * ii + 1] = in_ksl[ii] / factorial(ii, exact=True)

        pyst_line.append(pyst.elements.Multipole(knl=in_knl, ksl=in_ksl))
        in_knl_data.append(in_knl)
        in_ksl_data.append(in_ksl)
        in_bal_data.append(in_bal)

    temp = pyst.Line(pyst_line)
    st_line = st.Elements()
    st_line.append_line(temp)

    assert st_line.cbuffer.n_objects == num_tests
    assert len(in_knl_data) == num_tests
    assert len(in_ksl_data) == num_tests
    assert len(in_bal_data) == num_tests

    for ii in range(0, num_tests):
        assert pyst_line[ii].order == st_line.get(ii).order
        assert np.allclose(in_knl_data[ii], st_line.get(ii).knl, rtol=1e-15)
        assert np.allclose(in_ksl_data[ii], st_line.get(ii).ksl, rtol=1e-15)
        assert np.allclose(in_bal_data[ii], st_line.get(ii).bal, rtol=1e-15)

    for ii in range(0, num_tests):
        order = pyst_line[ii].order
        for jj in range(0, order + 1):
            new_knl_value = np.random.uniform(0.0, 100.0, 1)
            new_ksl_value = np.random.uniform(0.0, 100.0, 1)
            pyst_line[ii].knl[jj] = new_knl_value
            in_knl_data[ii][jj] = new_knl_value
            st_line.get(ii).set_knl(new_knl_value, jj)

            pyst_line[ii].ksl[jj] = new_ksl_value
            in_ksl_data[ii][jj] = new_ksl_value
            st_line.get(ii).set_ksl(new_ksl_value, jj)

            in_bal_data[ii][2 * jj] = new_knl_value / factorial(jj, exact=True)
            in_bal_data[ii][2 * jj + 1] = new_ksl_value / \
                factorial(jj, exact=True)

    assert st_line.cbuffer.n_objects == num_tests
    assert len(in_knl_data) == num_tests
    assert len(in_ksl_data) == num_tests
    assert len(in_bal_data) == num_tests

    for ii in range(0, num_tests):
        assert pyst_line[ii].order == st_line.get(ii).order
        assert np.allclose(in_knl_data[ii], st_line.get(ii).knl, rtol=1e-15)
        assert np.allclose(in_ksl_data[ii], st_line.get(ii).ksl, rtol=1e-15)
        assert np.allclose(in_bal_data[ii], st_line.get(ii).bal, rtol=1e-15)
