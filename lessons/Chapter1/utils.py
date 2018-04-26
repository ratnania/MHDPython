# coding: utf-8
import numpy as np
from spl.linalg.stencil import Vector, Matrix

def assemble_matrix_1d(V, kernel, M=None):

    # ... sizes
    [s1] = V.vector_space.starts
    [e1] = V.vector_space.ends
    [p1] = V.vector_space.pads

    k1 = V.quad_order
    spans_1 = V.spans
    basis_1 = V.basis
    weights_1 = V.weights
    # ...

    # ... data structure if not given
    if M is None:
        M = Matrix(V.vector_space, V.vector_space)
    # ...

    # ... element matrix
    mat = np.zeros((p1+1,2*p1+1), order='F')
    # ...

    # ... build matrices
    for ie1 in range(s1, e1+1-p1):
        i_span_1 = spans_1[ie1]

        bs = basis_1[:, :, :, ie1]
        w = weights_1[:, ie1]
        kernel(p1, k1, bs, w, mat)
        s1 = i_span_1 - p1 - 1
        M._data[s1:s1+p1+1,:] += mat[:,:]
    # ...

    return M
