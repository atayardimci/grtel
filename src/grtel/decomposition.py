"""TODO: This class is not used and needs to be updated"""
import numpy as np
import pandas as pd

import scipy.linalg
import scipy.sparse.linalg

from hottbox.core import Tensor, TensorTKD
from hottbox.algorithms.decomposition import HOSVD, HOOI
from hottbox.utils.generation import residual_tensor

# TODO: Old incomplete code, not used
class GLTD:
    """THE GLTD CLASS.

    Implemented for third order tensor decomposition.
    Last mode should be the regularized mode.
    """
    def __init__(self, S=None, beta=0, max_iter=50, epsilon=1e-2, tol=1e-4, verbose=False):
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.tol = tol
        self.verbose = verbose

    def calculate_L(self, S):
        if S == None:
            return None
        else:
            D = np.identity(len(S))
            for i in range(len(S)):
                total = 0
                for j in range(len(S)):
                    total += S[i,j]
                D[i,i] = total
            return D - S

    def decompose(self, tensor, rank):
        if not isinstance(tensor, Tensor):
            raise TypeError("Parameter `tensor` should be an object of `Tensor` class!")
        if not isinstance(rank, tuple):
            raise TypeError("Parameter `rank` should be passed as a tuple!")
        if tensor.order != len(rank):
            raise ValueError("Parameter `rank` should be a tuple of same length as the order of a tensor:\n"
                             "{} != {} (tensor.order != len(rank))".format(tensor.order, len(rank)))

        cost = []
        converged = False

        tensor_tkd = None
        fmat_gltd = self._init_fmat(tensor, rank)
        norm = tensor.frob_norm
        for _ in range(self.max_iter):
            # Update factor matrices
            # step 1
            V, W = fmat_gltd[1], fmat_gltd[2]
            VVT = np.dot(V, V.T)
            WWT = np.dot(W, W.T)

            A = tensor.mode_n_product(VVT, mode=1, inplace=False)
            B = tensor.mode_n_product(WWT, mode=2, inplace=False)

            n = tensor.shape[0]
            F = np.zeros((n,n))

            for i in range(n):
                for j in range(n):
                    F[i,j] = np.trace(np.dot(A[i,:,:].T, B[j,:,:]))

            U, _, _ = scipy.linalg.svd(F)
            U = U[:,:rank[0]]
            fmat_gltd[0] = U

            # step 2
            U, W = fmat_gltd[0], fmat_gltd[2]
            UUT = np.dot(U, U.T)
            WWT = np.dot(W, W.T)

            A = tensor.mode_n_product(UUT, mode=0, inplace=False)
            B = tensor.mode_n_product(WWT, mode=2, inplace=False)

            n = tensor.shape[1]
            G = np.zeros((n,n))

            for i in range(n):
                for j in range(n):
                    G[i,j] = np.trace(np.dot(A[:,i,:].T, B[:,j,:]))

            V, _, _ = scipy.linalg.svd(G)
            V = V[:,:rank[1]]
            fmat_gltd[1] = V

            # step 3
            U, V = fmat_gltd[0], fmat_gltd[1]
            UUT = np.dot(U, U.T)
            VVT = np.dot(V, V.T)

            A = tensor.mode_n_product(UUT, mode=0, inplace=False)
            B = tensor.mode_n_product(VVT, mode=1, inplace=False)

            n = tensor.shape[2]
            H = np.zeros((n,n))

            for i in range(n):
                for j in range(n):
                    H[i,j] = np.trace(np.dot(A[:,:,i].T, B[:,:,j]))

            # TODO: Graph-regularization here
            # Without Graph-regularization
            reg_H = H

            W, _, _ = scipy.linalg.svd(reg_H)
            W = W[:,:rank[2]]
            fmat_gltd[2] = W

            # Update core
            core = tensor.copy()
            for mode, fmat in enumerate(fmat_gltd):
                core.mode_n_product(fmat.T, mode=mode)

            # Update cost
            tensor_tkd = TensorTKD(fmat=fmat_gltd, core_values=core.data)
            residual = residual_tensor(tensor, tensor_tkd)

            # cost function that Ilia uses
            cost_ilia = abs(residual.frob_norm / norm)
            cost.append(cost_ilia)


            # Check termination conditions
            if cost[-1] <= self.epsilon:
                if self.verbose:
                    print('Relative error of approximation has reached the acceptable level: {}'.format(cost[-1]))
                break
            if len(cost) >= 2 and abs(cost[-2] - cost[-1]) <= self.tol:
                converged = True
                if self.verbose:
                    print('Converged in {} iteration(s)'.format(len(cost)))
                break

        if not converged and cost[-1] > self.epsilon:
            print('Maximum number of iterations ({}) has been reached. '
                  'Variation = {}'.format(self.max_iter, abs(cost[-2] - cost[-1])))
        return tensor_tkd

    def _init_fmat(self, tensor, rank):
        """Initialize factor matrices using HOSVD"""
        # TODO: initialize fmat with HOSVD

        # initialize fmat as identity matrices
        fmat = []
        fmat.append(np.identity(tensor.shape[0])[:,:rank[0]])
        fmat.append(np.identity(tensor.shape[1])[:,:rank[1]])
        fmat.append(np.identity(tensor.shape[2])[:,:rank[2]])

        return fmat