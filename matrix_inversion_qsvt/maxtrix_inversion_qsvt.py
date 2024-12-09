import numpy as np
import scipy.linalg

import pyqsp
from pyqsp.angle_sequence import Chebyshev, QuantumSignalProcessingPhases

from helpers import *

class MatrixInversionQSVT:
    def __init__(self, A, epsilon=1e-2, method="laurent", verbose=True):
        """
        Implements matrix inversion using Quantum Singular Value Transformation (QSVT). 
        This class constructs a quantum circuit to approximate the inverse of a given matrix 
        with given precision, leveraging polynomial approximation of 1/x.
        """
        self.verbose = verbose # Print intermediate steps is True

        self.A = A
        # TODO: alpha and kappa may need to be replaced by their respective upper bounds
        self.alpha = np.linalg.norm(A, ord=2)  # Spectral norm
        self.A_rescaled = self.A / self.alpha
        self.kappa = np.linalg.cond(A) # Condition number
        self.eigvals = sorted(np.linalg.eigvals(self.A_rescaled)) # Eigenvalues
         
        self.epsilon = epsilon # Desired presision
        self.N = max(A.shape[0], A.shape[1]) # Matrix dimension
        self.n = int(np.ceil(np.log2(self.N))) # Minimum number of quibts to encode A
        self.m = 1 # Number of ancillas for the block-encoding of A

        pg = pyqsp.poly.PolyOneOverX()
        pcoefs, scale = pg.generate(kappa=self.kappa, 
                                    epsilon=self.epsilon,
                                    return_scale=True)
        poly = Chebyshev(pcoefs) # Chebyshev decomposition of 1/x
        phases = np.array(QuantumSignalProcessingPhases(poly, signal_operator="Wx", measurement="x")) # QSP phases in the Wx convention
        
        def wx_to_reflection(phases):
            """
            Converts QSP-angles in the Wx convention into reflection convention.
            """
            phases[0] = phases[0] + (2*(len(phases)-1) - 1) * np.pi / 4
            phases[1:-1] = phases[1:-1] - np.pi / 2
            phases[-1] = phases[-1] - np.pi / 4
            return phases

        self.pcoefs = pcoefs
        self.scale = scale
        self.phases_wx = phases.copy()
        self.phases = wx_to_reflection(phases)    
        self.d = len(phases) - 1
        self.circuit = self._matrix_inversion()
        self.A_out = self.circuit[:2**self.n, :2**self.n]
        self.eigvals_out = sorted(np.linalg.eigvals(self.A_out)) # Resulting eigenvalues

        if self.verbose: 
            print("N =", self.N)
            print("kappa =", self.kappa)
            print("alpha =", self.alpha)
            print("A is Hermitian:", np.allclose(A, A.T.conj()))
            print("Re(eigenvalues):", np.round(np.real(self.eigvals), 5))
            print("Im(eigenvalues):", np.round(np.imag(self.eigvals), 5))
            print("d =", self.d)
            print("Output is Hermitian:", np.allclose(self.A_out, self.A_out.T.conj()))
            print("Re(output eigenvalues):", np.round(np.real(self.eigvals_out), 5))
            print("Im(output eigenvalues):", np.round(np.imag(self.eigvals_out), 5))

    def _block_encode_A(self):
        """
        Creates a block-encoded unitary matrix U that embeds A / alpha in its top-left block.
        """
        I = np.eye(self.A_rescaled.shape[0])
        top_right_block = scipy.linalg.sqrtm(I - self.A_rescaled @ self.A_rescaled.T.conj())
        bottom_left_block = scipy.linalg.sqrtm(I - self.A_rescaled.T.conj() @ self.A_rescaled)

        U = np.block([[self.A_rescaled, top_right_block],
                      [bottom_left_block, -1*self.A_rescaled.T.conj()]])
        U_dagger = U.T.conj()
        return U, U_dagger

    def _matrix_inversion(self):
        """
        Creates a block-encoding of A^(-1).
        """
        def P_phi(phi, m):
            P = kron_list([X] + [P0]*m) + kron_list([I]*(m+1)) - kron_list([I] + [P0]*m)
            Rz = np.array([[np.exp(-1j*phi),0],[0,np.exp(1j*phi)]])
            return P @ kron_list([Rz] + [I]*m) @ P

        U_A, U_A_dagger = self._block_encode_A() 
        if self.verbose: print("Block-encoding is Hermitian:", np.allclose(U_A, U_A_dagger))

        if self.d%2 == 1: # d is odd
            circuit = np.kron(P_phi(self.phases[0], self.m), np.eye(2**self.n))
            circuit = circuit @ np.kron(I, U_A)
            circuit = circuit @ np.kron(P_phi(self.phases[1], self.m), np.eye(2**self.n))
            for i in range((self.d-1) // 2):
                circuit = circuit @ np.kron(I, U_A_dagger)
                circuit = circuit @ np.kron(P_phi(self.phases[2*i + 2], self.m), np.eye(2**self.n))
                circuit = circuit @ np.kron(I, U_A)
                circuit = circuit @ np.kron(P_phi(self.phases[2*i + 3], self.m), np.eye(2**self.n))

        if self.d%2 == 0: # d is even
            circuit = np.kron(P_phi(self.phases[0], self.m), np.eye(2**self.n))
            for i in range(self.d // 2):
                circuit = circuit @ np.kron(I, U_A_dagger)
                circuit = circuit @ np.kron(P_phi(self.phases[2*i + 1], self.m), np.eye(2**self.n))
                circuit = circuit @ np.kron(I, U_A)
                circuit = circuit @ np.kron(P_phi(self.phases[2*i + 2], self.m), np.eye(2**self.n))
                
        circuit = kron_list([H] + [I]*self.m + [I]*self.n) @ circuit @ kron_list([H] + [I]*self.m + [I]*self.n)
        return circuit

class QuantumSolverQSVT(MatrixInversionQSVT):
    """
    Extends MatrixInversionQSVT to solve linear systems using QSVT-based matrix inversion.
    """
    def __init__(self, A, epsilon=1e-2, method="laurent", verbose=True):
        """
        Initializes the QuantumSolverQSVT with the given matrix A and parameters.
        """
        super().__init__(A, epsilon, method, verbose)
        
    def solve(self, b, return_error=False):
        """
        Solves the linear system Ax = b using QSVT-based matrix inversion.
        """
        assert len(b) == self.N, f"Mismatch in dimension of b: expected {self.N} but received {len(b)}."
        b_norm = np.linalg.norm(b)
        b_normalized = b / b_norm # Normalization
        zero_state = np.array([1,0])
        out_state = self.circuit @ kron_list([zero_state]*(self.m+1) + [b_normalized])
        result = out_state[:self.N]
        sol = result / self.scale / self.alpha * b_norm
        if return_error:
            ground_truth = np.linalg.solve(self.A,b)
            error = np.linalg.norm(ground_truth - sol)
            return sol, error
        else:
            return sol
            
    def state_preparation_b(self, b):
        """
        Creates a state-preparation unitary matrix Q s.t. Q|0> = |b> / ||b>|
        """
        assert len(b) == self.N, f"Mismatch in dimension of b: expected {self.N} but received {len(b)}."
        B = np.column_stack((b / np.linalg.norm(b), np.random.randn(self.N, self.N - 1)))
        # Apply QR decomposition to B to get an orthonormal basis
        # The Q matrix from the QR decomposition will be unitary, and the first column will be b
        Q, _ = scipy.linalg.qr(B, mode="economic")
        return Q