import cudaq
import numpy as np
from typing import Union, List
import math
import scipy.linalg

def QR_decomposition(b):  
    b_norm = np.linalg.norm(b)
    N = len(b)
    B = np.column_stack((b / b_norm, np.random.randn(N, N - 1)))
    Q, _ = scipy.linalg.qr(B, mode='economic')
    return Q.flatten().tolist()  # Convert to 1D list for CUDAQ

@cudaq.kernel
def initialize_b(qb: cudaq.qvector, Q: list[float]):
    # Register and apply unitary
    cudaq.compute_action("Q", Q)
    Q(qb)

@cudaq.kernel
def QFT(q: cudaq.qvector):
    for i in range(len(q)):
        h(q[i])
        for j in range(i + 1, len(q)):
            angle = math.pi / (2**(j - i))  # Corrected angle calculation
            ctrl_rz(angle, q[j], q[i])

@cudaq.kernel
def QPE(q_eigenvalue: cudaq.qvector, q_eigenvector: cudaq.qvector, U: list[float]):
    # Register unitary
    cudaq.compute_action("U", U)
    
    # Apply Hadamard gates
    for i in range(len(q_eigenvalue)):
        h(q_eigenvalue[i])
   
    # Controlled-U operations
    for i in range(len(q_eigenvalue)):
        for j in range(2**i):
            ctrl_u(q_eigenvalue[i], q_eigenvector, U)

    # Inverse QFT
    QFT.adj(q_eigenvalue)

@cudaq.kernel
def controlled_rotations(q_eigenvalue: cudaq.qvector, q_flag: cudaq.qvector, thetas: list[float]):
    for i in range(len(q_eigenvalue)):
        ctrl_ry(thetas[i], q_eigenvalue[i], q_flag[0])

@cudaq.kernel
def HHL(A: list[float], b: list[float], thetas: list[float]):
    # Calculate number of qubits needed
    n = int(math.log2(len(b)))
    
    # Allocate quantum registers
    qb = cudaq.qvector(n)    # Vector qubits
    ql = cudaq.qvector(n)    # Eigenvalue qubits
    qf = cudaq.qvector(1)    # Flag qubit
    
    # 1. Initialize state |b>
    Q = QR_decomposition(b)
    initialize_b(qb, Q)
    
    # 2. Quantum Phase Estimation
    # Convert A to unitary form
    A_matrix = np.array(A).reshape(2**n, 2**n)
    U = np.exp(1j * A_matrix * 2 * np.pi)
    U_list = U.flatten().tolist()
    
    QPE(ql, qb, U_list)
    
    # 3. Controlled rotations
    controlled_rotations(ql, qf, thetas)
    
    # 4. Inverse QPE
    QPE.adj(ql, qb, U_list)
    
    return mz(qf)  # Return measurement results

# Example usage
def main():
    n = 2  # Start with a smaller example
    A = np.ones((2**n, 2**n))
    b = np.ones(2**n)
    
    # Calculate eigenvalues and rotation angles
    lambdas, _ = np.linalg.eig(A)
    thetas = np.arcsin(np.min(np.abs(lambdas)) / np.abs(lambdas)).tolist()
    
    # Flatten matrices for CUDAQ
    A_flat = A.flatten().tolist()
    b_flat = b.tolist()
    
    # Create and run the kernel
    result = cudaq.sample(HHL)(A_flat, b_flat, thetas)
    print(result)

if __name__ == "__main__":
    main()