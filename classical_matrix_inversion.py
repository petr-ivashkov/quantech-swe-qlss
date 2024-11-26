import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Generate a large sparse matrix
def generate_sparse_matrix(n, density=0.01):
    """
    Generate a random sparse matrix of size (n x n) with given density.
    The matrix is symmetric and positive definite.
    """
    A = sp.random(n, n, density=density, format='csr')
    A = 0.5 * (A + A.T)  # Make it symmetric
    A += n * sp.eye(n)   # Make it positive definite
    return A

# Problem dimensions (reduced by a factor of 10)
subvector_dim = 27930  # Reduced size of each subvector x_1, x_2, x_3
num_subvectors = 3    # Number of subvectors
total_dim = subvector_dim * num_subvectors  # Total size of x

# Generate the sparse matrix A and the RHS vector b
A = generate_sparse_matrix(total_dim, density=0.00001)
b = np.random.rand(total_dim)

# Preconditioner: Incomplete LU (ILU)
ilu = spla.spilu(A)
M = spla.LinearOperator(A.shape, matvec=lambda x: ilu.solve(x))

# Solve using Conjugate Gradient
x, info = spla.cg(A, b, M=M, tol=1e-6, maxiter=1000)

# Check results
if info == 0:
    print("Conjugate Gradient converged successfully.")
elif info > 0:
    print(f"Conjugate Gradient did not converge after {info} iterations.")
else:
    print("Conjugate Gradient failed due to an error.")

# Extract subvectors x_1, x_2, x_3
x_1 = x[:subvector_dim]
x_2 = x[subvector_dim:2 * subvector_dim]
x_3 = x[2 * subvector_dim:]

# Compute the residual ||Ax - b||
residual = np.linalg.norm(A @ x - b)
print(f"Residual norm: {residual}")

# Print the first three entries of each subvector
print("First three entries of x_1:", x_1[:3])
print("First three entries of x_2:", x_2[:3])
print("First three entries of x_3:", x_3[:3])
