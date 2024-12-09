import numpy as np

# Helper functions and definitions

X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
P0 = np.array([[1,0],[0,0]])
I = np.array([[1,0],[0,1]])
H = np.array([[1,1],[1,-1]]) / np.sqrt(2)

def kron_list(arr_list):
    """
    Takes a list of numpy arrays [A1, A2, ...] and computes their tensor product A1 (x) A2 (x) ...
    """
    if len(arr_list) == 1:
        return arr_list[0]
    else:
        return np.kron(arr_list[0], kron_list(arr_list[1:]))

def generate_random_unitary(dim) -> np.ndarray:
    """
    Generates a random unitary matrix of size dim x dim.
    """
    # Create a random complex matrix
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)

    # Perform QR decomposition
    q, r = np.linalg.qr(random_matrix)

    # Ensure unitary property by normalizing R"s diagonal
    d = np.diagonal(r)
    d = d / np.abs(d)
    unitary_matrix = q * d

    return unitary_matrix