import numpy as np
import sys

class LBM_2_Carlemann:
    def __init__(self):
        # Constants for the D1Q3 lattice (1D, 3 velocities)
        self.w = np.array([2/3, 1/6, 1/6])  # Weights for D1Q3 lattice
        self.e = np.array([0, 1, -1])  # Lattice directions: [0, +1, -1]
        self.c_s = 1 / np.sqrt(3)  # Speed of sound for D1Q3 lattice...is this the delta_x\delta_t ??
        self.g = 9.81  # Acceleration due to gravity (m/s^2)
        self.kn = 0.000000001 #Knudsen number, much less than 1 for chapman-enskogg expansion

        # Parameters
        self.tau = 1.0  # Relaxation time
        self.Nx = 5  # Number of grid points...my code for the F matrices makes the kernel die if this number is 81 or higher
        self.L = 10.0  # Length of the domain (in meters)

        # Initialize macroscopic variables: density(height) and velocity field
        self.h = np.ones(self.Nx)  # height field
        self.u = np.zeros(self.Nx)  # Velocity field

        # Initialize distribution functions (f_i), f has 3 directions (D1Q3)
        self.f = np.zeros((self.Nx, 3))  # Distribution functions for D1Q3
        self.feq = np.zeros_like(self.f)  # Equilibrium distribution functions

    '''
    # Helper function to compute the equilibrium distribution function for D1Q3
    def equilibrium(self):
        """
        Compute the equilibrium distribution function f_i^{(eq)} based on macroscopic variables.
        """
        feq = np.zeros((self.Nx, 3))
        usqr = self.u**2  # squared velocity
        for i in range(3):
            cu = self.e[i] * self.u  # Dot product of e_i and velocity
            feq[:, i] = self.w[i] * self.h * (1 + 3 * cu / self.c_s**2 + 9 / 2 * (cu / self.c_s**2)*2 - 3 / 2 * usqr / self.c_s**2)
        return feq
    
    # Helper function to compute moments (density and momentum)
    def compute_moments(self):
        """
        Compute the moments of the distribution function.
        Moment 0: Density, Moment 1: Momentum
        """
        moment_0 = np.sum(self.f, axis=-1)  # Moment 0: density
        moment_1 = np.sum(self.f * self.e, axis=-1)  # Moment 1: momentum (density * velocity)
        return moment_0, moment_1
    
    
    #function to build the carleman linearization matrices...keep?
    def carleman_lin_matrix(self):
        Cc = gen_collision(self)
        Cs = gen_streaming(self)
        C = Cc + Cs
        return C
    '''
     #return a 1D array with one non-zero element of value 1 at specified index
    def one_nonzero(self,dim, n):
        array = np.zeros((dim))
        if n>-1 and n<dim:
            array[n] = 1
        return array
    #make the F matrices for the collision matrix for n grid points
    def gen_F(self):
        f1 = np.zeros((3,3))
        f1[0,1] = f1[0,2] = 1
        f1[1,1] = 1/(2*self.c_s) - 1
        f1[1,2] = -1/(2*self.c_s)
        f1[2,1] = -1/(2*self.c_s)
        f1[2,2] = 1/(2*self.c_s) -1
        f1 = (1/(self.tau*self.kn))*f1
        
        f2 = np.zeros((3,9))
        for i in range(9):
            f2[0,i] = -self.g
            f2[1,i]  = self.g
            f2[2,i]  = self.g
        f2[0,4] = f2[0,4] -4 
        f2[0,8] = f2[0,8] -4
        f2[0,5] = f2[0,5] +4 
        f2[0,7] = f2[0,7] +4
        for i in range(2):
            f2[i+1, 4] = f2[i+1, 4] +2
            f2[i+1, 8] = f2[i+1, 8] +2

            f2[i+1, 5] = f2[i+1, 5] -2
            f2[i+1, 7] = f2[i+1, 7] -2
        f2 = (1/(2*self.tau*self.kn*self.c_s**2))*f2

        f3 = np.zeros((3,9))
        f3[0,4] = 2
        f3[0,8] = 2
        f3[0,5] = -2 
        f3[0,7] = -2
        for i in range(2):
            f3[i+1, 4] = -1
            f3[i+1, 8] = -1

            f3[i+1, 5] = 1
            f3[i+1, 7] = 1
        f3 = np.hstack((f3,f3,f3))
        f3 = (1/(self.tau*self.kn))*f3

        #generalise to n grid points...strategy is to stack matrices, not create matrix of matrices I think...?
        n = self.Nx
        Q = len(self.e)
        '''
        F1 = np.zeros((dim,Q, dim*Q))
        F2 = np.zeros((dim,Q, (dim**2)*(Q**2)))
        F3 = np.zeros((dim,Q, (dim**3)*(Q**3)))
        '''
        I = self.one_nonzero(n, 0)
        F1 = np.kron(I, f1)
        F2 = np.kron(np.kron(I,I) , f2)
        F3 = np.kron(np.kron(np.kron(I, I), I), f3)
        

        for i in range(n-1):
            print("this is i: " , i+1)
            I = self.one_nonzero(n, i+1)
            F1 = np.vstack((F1, np.kron(I, f1)))
            F2 = np.vstack((F2, np.kron(np.kron(I,I) , f2)))
            F3 = np.vstack((F3, np.kron(np.kron(np.kron(I, I), I), f3)))
        
        return F1,F2,F3
        
    #make A matrices for the collision matrix
    def gen_A(self, F1,F2,F3):
        A11 = F1
        print("just made A11 \n")
        A12 = F2
        print("just made A12 \n")
        A13 = F3
        print("just made A13 \n")
        Q= len(self.e)
        n = self.Nx
        dim = n*Q
        I = np.identity(dim)
        A22 = np.kron(F1,I) + np.kron(I, F1)
        #slows down at A22, kernel dies after with Nx = 50
        print("just made A22 \n")
        A23 = np.kron(F2,I) + np.kron(I, F2)
        print("just made A23 \n")
        A33 = np.kron(np.kron(F1,I), I) + np.kron(np.kron(I,F1), I) + np.kron(np.kron(I,I), F1)
        print("just made A33 \n")

        return A11, A12, A13, A22, A23, A33
    #make collision matrix 
    def gen_collision(self, A11, A12, A13, A22, A23, A33):
        C1 = np.vstack((A11,np.zeros((A22.shape[0]+A33.shape[0],A11.shape[1]))))
        C2 = np.vstack((A12,A22,np.zeros((A33.shape[0],A12.shape[1]))))
        C3 = np.vstack((A13, A23, A33))

        Cc = np.hstack((C1,C2,C3))

        return Cc
    
    #make streaming matrix, restricted to NN, 2nd order accuracy...only makes sense if we are considering more than 1 grid point
    def gen_streaming(self):
        Q= len(self.e)
        n = self.Nx
        inv_delta = n/(2*self.L)
        dim = n*Q
        I = np.identity(dim)
        S = np.zeros((dim, dim))
        for i in range(dim):
            #deal with edge case here...periodic or bounce back BC...here I do code for periodic
            if i<Q:
                S[i,i+Q] = inv_delta*self.e[(i%3)-1]
                S[i, (dim-Q)+i] = -inv_delta*self.e[(i%3)-1]
            elif i>dim -Q - 1:
                S[i,dim-i] = inv_delta*self.e[(i%3)-1]
                S[i,i-Q] = -inv_delta*self.e[(i%3)-1]
            else:
                S[i,i+Q] = inv_delta*self.e[(i%3)-1]
                S[i,i-Q] = -inv_delta*self.e[(i%3)-1]

        B11 = S
        B22 = np.kron(S,I) + np.kron(I, S)
        B33 = np.kron(np.kron(S,I), I) + np.kron(np.kron(I,S), I) + np.kron(np.kron(I,I), S)

        C1 = np.vstack((B11,np.zeros((B22.shape[0]+B33.shape[0],B11.shape[1]))))
        C2 = np.vstack((np.zeros((B11.shape[0],B22.shape[1])),B22,np.zeros((B33.shape[0],B22.shape[1]))))
        C3 = np.vstack((np.zeros((B11.shape[0]+B22.shape[0],B33.shape[1])), B33))

        Cs = np.hstack((C1,C2,C3))

        return Cs

        
        


#Test to see if I have the right matrices
Matrix_C = LBM_2_Carlemann()
F1,F2,F3  = Matrix_C.gen_F()

A11, A12, A13, A22, A23, A33 = Matrix_C.gen_A(F1,F2,F3)
Cc = Matrix_C.gen_collision(A11, A12, A13, A22, A23, A33)
Cs = Matrix_C.gen_streaming()

C = Cc +Cs
print("This is the dimension of C: \n", C.shape)
#print collision matrix and intermediate matrices
np.set_printoptions(threshold=sys.maxsize)
print("checking the dimensions of the matrices \n")
print("dim Matrix F1: \n", F1.shape)
print("dim Matrix F2: \n", F2.shape)
print("dim Matrix F3: \n", F3.shape)
print(" \n")

'''
print("dim Matrix A11: \n", A11.shape)
print("dim Matrix A12: \n", A12.shape)
print("dim Matrix A13: \n", A13.shape)
print("dim Matrix A22: \n", A22.shape)
print("dim Matrix A23: \n", A23.shape)
print("dim Matrix A33: \n", A33.shape)
print(" \n")
print("dim Carleman collision matrix: \n", C.shape)
print("\n")
print("checking the entries of the matrices \n")
print("Matrix F1: \n", F1)
print("Matrix F2: \n", F2)
print("Matrix F3: \n", F3)
print(" \n")
print("Matrix A11: \n", A11)
print("Matrix A12: \n", A12)
print("Matrix A13: \n", A13)
print("Matrix A22: \n", A22)
print("Matrix A23: \n", A23)
print("Matrix A33: \n", A33)
print(" \n")
print("Carleman collision matrix: \n", C)
'''
    

    
