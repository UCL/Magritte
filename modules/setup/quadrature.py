from numpy.polynomial.hermite import hermroots, hermval
from math                     import factorial
from setupFunctions           import CArray

def H_roots(n):
    coeffs  = [0.0 for _ in range(n)]
    coeffs += [1.0]
    return hermroots(coeffs)


def H_weights(n):
    coeffs  = [0.0 for _ in range(n-1)]
    coeffs += [1.0]
    H = hermval(H_roots(n), coeffs)
    return 2**(n-1) * factorial(n) / (n*H)**2


def write_quadrature_file(fileName, n):
    roots   = CArray(H_roots(n).tolist())
    weights = CArray(H_weights(n).tolist())
    with open(fileName, 'w') as file:
        file.write(f'// Written by quadrature.py                              \n\n')
        file.write(f'const int    N_QUADRATURE_POINTS = {n};                  \n\n')
        file.write(f'const double H_roots   [N_QUADRATURE_POINTS] = {roots};  \n\n')
        file.write(f'const double H_weights [N_QUADRATURE_POINTS] = {weights};    ')
