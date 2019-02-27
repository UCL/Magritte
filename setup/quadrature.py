from numpy.polynomial.hermite import hermroots, hermval
from math                     import factorial


def H_roots(n):
    coeffs  = [0.0 for _ in range(n)]
    coeffs += [1.0]
    return hermroots(coeffs)


def H_weights(n):
    coeffs  = [0.0 for _ in range(n-1)]
    coeffs += [1.0]
    H = hermval(H_roots(n), coeffs)
    return 2**(n-1) * factorial(n) / (n*H)**2
