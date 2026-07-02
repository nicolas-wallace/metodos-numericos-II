import numpy as np
from typing import Callable, Dict

def _apply_finite_difference(f: Callable[[float], float], x: float, h: float, coefs: Dict[float, float], denom: float) -> float:
    """Aplica a fórmula de diferença finita."""
    result = sum(weight * f(x + offset * h) for offset, weight in coefs.items())
    return result / denom

class Derivative:
    """
    Classe utilitária para cálculo de derivadas numéricas.
    Implementa diferenças finitas progressivas (forward), regressivas (backward) e centrais,
    com erros de aproximação linear (O(h)), quadrático (O(h²)), cúbico (O(h³)) e quártico (O(h⁴)).
    """
    
    @staticmethod
    def forward(f: Callable[[float], float], x: float, h: float, order: int = 1, error_degree: int = 1) -> float:
        formulas = {
            1: {
                1: ({1: 1, 0: -1}, h),
                2: ({2: -1, 1: 4, 0: -3}, 2 * h),
                3: ({3: 2, 2: -9, 1: 18, 0: -11}, 6 * h),
                4: ({4: -3, 3: 16, 2: -36, 1: 48, 0: -25}, 12 * h)
            },
            2: {
                1: ({2: 1, 1: -2, 0: 1}, h**2),
                2: ({3: -1, 2: 4, 1: -5, 0: 2}, h**2),
                3: ({4: 11, 3: -56, 2: 114, 1: -104, 0: 35}, 12 * h**2),
                4: ({5: -10, 4: 61, 3: -156, 2: 214, 1: -154, 0: 45}, 12 * h**2)
            },
            3: {
                1: ({3: 1, 2: -3, 1: 3, 0: -1}, h**3),
                2: ({4: -1.5, 3: 7, 2: -12, 1: 9, 0: -2.5}, h**3),
                3: ({5: 7, 4: -47, 3: 122, 2: -154, 1: 95, 0: -23}, 4 * h**3),
                4: ({6: -15, 5: 104, 4: -319, 3: 544, 2: -533, 1: 280, 0: -61}, 8 * h**3)
            }
        }
        coefs, denom = formulas[order][error_degree]
        return _apply_finite_difference(f, x, h, coefs, denom)

    @staticmethod
    def backward(f: Callable[[float], float], x: float, h: float, order: int = 1, error_degree: int = 1) -> float:
        formulas = {
            1: {
                1: ({0: 1, -1: -1}, h),
                2: ({0: 3, -1: -4, -2: 1}, 2 * h),
                3: ({0: 11, -1: -18, -2: 9, -3: -2}, 6 * h),
                4: ({0: 25, -1: -48, -2: 36, -3: -16, -4: 3}, 12 * h)
            },
            2: {
                1: ({0: 1, -1: -2, -2: 1}, h**2),
                2: ({0: 2, -1: -5, -2: 4, -3: -1}, h**2),
                3: ({0: 35, -1: -104, -2: 114, -3: -56, -4: 11}, 12 * h**2),
                4: ({0: 45, -1: -154, -2: 214, -3: -156, -4: 61, -5: -10}, 12 * h**2)
            },
            3: {
                1: ({0: 1, -1: -3, -2: 3, -3: -1}, h**3),
                2: ({0: 2.5, -1: -9, -2: 12, -3: -7, -4: 1.5}, h**3),
                3: ({0: 23, -1: -95, -2: 154, -3: -122, -4: 47, -5: -7}, 4 * h**3),
                4: ({0: 61, -1: -280, -2: 533, -3: -544, -4: 319, -5: -104, -6: 15}, 8 * h**3)
            }
        }
        coefs, denom = formulas[order][error_degree]
        return _apply_finite_difference(f, x, h, coefs, denom)

    @staticmethod
    def central(f: Callable[[float], float], x: float, h: float, order: int = 1, error_degree: int = 2) -> float:
        # Central errors are naturally even (O(h^2), O(h^4)), except for some specialized formulas.
        # Mapping according to the provided report.
        formulas = {
            1: {
                1: ({0.5: 1, -0.5: -1}, h),
                2: ({1: 1, -1: -1}, 2 * h),
                3: ({-1.5: 1, -0.5: -27, 0.5: 27, 1.5: -1}, 24 * h),
                4: ({-2: 1, -1: -8, 1: 8, 2: -1}, 12 * h)
            },
            2: {
                1: ({1: 1, 0: -2, -1: 1}, h**2),
                2: ({1: 1, 0: -2, -1: 1}, h**2), # Central O(h^2) 2nd derivative is same as O(h) in some lit, matching report
                3: ({1: 1, 0: -2, -1: 1}, h**2), # Using same for fallback
                4: ({-2: -1, -1: 16, 0: -30, 1: 16, 2: -1}, 12 * h**2)
            },
            3: {
                1: ({2: 1, 1: -2, -1: 2, -2: -1}, 2 * h**3), # Note report has (2h) typo, probably 2h^3
                2: ({2: 1, 1: -2, -1: 2, -2: -1}, 2 * h**3),
                3: ({-2.5: -11, -1.5: 35, -0.5: -38, 0.5: 14, 1.5: 1, 2.5: -1}, 8 * h**3),
                4: ({-3: -1, -2: 8, -1: -13, 1: 13, 2: -8, 3: 1}, -8 * h**3)
            }
        }
        
        coefs, denom = formulas[order][error_degree]
        return _apply_finite_difference(f, x, h, coefs, denom)

if __name__ == "__main__":
    def f_test(x):
        return x**2
    
    x_val = 8.0
    dx = 0.5
    print("="*50)
    print("TESTE DE DERIVADAS - F(X) = X^2")
    print(f"X = {x_val}, dX = {dx}")
    print("="*50)
    
    print(f"Valor Exato 1a Derivada (2x): {2*x_val}")
    print(f"Forward O(h)   1a: {Derivative.forward(f_test, x_val, dx, order=1, error_degree=1)}")
    print(f"Backward O(h^2) 1a: {Derivative.backward(f_test, x_val, dx, order=1, error_degree=2)}")
    print(f"Central O(h^4)  1a: {Derivative.central(f_test, x_val, dx, order=1, error_degree=4)}")
    
    print(f"\nValor Exato 2a Derivada (2): {2.0}")
    print(f"Forward O(h)   2a: {Derivative.forward(f_test, x_val, dx, order=2, error_degree=1)}")
    print(f"Central O(h^4)  2a: {Derivative.central(f_test, x_val, dx, order=2, error_degree=4)}")
    
    print(f"\nValor Exato 3a Derivada (0): {0.0}")
    print(f"Central O(h^4)  3a: {Derivative.central(f_test, x_val, dx, order=3, error_degree=4)}")
