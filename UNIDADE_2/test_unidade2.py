from numerics_integration import NewtonCotes, GaussianQuadrature, ExponentialTransforms, DoubleIntegration
import math

def run_tests():
    print("="*60)
    print("UNIDADE 2 - INTEGRAÇÃO NUMÉRICA (TESTES)")
    print("="*60)

    # 1. Newton-Cotes
    # f(x) = (sin(2x) + 4x^2 + 3x)^2, x in [0, 1]
    # analytical approx: 17.8764703
    def f1(x):
        return (math.sin(2*x) + 4*x**2 + 3*x)**2

    print("\n[Newton-Cotes - fechadas vs abertas]")
    print(f"Trapezoidal (100): {NewtonCotes.trapezoidal(f1, 0, 1, 100):.6f}")
    print(f"Simpson 1/3 (10):  {NewtonCotes.simpson_1_3(f1, 0, 1, 10):.6f}")
    print(f"Boole (5):         {NewtonCotes.boole(f1, 0, 1, 5):.6f}")
    print(f"Open Degree 1(10): {NewtonCotes.open_degree_1(f1, 0, 1, 10):.6f}")

    # 2. Gaussian Quadrature
    print("\n[Quadraturas de Gauss]")
    # Legendre: integral de f1 em [0, 1]
    print(f"Gauss-Legendre (n=4) : {GaussianQuadrature.legendre(f1, 0, 1, 4):.6f}")
    
    # Hermite: integral de e^{-x^2} * cos(x) em [-inf, inf]
    def f_hermite(x): return math.cos(x)
    print(f"Gauss-Hermite (n=4)  : {GaussianQuadrature.hermite(f_hermite, 4):.6f}")
    
    # Laguerre: integral de e^{-x} * x^2 em [0, inf]
    def f_laguerre(x): return x**2
    print(f"Gauss-Laguerre (n=3) : {GaussianQuadrature.laguerre(f_laguerre, 3):.6f}")

    # Chebyshev: integral de f(x)/sqrt(1-x^2) em [-1, 1], f(x)=x^2
    def f_cheby(x): return x**2
    print(f"Gauss-Chebyshev(n=4) : {GaussianQuadrature.chebyshev(f_cheby, 4):.6f}")

    # 3. Singularities
    print("\n[Integração com Singularidades]")
    # Integral de x^{-1/2} em [0,1] -> = 2
    def f_sing_1(x):
        # Evitar divisão exata por zero na eval
        if x <= 1e-15: return 0
        return x**(-0.5)
    
    print(f"Exponencial Simples (n=50) : {ExponentialTransforms.simple_exponential(f_sing_1, 0, 1, 50):.6f}")
    print(f"Duplo Exponencial (n=50)   : {ExponentialTransforms.double_exponential(f_sing_1, 0, 1, 50):.6f}")
    
    # 4. Integrais Duplas
    print("\n[Integral Dupla]")
    def f2d(x, y):
        return x*y
    print(f"Simpson 2D f=xy em [0,1]x[0,1]: {DoubleIntegration.simpson_2d(f2d, 0, 1, 0, 1, 10, 10):.6f} (Esperado 0.25)")

if __name__ == "__main__":
    run_tests()
