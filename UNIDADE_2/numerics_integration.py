import numpy as np
import math

class NewtonCotes:
    """Implementações das regras de Newton-Cotes Fechadas e Abertas com suporte a partição (composta)."""
    @staticmethod
    def _composite(func, a, b, n, rule):
        h = (b - a) / n
        return sum(rule(func, a + i * h, a + (i + 1) * h) for i in range(n))

    @classmethod
    def trapezoidal(cls, f, a, b, n=1):
        def rule(f_sub, a_sub, b_sub):
            return (b_sub - a_sub) * (f_sub(a_sub) + f_sub(b_sub)) / 2.0
        return cls._composite(f, a, b, n, rule)
        
    @classmethod
    def simpson_1_3(cls, f, a, b, n=1):
        def rule(f_sub, a_sub, b_sub):
            m = (a_sub + b_sub) / 2.0
            return (b_sub - a_sub) * (f_sub(a_sub) + 4*f_sub(m) + f_sub(b_sub)) / 6.0
        return cls._composite(f, a, b, n, rule)

    @classmethod
    def simpson_3_8(cls, f, a, b, n=1):
        def rule(f_sub, a_sub, b_sub):
            h_sub = (b_sub - a_sub) / 3.0
            return 3.0 * h_sub / 8.0 * (f_sub(a_sub) + 3*f_sub(a_sub+h_sub) + 3*f_sub(a_sub+2*h_sub) + f_sub(b_sub))
        return cls._composite(f, a, b, n, rule)

    @classmethod
    def boole(cls, f, a, b, n=1):
        def rule(f_sub, a_sub, b_sub):
            h_sub = (b_sub - a_sub) / 4.0
            return 2.0 * h_sub / 45.0 * (7*f_sub(a_sub) + 32*f_sub(a_sub+h_sub) + 12*f_sub(a_sub+2*h_sub) + 32*f_sub(a_sub+3*h_sub) + 7*f_sub(b_sub))
        return cls._composite(f, a, b, n, rule)

    @classmethod
    def open_degree_1(cls, f, a, b, n=1):
        def rule(f_sub, a_sub, b_sub):
            h_sub = (b_sub - a_sub) / 3.0
            return 3.0 * h_sub / 2.0 * (f_sub(a_sub+h_sub) + f_sub(a_sub+2*h_sub))
        return cls._composite(f, a, b, n, rule)

class GaussianQuadrature:
    """Implementações das Quadraturas de Gauss dinâmicas usando NumPy para buscar nós e pesos polinomiais reais."""
    @staticmethod
    def legendre(f, a, b, deg):
        x, w = np.polynomial.legendre.leggauss(deg)
        x_mapped = 0.5 * (b - a) * x + 0.5 * (a + b)
        return 0.5 * (b - a) * np.sum(w * np.vectorize(f)(x_mapped))
        
    @staticmethod
    def hermite(f, deg): 
        x, w = np.polynomial.hermite.hermgauss(deg)
        return np.sum(w * np.vectorize(f)(x))

    @staticmethod
    def laguerre(f, deg):
        x, w = np.polynomial.laguerre.laggauss(deg)
        return np.sum(w * np.vectorize(f)(x))

    @staticmethod
    def chebyshev(f, deg):
        k = np.arange(1, deg + 1)
        x = np.cos((2 * k - 1) * np.pi / (2 * deg))
        w = np.full(deg, np.pi / deg)
        return np.sum(w * np.vectorize(f)(x))

class ExponentialTransforms:
    """Métodos de integração para lidar com funções que possuem singularidades nos limites (T -> infinito)."""
    @staticmethod
    def simple_exponential(f, a, b, n, T=3.5):
        h = 2 * T / n
        t = np.linspace(-T, T, n+1)
        phi = np.tanh(t)
        dphi = 1.0 / np.cosh(t)**2
        
        x = 0.5 * (a + b) + 0.5 * (b - a) * phi
        weights = np.full(n + 1, h)
        weights[0] = weights[-1] = h / 2.0
        
        f_val = np.vectorize(f)(x)
        return 0.5 * (b - a) * np.sum(weights * f_val * dphi)

    @staticmethod
    def double_exponential(f, a, b, n, T=3.5):
        h = 2 * T / n
        t = np.linspace(-T, T, n+1)
        
        st = np.sinh(t)
        ct = np.cosh(t)
        phi = np.tanh(np.pi / 2.0 * st)
        dphi = (np.pi / 2.0 * ct) / (np.cosh(np.pi / 2.0 * st)**2)
        
        x = 0.5 * (a + b) + 0.5 * (b - a) * phi
        weights = np.full(n + 1, h)
        weights[0] = weights[-1] = h / 2.0
        
        f_val = np.vectorize(f)(x)
        return 0.5 * (b - a) * np.sum(weights * f_val * dphi)

class DoubleIntegration:
    """Ferramentas para integração bidimensional (e.g. Volumes, Áreas de Superfície)."""
    @staticmethod
    def simpson_2d(f, ax, bx, ay, by, nx, ny):
        hx = (bx - ax) / nx
        hy = (by - ay) / ny
        def w_simpson(i, n):
            if i == 0 or i == n: return 1
            return 4 if i % 2 != 0 else 2
            
        result = 0.0
        for i in range(nx + 1):
            x = ax + i * hx
            wx = w_simpson(i, nx) * hx / 3.0
            for j in range(ny + 1):
                y = ay + j * hy
                wy = w_simpson(j, ny) * hy / 3.0
                result += wx * wy * f(x, y)
        return result
