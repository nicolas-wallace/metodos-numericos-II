import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class FDM1D:
    """Metodo das Diferencas Finitas (MDF) em 1D para PVCs do tipo c2*u'' + c1*u' + c0*u = f."""

    @staticmethod
    def solve(a, b, ua, ub, f_func, c2, c1, c0, N):
        """
        Resolve PVC linear de 2a ordem via MDF com diferencas centrais.

        Parametros
        ----------
        a, b   : extremos do dominio
        ua, ub : condicoes de Dirichlet
        f_func : callable, termo fonte f(x)
        c2, c1, c0 : coeficientes da EDO
        N      : numero de subintervalos (N+1 pontos internos)
        """
        # Malha interna (N-1 nos internos entre a e b)
        x_int = np.linspace(a, b, N + 1)[1:-1]   # N-1 pontos internos
        n = len(x_int)
        dx = (b - a) / N

        a_coef = c2 / dx**2 - c1 / (2 * dx)    # sub-diagonal
        b_coef = -2 * c2 / dx**2 + c0           # diagonal
        c_coef = c2 / dx**2 + c1 / (2 * dx)    # super-diagonal

        # Monta sistema tridiagonal esparso
        A = diags(
            [a_coef * np.ones(n - 1), b_coef * np.ones(n), c_coef * np.ones(n - 1)],
            [-1, 0, 1],
            format='csc'
        ).toarray()

        rhs = np.array([f_func(xi) for xi in x_int], dtype=float)
        # Ajustar condicoes de contorno
        rhs[0] -= a_coef * ua
        rhs[-1] -= c_coef * ub

        u_int = np.linalg.solve(A, rhs)

        x_full = np.concatenate([[a], x_int, [b]])
        u_full = np.concatenate([[ua], u_int, [ub]])
        return x_full, u_full


class FDM2D:
    """MDF 2D para equacao de Poisson: lap(u) = f(x,y), condicoes de Dirichlet."""

    @staticmethod
    def solve(Lx, Ly, N, f_func,
              u_norte=0.0, u_sul=0.0, u_leste=0.0, u_oeste=0.0,
              tol=1e-6, max_iter=10000):
        """
        Resolve Poisson em [0,Lx]x[0,Ly] com N subintervalos em cada direcao.
        Usa iteracao de Gauss-Seidel.
        """
        nx = ny = N
        dx = Lx / (nx + 1)
        dy = Ly / (ny + 1)

        u = np.zeros((ny + 2, nx + 2))
        # Condicoes de contorno
        u[0, :] = u_sul
        u[-1, :] = u_norte
        u[:, 0] = u_oeste
        u[:, -1] = u_leste

        xs = np.linspace(dx, Lx - dx, nx)
        ys = np.linspace(dy, Ly - dy, ny)

        inv_dx2 = 1.0 / dx**2
        inv_dy2 = 1.0 / dy**2
        denom = 2 * (inv_dx2 + inv_dy2)

        for _ in range(max_iter):
            u_old = u.copy()
            for i in range(1, ny + 1):
                for j in range(1, nx + 1):
                    rhs = (
                        (u[i, j + 1] + u[i, j - 1]) * inv_dx2
                        + (u[i + 1, j] + u[i - 1, j]) * inv_dy2
                        - f_func(xs[j - 1], ys[i - 1])
                    )
                    u[i, j] = rhs / denom

            if np.max(np.abs(u - u_old)) < tol:
                break

        return u, xs, ys


class FEM1D:
    """Metodo dos Elementos Finitos (MEF) em 1D com elementos lineares (Galerkin)."""

    @staticmethod
    def solve(a, b, ua, ub, f_val, c2, c1, c0, n_elem):
        """
        Resolve c2*u'' + c1*u' + c0*u = f via MEF com n_elem elementos lineares.
        """
        n_nos = n_elem + 1
        h = (b - a) / n_elem
        nodes = np.linspace(a, b, n_nos)

        K = np.zeros((n_nos, n_nos))
        F = np.zeros(n_nos)

        # Matrizes locais analiticas para elementos lineares uniformes
        # Formulacao fraca: integral(c2*u'*v') - integral(c1*u'*v) - integral(c0*u*v) = -integral(f*v)
        # => K_e = Kd - Kc - Kr,  F_e = -f_val*h/2 * [1,1]

        # Difusao: c2/h * [[1,-1],[-1,1]]
        Kd = (c2 / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        # Conveccao antissimetrica: c1/2 * [[-1,1],[-1,1]]
        Kc = (c1 / 2.0) * np.array([[-1.0, 1.0], [-1.0, 1.0]])
        # Reacao (massa consistente): c0*h/6 * [[2,1],[1,2]]
        Kr = (c0 * h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
        # Forca
        Fe_e = (-f_val * h / 2.0) * np.array([1.0, 1.0])

        for e in range(n_elem):
            dofs = [e, e + 1]
            K_e = Kd - Kc - Kr
            for i, gi in enumerate(dofs):
                for j, gj in enumerate(dofs):
                    K[gi, gj] += K_e[i, j]
                F[gi] += Fe_e[i]

        # Aplicar condicoes de Dirichlet por penalizacao de linha/coluna
        for bc_node, bc_val in [(0, ua), (n_nos - 1, ub)]:
            F -= K[:, bc_node] * bc_val
            K[bc_node, :] = 0
            K[:, bc_node] = 0
            K[bc_node, bc_node] = 1
            F[bc_node] = bc_val

        u = np.linalg.solve(K, F)
        return nodes, u
