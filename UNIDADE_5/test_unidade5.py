from bvp_solvers import FDM1D, FDM2D, FEM1D
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_tests():
    print("="*60)
    print("UNIDADE 5 - PROBLEMAS DE VALOR DE CONTORNO (TESTES)")
    print("="*60)

    # -------------------------------------------------------
    # PVC1: u'' - u = 0, u(0)=0, u(1)=1, N=8 particoes
    # Solucao exata: u(x) = (e^{-x} - e^x) / (e^{-1} - e)
    # -------------------------------------------------------
    print("\n[PVC-1: u'' - u = 0,  u(0)=0, u(1)=1]")
    def exact_pvc1(x):
        return (math.exp(-x) - math.exp(x)) / (math.exp(-1) - math.e)

    x_fd, u_fd = FDM1D.solve(0, 1, 0, 1, lambda x: 0.0, c2=1, c1=0, c0=-1, N=8)
    x_fe, u_fe = FEM1D.solve(0, 1, 0, 1, 0.0, c2=1, c1=0, c0=-1, n_elem=8)

    print(f"{'x':>6}  {'Exata':>10}  {'MDF':>10}  {'Erro MDF':>10}  {'MEF':>10}  {'Erro MEF':>10}")
    for i, x in enumerate(x_fd):
        ex = exact_pvc1(x)
        mdf = u_fd[i]
        # interpola MEF no mesmo ponto
        mef = np.interp(x, x_fe, u_fe)
        err_mdf = abs(mdf - ex) / (abs(ex) + 1e-15)
        err_mef = abs(mef - ex) / (abs(ex) + 1e-15)
        print(f"{x:6.3f}  {ex:10.6f}  {mdf:10.6f}  {err_mdf:10.6f}  {mef:10.6f}  {err_mef:10.6f}")

    # -------------------------------------------------------
    # PVC2: Laplaciano(u) = 4, dominio [0,1]x[0,1], bordas=0
    # -------------------------------------------------------
    print("\n[PVC-2: lap(u)=4, Dirichlet=0 em todo contorno, N=8]")
    u2d, xs, ys = FDM2D.solve(1.0, 1.0, 7, f_func=lambda x, y: 4.0,
                               u_norte=0, u_sul=0, u_leste=0, u_oeste=0)
    n_int = u2d.shape[0] - 2
    mid = n_int // 2
    print("Linha central (y=0.5):")
    print([f"{u2d[mid+1, j+1]:.4f}" for j in range(n_int)])
    print(f"Valor central u(0.5,0.5) = {u2d[mid+1, mid+1]:.6f}")

    # -------------------------------------------------------
    # PVC3: u'' + 7u' - u = 2, u(0)=10, u(2)=1, h=0.1
    # Solucao exata: u(x) = C1*e^{r1*x} + C2*e^{r2*x} - 2
    # -------------------------------------------------------
    print("\n[PVC-3: u'' + 7u' - u = 2,  u(0)=10, u(2)=1]")
    disc = math.sqrt(49 + 4)  # sqrt(53)
    r1 = (-7 + disc) / 2
    r2 = (-7 - disc) / 2
    # Condicoes: C1 + C2 = 12, C1*e^{2r1} + C2*e^{2r2} = 3
    A_sys = np.array([[1, 1], [math.exp(2*r1), math.exp(2*r2)]])
    rhs_sys = np.array([12.0, 3.0])
    C1, C2 = np.linalg.solve(A_sys, rhs_sys)

    def exact_pvc3(x):
        return C1 * math.exp(r1 * x) + C2 * math.exp(r2 * x) - 2

    N3 = 20  # 20 subintervalos = h=0.1
    x_fd3, u_fd3 = FDM1D.solve(0, 2, 10, 1, lambda x: 2.0, c2=1, c1=7, c0=-1, N=N3)
    x_fe3, u_fe3 = FEM1D.solve(0, 2, 10, 1, 2.0, c2=1, c1=7, c0=-1, n_elem=N3)

    checkpts = [0.0, 0.1, 0.5, 1.0, 1.5, 1.9, 2.0]
    print(f"{'x':>5}  {'Exata':>10}  {'MDF':>10}  {'Err MDF':>8}  {'MEF':>10}  {'Err MEF':>8}")
    for xp in checkpts:
        ex = exact_pvc3(xp)
        mdf = float(np.interp(xp, x_fd3, u_fd3))
        mef = float(np.interp(xp, x_fe3, u_fe3))
        err_mdf = abs(mdf - ex) / (abs(ex) + 1e-15)
        err_mef = abs(mef - ex) / (abs(ex) + 1e-15)
        print(f"{xp:5.2f}  {ex:10.6f}  {mdf:10.6f}  {err_mdf:8.6f}  {mef:10.6f}  {err_mef:8.6f}")

    # -------------------------------------------------------
    # Gerar grafico comparativo do PVC3
    # -------------------------------------------------------
    x_exact = np.linspace(0, 2, 300)
    u_exact = np.array([exact_pvc3(xi) for xi in x_exact])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_exact, u_exact, 'k-',  lw=2,   label='Solução Exata')
    ax.plot(x_fd3,   u_fd3,   'b--', lw=1.5, marker='o', ms=4, label='MDF (N=20)')
    ax.plot(x_fe3,   u_fe3,   'r:',  lw=1.5, marker='s', ms=4, label='MEF (N=20)')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u(x)', fontsize=12)
    ax.set_title("PVC: $u'' + 7u' - u = 2$, $u(0)=10$, $u(2)=1$", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig('plot_unidade5.pdf', dpi=150)
    fig.savefig('plot_unidade5.png', dpi=150)
    print("\nGrafico salvo em plot_unidade5.pdf / .png")

if __name__ == "__main__":
    run_tests()
