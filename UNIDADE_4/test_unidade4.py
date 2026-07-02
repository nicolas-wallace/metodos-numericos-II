from ivp_solvers import IVPSolvers
import numpy as np

def run_tests():
    print("="*60)
    print("UNIDADE 4 - PROBLEMAS DE VALOR INICIAL (TESTES)")
    print("="*60)
    
    np.set_printoptions(precision=6, suppress=True)

    print("\n[PVI-1 Canônico Escalar]")
    # dy/dt = (2/3)y, y(0) = 2
    # Analítico: y(t) = 2 * exp(2t/3) -> y(8) = 414.25449793
    def f_scalar(t, y): return (2.0/3.0) * y
    y_true_8 = 2.0 * np.exp(16.0 / 3.0)
    
    for dt in [0.1, 0.01]:
        print(f"\n--- dt = {dt} ---")
        _, y_euler_ex = IVPSolvers.euler_explicit(f_scalar, [2.0], 0.0, 8.0, dt)
        _, y_euler_im = IVPSolvers.euler_implicit(f_scalar, [2.0], 0.0, 8.0, dt)
        _, y_rk4 = IVPSolvers.runge_kutta_4(f_scalar, [2.0], 0.0, 8.0, dt)
        _, y_pc4 = IVPSolvers.predictor_corrector_4(f_scalar, [2.0], 0.0, 8.0, dt)
        
        print(f"Euler Explícito : {y_euler_ex[-1][0]:.6f} (Erro: {abs(y_euler_ex[-1][0] - y_true_8):.2e})")
        print(f"Euler Implícito : {y_euler_im[-1][0]:.6f} (Erro: {abs(y_euler_im[-1][0] - y_true_8):.2e})")
        print(f"RK4             : {y_rk4[-1][0]:.6f} (Erro: {abs(y_rk4[-1][0] - y_true_8):.2e})")
        print(f"Pred.-Corret.   : {y_pc4[-1][0]:.6f} (Erro: {abs(y_pc4[-1][0] - y_true_8):.2e})")

    print("\n[PVI-2 Balístico (Sistema Acoplado)]")
    # v'(t) = -10 - 0.125 * v(t)
    # y'(t) = v(t)
    # S(0) = [v(0), y(0)] = [5.0, 200.0]
    
    def f_ballistic(t, S):
        v, y = S
        return np.array([-10.0 - 0.125 * v, v])

    def extrair_metricas(t_vals, y_vals):
        y_max = np.max(y_vals[:, 1])
        # Achar tempo e velocidade de impacto interpolando ao cruzar y=0 (se cruzar)
        # Assumindo que cruzou:
        for i in range(1, len(y_vals)):
            if y_vals[i, 1] <= 0 and y_vals[i-1, 1] > 0:
                y0, y1 = y_vals[i-1, 1], y_vals[i, 1]
                t0, t1 = t_vals[i-1], t_vals[i]
                v0, v1 = y_vals[i-1, 0], y_vals[i, 0]
                
                frac = y0 / (y0 - y1)
                t_impact = t0 + frac * (t1 - t0)
                v_impact = v0 + frac * (v1 - v0)
                return y_max, t_impact, v_impact
        return y_max, None, None

    for dt in [0.1, 0.01, 0.001]:
        print(f"\n--- dt = {dt} ---")
        t_rk3, y_rk3 = IVPSolvers.runge_kutta_3(f_ballistic, [5.0, 200.0], 0.0, 10.0, dt)
        y_max_3, t_imp_3, v_imp_3 = extrair_metricas(t_rk3, y_rk3)
        print(f"RK3 : y_max={y_max_3:.6f}, t_imp={t_imp_3:.6f}, v_imp={v_imp_3:.6f}")
        
        t_pc4, y_pc4 = IVPSolvers.predictor_corrector_4(f_ballistic, [5.0, 200.0], 0.0, 10.0, dt)
        y_max_4, t_imp_4, v_imp_4 = extrair_metricas(t_pc4, y_pc4)
        print(f"PC4 : y_max={y_max_4:.6f}, t_imp={t_imp_4:.6f}, v_imp={v_imp_4:.6f}")

if __name__ == "__main__":
    run_tests()
