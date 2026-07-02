import numpy as np

class IVPSolvers:
    """Implementações orientadas a objetos de métodos numéricos para Problemas de Valor Inicial (PVIs)."""
    
    @staticmethod
    def euler_explicit(f, y0, t0, tf, dt):
        t_vals = [t0]
        y_vals = [np.array(y0, dtype=float)]
        t, y = t0, y_vals[0].copy()
        
        while t < tf - 1e-12:
            y = y + dt * np.array(f(t, y))
            t += dt
            t_vals.append(t)
            y_vals.append(y.copy())
        return np.array(t_vals), np.array(y_vals)

    @staticmethod
    def euler_implicit(f, y0, t0, tf, dt, tol=1e-8, max_iter=100):
        t_vals = [t0]
        y_vals = [np.array(y0, dtype=float)]
        t, y = t0, y_vals[0].copy()
        
        while t < tf - 1e-12:
            t_next = t + dt
            y_next = y.copy() 
            
            for _ in range(max_iter):
                y_next_new = y + dt * np.array(f(t_next, y_next))
                if np.linalg.norm(y_next_new - y_next) < tol:
                    y_next = y_next_new
                    break
                y_next = y_next_new
                
            t = t_next
            y = y_next
            t_vals.append(t)
            y_vals.append(y.copy())
        return np.array(t_vals), np.array(y_vals)

    @staticmethod
    def runge_kutta_3(f, y0, t0, tf, dt):
        t_vals = [t0]
        y_vals = [np.array(y0, dtype=float)]
        t, y = t0, y_vals[0].copy()
        
        while t < tf - 1e-12:
            k1 = np.array(f(t, y))
            k2 = np.array(f(t + dt/2, y + dt*k1/2))
            k3 = np.array(f(t + dt, y + dt*(-k1 + 2*k2)))
            
            y = y + (dt/6) * (k1 + 4*k2 + k3)
            t += dt
            t_vals.append(t)
            y_vals.append(y.copy())
        return np.array(t_vals), np.array(y_vals)

    @staticmethod
    def runge_kutta_4(f, y0, t0, tf, dt):
        t_vals = [t0]
        y_vals = [np.array(y0, dtype=float)]
        t, y = t0, y_vals[0].copy()
        
        while t < tf - 1e-12:
            k1 = np.array(f(t, y))
            k2 = np.array(f(t + dt/2, y + dt*k1/2))
            k3 = np.array(f(t + dt/2, y + dt*k2/2))
            k4 = np.array(f(t + dt, y + dt*k3))
            
            y = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            t += dt
            t_vals.append(t)
            y_vals.append(y.copy())
        return np.array(t_vals), np.array(y_vals)

    @classmethod
    def predictor_corrector_4(cls, f, y0, t0, tf, dt, tol=1e-8, max_iter=100):
        # Arranque com RK4 para os 4 primeiros pontos (0, 1, 2, 3)
        t_rk, y_rk = cls.runge_kutta_4(f, y0, t0, t0 + 3*dt, dt)
        t_vals = list(t_rk)
        y_vals = list(y_rk)
        
        if t_vals[-1] >= tf - 1e-12:
            return np.array(t_vals), np.array(y_vals)
            
        t = t_vals[-1]
        while t < tf - 1e-12:
            f0 = np.array(f(t_vals[-1], y_vals[-1]))
            f1 = np.array(f(t_vals[-2], y_vals[-2]))
            f2 = np.array(f(t_vals[-3], y_vals[-3]))
            f3 = np.array(f(t_vals[-4], y_vals[-4]))
            
            t_next = t + dt
            
            # Predição (AB4)
            y_pred = y_vals[-1] + (dt/24) * (55*f0 - 59*f1 + 37*f2 - 9*f3)
            y_corr = y_pred.copy()
            
            # Correção (AM4)
            for _ in range(max_iter):
                f_p = np.array(f(t_next, y_corr))
                y_corr_new = y_vals[-1] + (dt/24) * (9*f_p + 19*f0 - 5*f1 + f2)
                
                if np.linalg.norm(y_corr_new - y_corr) < tol:
                    y_corr = y_corr_new
                    break
                y_corr = y_corr_new
                
            t = t_next
            t_vals.append(t)
            y_vals.append(y_corr.copy())
        return np.array(t_vals), np.array(y_vals)
