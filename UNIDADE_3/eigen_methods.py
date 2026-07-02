import numpy as np
import scipy.linalg

class EigenMethods:
    """Implementações de métodos numéricos para o cálculo de Autovalores e Autovetores."""
    
    @staticmethod
    def _normalize(v):
        norm = np.linalg.norm(v)
        if norm < 1e-15:
            return v
        return v / norm

    @classmethod
    def power_method_regular(cls, A, v0, tol=1e-6, max_iter=1000):
        """Método da Potência Regular: encontra o autovalor dominante."""
        v = cls._normalize(np.array(v0, dtype=float))
        lam_old = 0.0
        
        for _ in range(max_iter):
            y = A @ v
            lam_new = v @ y
            v = cls._normalize(y)
            
            if np.abs(lam_new - lam_old) < tol:
                return lam_new, v
            lam_old = lam_new
            
        return lam_old, v

    @classmethod
    def power_method_inverse(cls, A, v0, tol=1e-6, max_iter=1000):
        """Método da Potência Inversa (com fatoração LU): encontra o menor autovalor em módulo."""
        v = cls._normalize(np.array(v0, dtype=float))
        lu, piv = scipy.linalg.lu_factor(A)
        lam_old = 0.0
        
        for _ in range(max_iter):
            y = scipy.linalg.lu_solve((lu, piv), v)
            v_new = cls._normalize(y)
            
            # Aproximação do autovalor pelo Quociente de Rayleigh com v
            lam_new = v_new @ (A @ v_new)
            
            if np.abs(lam_new - lam_old) < tol:
                return lam_new, v_new
            lam_old = lam_new
            v = v_new
            
        return lam_old, v

    @classmethod
    def power_method_shifted(cls, A, v0, mu, tol=1e-6, max_iter=1000):
        """Método da Potência Deslocada: encontra autovalor próximo a mu."""
        n = A.shape[0]
        A_shifted = A - mu * np.eye(n)
        lam_shifted, v = cls.power_method_inverse(A_shifted, v0, tol, max_iter)
        return lam_shifted + mu, v

    @staticmethod
    def householder_tridiagonalize(A):
        """Tridiagonalização de Householder para matrizes simétricas."""
        n = A.shape[0]
        A_k = A.copy().astype(float)
        H_acc = np.eye(n)
        
        for i in range(n - 2):
            x = A_k[i+1:, i]
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-12:
                continue
                
            v = np.zeros_like(x)
            v[0] = norm_x if x[0] < 0 else -norm_x
            
            N = x - v
            norm_N = np.linalg.norm(N)
            if norm_N < 1e-12:
                continue
            n_vec = N / norm_N
            
            H_i = np.eye(n)
            H_i[i+1:, i+1:] = np.eye(n - i - 1) - 2.0 * np.outer(n_vec, n_vec)
            
            A_k = H_i @ A_k @ H_i
            H_acc = H_acc @ H_i
            
        return A_k, H_acc

    @staticmethod
    def qr_decomposition(A):
        """Decomposição QR usando reflexões de Householder iterativas."""
        m, n = A.shape
        R = A.copy().astype(float)
        Q = np.eye(m)
        
        for i in range(min(m, n)):
            x = R[i:, i]
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-12:
                continue
                
            v = np.zeros_like(x)
            v[0] = norm_x if x[0] < 0 else -norm_x
            
            N = x - v
            norm_N = np.linalg.norm(N)
            if norm_N < 1e-12:
                continue
            n_vec = N / norm_N
            
            H_i = np.eye(m)
            H_i[i:, i:] = np.eye(m - i) - 2.0 * np.outer(n_vec, n_vec)
            
            R = H_i @ R
            Q = Q @ H_i.T
            
        return Q, R

    @classmethod
    def qr_algorithm(cls, A, tol=1e-6, max_iter=1000):
        """Método QR iterativo para encontrar autovalores."""
        A_k = A.copy().astype(float)
        P = np.eye(A.shape[0])
        
        for _ in range(max_iter):
            Q, R = cls.qr_decomposition(A_k)
            A_k = R @ Q
            P = P @ Q
            
            # Soma dos quadrados fora da diagonal (abaixo da diagonal para matriz genérica convergindo)
            off_diag = np.sum(np.tril(A_k, -1)**2)
            if off_diag < tol:
                break
                
        return np.diag(A_k), P

    @staticmethod
    def jacobi_method(A, tol=1e-6, max_iter=1000):
        """Método de Jacobi para diagonalizar matrizes simétricas."""
        n = A.shape[0]
        A_k = A.copy().astype(float)
        J_acc = np.eye(n)
        
        for _ in range(max_iter):
            # Acha o maior elemento fora da diagonal
            off_diag = np.abs(A_k - np.diag(np.diag(A_k)))
            max_idx = np.unravel_index(np.argmax(off_diag, axis=None), off_diag.shape)
            i, j = max_idx
            
            if off_diag[i, j] < tol:
                break
                
            if np.abs(A_k[i, i] - A_k[j, j]) < 1e-15:
                theta = np.pi / 4.0 if A_k[i, j] > 0 else -np.pi / 4.0
            else:
                theta = 0.5 * np.arctan(2.0 * A_k[i, j] / (A_k[i, i] - A_k[j, j]))
                
            J = np.eye(n)
            c, s = np.cos(theta), np.sin(theta)
            J[i, i] = c
            J[j, j] = c
            J[i, j] = s
            J[j, i] = -s
            
            A_k = J.T @ A_k @ J
            J_acc = J_acc @ J
            
        return np.diag(A_k), J_acc

class SVDMethods:
    """Implementação da Decomposição em Valores Singulares baseada no Método de Jacobi."""
    
    @staticmethod
    def compute_svd(A, eps=1e-7):
        A = np.array(A, dtype=float)
        m, n = A.shape
        At = A.T
        
        # Calcular SVD por A^T A ou A A^T usando Jacobi
        is_m_less_n = (m < n)
        base = A @ At if is_m_less_n else At @ A
        
        eigenvalues, eigenvectors = EigenMethods.jacobi_method(base)
        
        # Ordenar os autovalores
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        sigmas = np.sqrt(np.maximum(eigenvalues, 0.0))
        
        Sigma = np.zeros((m, n))
        for i in range(min(m, n)):
            Sigma[i, i] = sigmas[i] if sigmas[i] > eps else 0.0
            
        rank = np.sum(sigmas > eps)
        
        if is_m_less_n:
            U = eigenvectors
            V = np.zeros((n, n))
            for i in range(m):
                if sigmas[i] > eps:
                    V[:, i] = (At @ U[:, i]) / sigmas[i]
            
            # Ortogonalizar o resto (Gram-Schmidt simplificado para nulos, ou deixar numpy tratar)
            # Para o contexto deste relatório, a base incompleta de V para m < n será complementada via null_space
            if rank < n:
                null_v = scipy.linalg.null_space(A)
                cols_to_fill = n - rank
                if null_v.shape[1] >= cols_to_fill:
                    V[:, rank:rank+cols_to_fill] = null_v[:, :cols_to_fill]
        else:
            V = eigenvectors
            U = np.zeros((m, m))
            for i in range(n):
                if sigmas[i] > eps:
                    U[:, i] = (A @ V[:, i]) / sigmas[i]
                    
            if rank < m:
                null_u = scipy.linalg.null_space(At)
                cols_to_fill = m - rank
                if null_u.shape[1] >= cols_to_fill:
                    U[:, rank:rank+cols_to_fill] = null_u[:, :cols_to_fill]
                    
        A_rec = U @ Sigma @ V.T
        return U, Sigma, V, A_rec, rank
