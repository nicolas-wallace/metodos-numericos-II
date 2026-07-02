from eigen_methods import EigenMethods, SVDMethods
import numpy as np

def run_tests():
    print("="*60)
    print("UNIDADE 3 - AUTOVALORES E SVD (TESTES)")
    print("="*60)
    
    np.set_printoptions(precision=4, suppress=True)

    # Matrizes de teste conforme o relatorio
    A1 = np.array([[5, 2, 1], [2, 3, 1], [1, 1, 2]])
    A3 = np.array([
        [40, 8, 4, 2, 1],
        [8, 30, 12, 6, 2],
        [4, 12, 20, 1, 2],
        [2, 6, 1, 25, 4],
        [1, 2, 2, 4, 5]
    ])
    v0 = np.array([1, 1, 1])
    v0_5 = np.array([1, 1, 1, 1, 1])
    
    print("\n[Metodos da Potencia - A1]")
    lam_dom, v_dom = EigenMethods.power_method_regular(A1, v0)
    print(f"Potencia Regular : lam = {lam_dom:.5f}, v = {v_dom}")
    
    lam_inv, v_inv = EigenMethods.power_method_inverse(A1, v0)
    print(f"Potencia Inversa : lam = {lam_inv:.5f}, v = {v_inv}")
    
    lam_shift, v_shift = EigenMethods.power_method_shifted(A1, v0, mu=5)
    print(f"Deslocamento (5) : lam = {lam_shift:.5f}, v = {v_shift}")

    print("\n[Transformacoes Ortogonais e Decomposicoes - A3]")
    # Householder
    A_tri, H = EigenMethods.householder_tridiagonalize(A3)
    print("Tridiagonalizacao (Householder):")
    print(A_tri)

    # QR
    Q, R = EigenMethods.qr_decomposition(A3)
    print("\nDecomposicao QR (Matriz R):")
    print(R)
    
    lam_qr, P = EigenMethods.qr_algorithm(A3)
    print("\nAutovalores (QR Iterativo):", np.sort(lam_qr)[::-1])
    
    # Jacobi
    lam_jacobi, J = EigenMethods.jacobi_method(A3)
    print("\nAutovalores (Jacobi):", np.sort(lam_jacobi)[::-1])

    print("\n[Decomposicao SVD]")
    A_svd = np.array([[3, 2, 2], [2, 3, -2]])
    U, Sigma, V, A_rec, rank = SVDMethods.compute_svd(A_svd)
    
    print("Matriz Original:")
    print(A_svd)
    print("\nValores Singulares (Sigma):")
    print(np.diag(Sigma))
    print(f"\nPosto: {rank}")
    print("\nErro de Reconstrucao (Norma Maxima):", np.max(np.abs(A_svd - A_rec)))

if __name__ == "__main__":
    run_tests()
