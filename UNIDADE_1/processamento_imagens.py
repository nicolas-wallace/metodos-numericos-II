import numpy as np
import os
from PIL import Image
from scipy.signal import convolve2d

def get_gaussian_kernel_3x3(sigma: float = 1.0) -> np.ndarray:
    """Retorna um kernel Gaussiano 3x3 básico."""
    ax = np.linspace(-(3 - 1) / 2., (3 - 1) / 2., 3)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

def get_sobel_high_precision_kernels() -> tuple:
    """
    Retorna os kernels de Sobel X e Y baseados na derivada O(h^4) central:
    Derivada 1a O(h^4): [1, -8, 0, 8, -1] / 12.
    """
    d_kernel = np.array([1, -8, 0, 8, -1]) / 12.0
    smooth = np.array([1, 4, 6, 4, 1]) / 16.0
    
    # Sobel X: Derivada em X (linhas x colunas)
    sobel_x = np.outer(smooth, d_kernel)
    # Sobel Y: Derivada em Y
    sobel_y = np.outer(d_kernel, smooth)
    return sobel_x, sobel_y

def get_laplace_high_precision_kernel() -> np.ndarray:
    """
    Retorna o kernel Laplaciano O(h^4) central:
    Derivada 2a O(h^4): [-1, 16, -30, 16, -1] / 12.
    """
    d2_kernel = np.array([-1, 16, -30, 16, -1]) / 12.0
    
    # Laplaciano é d2/dx2 + d2/dy2
    # Para dx2: aplica d2 em x, identidade em y
    # Para dy2: aplica d2 em y, identidade em x
    laplace = np.zeros((5, 5))
    laplace[2, :] += d2_kernel # d2/dx2 (ao longo das colunas, na linha do meio)
    laplace[:, 2] += d2_kernel # d2/dy2 (ao longo das linhas, na coluna do meio)
    
    return laplace

def apply_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Aplica uma convolução 2D separadamente em cada canal de cor, ou na imagem toda se for cinza."""
    if len(image.shape) == 3:
        result = np.zeros_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            result[:, :, c] = convolve2d(image[:, :, c], kernel, mode='same', boundary='symm')
        return result
    else:
        return convolve2d(image, kernel, mode='same', boundary='symm')

def threshold_image(image: np.ndarray, t: float) -> np.ndarray:
    """Binariza a imagem (0 ou 255) baseado no threshold (tolerância) t."""
    return np.where(image < t, 0, 255).astype(np.uint8)

def alg1_sobel(image: np.ndarray, t: float) -> np.ndarray:
    """Algoritmo 1: Filtro Gaussiano -> Sobel X, Y -> Magnitude -> Threshold"""
    gauss_k = get_gaussian_kernel_3x3(1.0)
    img_smooth = apply_kernel(image, gauss_k)
    
    sobel_x, sobel_y = get_sobel_high_precision_kernels()
    
    # Se for RGB, converter para escala de cinza primeiro para o detector de bordas
    # Ou podemos calcular nos canais separados e somar, como pede o relatório. 
    # O relatório fala: "para obter resultado colorido aplicando individualmente em cada canal"
    
    gx = apply_kernel(img_smooth, sobel_x)
    gy = apply_kernel(img_smooth, sobel_y)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    return threshold_image(magnitude, t)

def alg2_laplace(image: np.ndarray, t: float) -> np.ndarray:
    """Algoritmo 2: Filtro Gaussiano -> Laplaciano -> Threshold"""
    gauss_k = get_gaussian_kernel_3x3(1.0)
    img_smooth = apply_kernel(image, gauss_k)
    
    laplace_k = get_laplace_high_precision_kernel()
    
    # O Laplaciano pode ter valores negativos. Usamos o valor absoluto.
    laplace_result = np.abs(apply_kernel(img_smooth, laplace_k))
    
    return threshold_image(laplace_result, t)


if __name__ == "__main__":
    # Testar com uma imagem sintética
    test_img = np.ones((10, 10))
    test_img[3:7, 3:7] = 9.0
    
    print("Testando Algoritmo 1 (Sobel):")
    res1 = alg1_sobel(test_img, 2.0)
    print(res1)
    
    print("\nTestando Algoritmo 2 (Laplace):")
    res2 = alg2_laplace(test_img, 2.0)
    print(res2)
    
    # Se a imagem do Batman existir, processar ela também
    batman_path = r"C:\Users\nicol\OneDrive\Área de Trabalho\Metodos-Numericos-II\Metodos-Numericos-II-master\LUCAS\Unidade 1\Processamento de Imagens\Batman.jpg"
    if os.path.exists(batman_path):
        print("\nProcessando a imagem Batman.jpg ...")
        img = np.array(Image.open(batman_path), dtype=np.float64)
        
        # Algoritmo 1 com threshold de 50
        out1 = alg1_sobel(img, 50)
        Image.fromarray(out1).save(r"C:\Users\nicol\OneDrive\Área de Trabalho\Metodos-Numericos-II\Metodos-Numericos-II-master\NICOLAS\UNIDADE_1\batman_sobel_refat.jpg")
        
        # Algoritmo 2 com threshold de 20
        out2 = alg2_laplace(img, 20)
        Image.fromarray(out2).save(r"C:\Users\nicol\OneDrive\Área de Trabalho\Metodos-Numericos-II\Metodos-Numericos-II-master\NICOLAS\UNIDADE_1\batman_laplace_refat.jpg")
        print("Imagens processadas salvas com sucesso em NICOLAS/UNIDADE_1!")
