import cv2
import numpy as np

def load_image(file_path):
    # Memuat gambar dari file
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

def calculate_glcm(image, distance=1, angle=0):
    # Mendapatkan dimensi gambar
    rows, cols = image.shape
    
    # Inisialisasi matriks GLCM
    glcm = np.zeros((256, 256), dtype=np.float32)
    
    # Menghitung GLCM
    for i in range(rows):
        for j in range(cols):
            # Menghitung koordinat tetangga berdasarkan jarak dan sudut
            if angle == 0:
                x = i
                y = j + distance
            elif angle == 45:
                x = i - distance
                y = j + distance
            elif angle == 90:
                x = i - distance
                y = j
            elif angle == 135:
                x = i - distance
                y = j - distance
            else:
                continue
            
            if x >= 0 and x < rows and y >= 0 and y < cols:
                glcm[image[i, j], image[x, y]] += 1
    
    # Normalisasi GLCM
    glcm /= glcm.sum()
    return glcm

def calculate_glcm_features(glcm):
    # Inisialisasi fitur
    contrast = 0
    dissimilarity = 0
    homogeneity = 0
    energy = 0
    correlation = 0
    asm = 0
    mean_i = np.mean(glcm, axis=0)
    mean_j = np.mean(glcm, axis=1)
    std_i = np.std(glcm, axis=0)
    std_j = np.std(glcm, axis=1)
    
    # Menghitung fitur dari GLCM
    for i in range(256):
        for j in range(256):
            contrast += (i - j) ** 2 * glcm[i, j]
            dissimilarity += abs(i - j) * glcm[i, j]
            homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
            energy += glcm[i, j] ** 2
            asm += glcm[i, j] ** 2
            if std_i[i] * std_j[j] > 0:
                correlation += ((i - mean_i[i]) * (j - mean_j[j]) * glcm[i, j]) / (std_i[i] * std_j[j])
    
    features = {
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation,
        'asm': asm
    }
    
    return features

def print_features(features):
    # Mencetak fitur GLCM
    for feature_name, feature_value in features.items():
        print(f"{feature_name}: {feature_value}")

if __name__ == "__main__":
    # File path gambar
    file_path = "images/sample_image.png"
    
    # Memuat gambar
    image = load_image(file_path)
    
    # Menghitung GLCM
    glcm = calculate_glcm(image)
    
    # Menghitung fitur GLCM
    features = calculate_glcm_features(glcm)
    
    # Mencetak fitur GLCM
    print_features(features)
