import os
import cv2
import math
import numpy as np
import sys

# Function to compute GCDs between consecutive pixel intensity values (1D flattened)
def get_gcds(intensities):
    gcds = set()
    for i in range(len(intensities) - 1):
        g = math.gcd(intensities[i], intensities[i + 1])
        if g != 1:
            gcds.add(g)
    return gcds

# Function to compute frequency of GCDs between all unique intensity pairs in a grayscale image
def get_frequencies(gray_image):
    unique_vals = np.unique(gray_image.flatten())
    freq = {}
    gcd_values = set()
    for i in range(len(unique_vals)):
        for j in range(i + 1, len(unique_vals)):
            g = math.gcd(unique_vals[i], unique_vals[j])
            if g != 1:
                freq[g] = freq.get(g, 0) + 1
                gcd_values.add(g)
    freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
    return freq, gcd_values

# Creates a filter from top 9 most frequent GCD values, reshaped into 3x3
def find_filter1(freqs):
    top = list(freqs.keys())[:9]
    matrix = np.array(top).reshape(3, 3)
    return matrix / np.sum(matrix)

# Creates a filter from sorted GCD values sampled across the range
def find_filter2(gcds):
    vals = sorted(gcds)
    step = len(vals) // 9
    selected = [vals[i * step] for i in range(9)] if step > 0 else [1] * 9
    matrix = np.array(selected).reshape(3, 3)
    return matrix / np.sum(matrix)

# Adaptive filter based on overall brightness
def find_filter3(brightness):
    if brightness < 85:  # Dark image
        return np.array([[90, 90, 90], [90, 180, 90], [90, 90, 90]]) / 450
    elif brightness < 170:  # Medium brightness
        return np.array([[30, 60, 30], [60, 100, 60], [30, 60, 30]]) / 520
    else:  # Bright image
        return np.array([[20, 30, 20], [30, 95, 30], [20, 30, 20]]) / 360

# Filter using Interquartile Range (IQR) to measure image contrast
def find_filter4_iqr(gray_image):
    q1, q3 = np.percentile(gray_image, [25, 75])
    iqr = q3 - q1
    base = np.full((3, 3), iqr)
    base[1, 1] = iqr * 2  # Emphasize center pixel
    return base / np.sum(base)

# Bilateral filter preserves edges while smoothing
def find_filter5_bilateral(gray_image):
    diameter = 7           # Pixel neighborhood size
    sigma_color = 25       # Intensity similarity
    sigma_space = 25       # Spatial closeness
    filtered = cv2.bilateralFilter(gray_image, diameter, sigma_color, sigma_space)
    return filtered

# Applies a convolutional filter to an image
def apply_filter(img, filt):
    return cv2.filter2D(img, -1, filt)

# Calculate thresholds for segmentation using vertical GCDs and all pairwise GCDs
def get_thresholds(gray_image):
    vertical = gray_image.flatten(order='F')  # Column-wise flattening
    vertical_gcds = get_gcds(vertical)

    values = np.unique(gray_image.flatten())
    all_gcds = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            g = math.gcd(values[i], values[j])
            if g != 1:
                all_gcds.append(g)

    return {
        "vertical": int(np.mean(list(vertical_gcds))) if vertical_gcds else 128,
        "allvalues": int(np.mean(all_gcds)) if all_gcds else 128
    }

# Save the processed image to disk if not already present
def save_image(img, base_name, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, base_name)
    if not os.path.exists(path):
        cv2.imwrite(path, img)
        print(f"✅ Saved: {path}")
    else:
        cv2.imwrite(path, img)
        print(f"⏩ Skipped: {path}")
    return path

# Main image processing function
def process_image(image_path, mode):
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Could not load image: {image_path}")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    base_name = os.path.basename(image_path)

    freqs, gcds = get_frequencies(gray)
    brightness = np.mean(gray)
    thresholds = get_thresholds(gray)

    # Generate all filters
    filters = {
        "smooth1": find_filter1(freqs),
        "smooth2": find_filter2(gcds),
        "smooth3": find_filter3(brightness),
        "smooth4": find_filter4_iqr(gray),
        "smooth5": find_filter5_bilateral(gray)  # Note: this is already a filtered image
    }

    for key, filt in filters.items():
        # Bilateral filtering already returns final image, so skip convolution
        filtered = filt if key == "smooth5" else apply_filter(gray, filt)
        save_image(filtered, base_name, f"output/{mode}/{key}")

        # Segmentation using thresholding
        for tname in ["vertical"]:
            threshold = thresholds[tname] + 48  # Offset to adjust segmentation sensitivity
            segmented = np.where(filtered < threshold, 0, 255).astype(np.uint8)
            save_image(segmented, base_name, f"output/{mode}/{key}_segm_{tname}")

# Processes all images in the specified folders recursively
def batch_process_folders(folders=["mias"]):
    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    mode = folder
                    process_image(os.path.join(root, file), mode)

# Entry point
if __name__ == "__main__":
    print(1)
    batch_process_folders(["mias"])
    print(2)
