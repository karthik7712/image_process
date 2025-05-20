import math
from PIL import Image
import numpy as np
import cv2

gcd_values = set()
thresholds = {}
results = []
frequencies = {}



def get_gcds(intensities):
    gcds = set()
    for i in range(len(intensities)):
        if i == len(intensities)-1:
            continue
        x, y = intensities[i], intensities[i+1]
        i+=1
        g = math.gcd(x, y)
        if g != 1:
            gcds.add(g)
    # print(np.mean(list(gcds)))
    return gcds

def select_numbers_with_least_variation(numbers, k):
    numbers.sort()
    n = len(numbers)
    interval = n // (k + 1)

    selected_numbers = []
    for i in range(1, k + 1):
        index = i * interval
        selected_numbers.append(numbers[index])

    return selected_numbers



def find_filter1(frequencies):
    top_gcd = []
    count = 0
    for key in frequencies.keys():
        if count<9:
            top_gcd.append(key)
            count += 1
        else:
            break

    matrix = [top_gcd[i:i + 3] for i in range(0, len(top_gcd), 3)]

    vals = np.array(matrix)
    x = np.sum(vals)
    filter_3x3 = vals / x
    # print("The filter is :\n", filter_3x3)
    return filter_3x3


def smooth1(image,filter):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_array = np.array(gray_image)
    filtered_image = np.zeros_like(img_array)

    for y in range(1, img_array.shape[0] - 1):
        for x in range(1, img_array.shape[1] - 1):
            filtered_image[y, x] = round(np.sum(img_array[y - 1:y + 2, x - 1:x + 2] * filter), 0)

    smoothed_image = np.clip(filtered_image, 0, 255)

    # smoothed_image = Image.fromarray(filtered_image.astype(np.uint8))
    # cv2.imshow("Most common filtered Image", filtered_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # smoothed_image.save(f"most_common_smoothed_{image_path}")
    return smoothed_image


def find_filter2():
    #least variant Element filter
    global gcd_values
    top_gcd2 = select_numbers_with_least_variation(list(gcd_values),9)
    # print(top_gcd2)
    filter_matrix2 = np.array(top_gcd2).reshape(3, 3) / np.sum(top_gcd2)
    # print("The second filter is :\n",filter_matrix2)
    return filter_matrix2


def smooth2(image,filter):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_array = np.array(gray_image)
    padded_array = np.pad(gray_image.astype(np.float64),((1, 1), (1, 1)), mode='edge')

    smoothed_image = np.zeros_like(img_array, dtype=np.float64)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            window = padded_array[i:i + 3, j:j + 3]
            smoothed_image[i, j] = np.sum(window * filter)

    smoothed_image2 = np.clip(smoothed_image, 0, 255).astype(np.uint8)

    # smoothed_image_2_pil = Image.fromarray(smoothed_image2)
    # smoothed_image_2_pil.save(f'least_variant_smoothed_{image_path}')
    # cv2.imshow("least variant filtered Image", smoothed_image2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return smoothed_image2

def get_frequencies(image_path):
    image = cv2.imread(image_path)
    if len(image.shape) == 3 and image.shape[2] == 3:  # If the image has 3 channels (BGR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Image is already grayscale
        gray_image = image

    unique_values = np.unique(gray_image.flatten())

    frequencies = {}
    for i in range(len(unique_values)):
        for j in range(i + 1, len(unique_values)):
            x, y = unique_values[i], unique_values[j]
            gcd = math.gcd(x, y)
            if (gcd != 1):
                if gcd in frequencies:
                    frequencies[gcd] += 1
                else:
                    frequencies[gcd] = 1
                gcd_values.add(gcd)
    frequencies = dict(sorted(frequencies.items(), key=lambda item: item[1], reverse=True))
    return frequencies



def gcd_threshold_segmentation(image):
    global thresholds
    # image = cv2.imread(image_path)
    if len(image.shape) == 3 and image.shape[2] == 3:  # If the image has 3 channels (BGR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Image is already grayscale
        gray_image = image
    vertical = []
    rows = len(gray_image)
    cols = len(gray_image[0])
    unique_values = np.unique(gray_image.flatten())

    #ALL PAIRS
    global frequencies
    for i in range(len(unique_values)):
        for j in range(i + 1, len(unique_values)):
            x, y = unique_values[i], unique_values[j]
            gcd = math.gcd(x, y)
            if(gcd!=1):
                if gcd in frequencies:
                    frequencies[gcd] += 1
                else:
                    frequencies[gcd] = 1
                gcd_values.add(gcd)
    frequencies = dict(sorted(frequencies.items(), key=lambda item: item[1], reverse=True))
    # print("Frequencies:",frequencies)
    # print("GCDs",gcd_values)


    #ITERATE VERTICALLY
    for col in range(cols):
        for row in range(rows):
            vertical.append(gray_image[row][col])
    # vertical = np.unique(vertical)


    vertical_gcds = get_gcds(vertical)
    mean_allvals = round(np.mean(list(gcd_values)),2)

    mean_vertical = round(np.mean(list(vertical_gcds)),2)
    median_vertical = round(np.median(list(vertical_gcds)),2)

    thresholds['allvalues'] = mean_allvals
    thresholds['vertical'] = mean_vertical
    # print("Threshold values for different images:", thresholds)

    return thresholds

def find_filter_based_on_thresholds(brightness):
    if brightness < 85:
        filter_A = np.array([[90, 90, 90],
                             [90, 180,90],
                             [90, 90, 90]]) / 450
        return filter_A
    elif brightness < 170:
        filter_B = np.array([[30, 60, 30],
                             [60, 100, 60],
                             [30, 60, 30]]) / 520
        return filter_B
    else:
        filter_C = np.array([[20, 30, 20],
                             [30, 95, 30],
                             [20, 30, 20]]) / 360
        return filter_C


def determine_brightness(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness_level = np.mean(gray_image)
    return brightness_level

def apply_filter_based_on_brightness(image_path):
    image = cv2.imread(image_path)
    brightness = determine_brightness(image)
    filter = find_filter_based_on_thresholds(brightness)
    smoothed_image = cv2.filter2D(image, -1, filter)
    cv2.imwrite(f"fixed_smoothed_{image_path}", smoothed_image)
    # cv2.imshow("Brightness based filtered Image", smoothed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(filter)
    return smoothed_image




def compute_iqr_thresholds(image):
    """ Compute the IQR-based threshold for adaptive filtering. """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flat_pixels = gray_image.flatten()

    q1 = np.percentile(flat_pixels, 25)
    q3 = np.percentile(flat_pixels, 75)
    iqr = q3 - q1

    lower_threshold = max(0, q1 - 1.5 * iqr)
    upper_threshold = min(255, q3 + 1.5 * iqr)

    return lower_threshold, upper_threshold


def iqr_based_smoothing(image):
    """ Apply IQR-based smoothing filter. """
    lower_threshold, upper_threshold = compute_iqr_thresholds(image)

    # Create a filter dynamically based on IQR values
    filter_matrix = np.array([
        [lower_threshold, (lower_threshold + upper_threshold) / 2, lower_threshold],
        [(lower_threshold + upper_threshold) / 2, upper_threshold, (lower_threshold + upper_threshold) / 2],
        [lower_threshold, (lower_threshold + upper_threshold) / 2, lower_threshold]
    ]) / (4 * upper_threshold)

    smoothed_image = cv2.filter2D(image, -1, filter_matrix)
    return smoothed_image



def edge_aware_filter(image):
    """ Apply edge-aware bilateral filtering. """
    filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return filtered_image


def apply_sequential_filters(image_path):
    """ Apply all six filters in a structured order before segmentation. """
    image = cv2.imread(image_path)

    # Step 1: Brightness-Based Filtering
    brightness_filtered = apply_filter_based_on_brightness(image_path)

    # Step 2: IQR-Based Smoothing
    iqr_filtered = iqr_based_smoothing(brightness_filtered)

    # Step 3: Most Frequent GCD Values Filter
    frequencies = get_frequencies(image_path)  # Use existing function
    most_common_filter = find_filter1(frequencies)
    most_common_filtered = smooth1(image_path, most_common_filter)

    # Step 4: Local Contrast Enhancement
    # contrast_enhanced = local_contrast_enhancement(most_common_filtered)

    # Step 5: Least Variant GCD Values Filter
    least_variant_filter = find_filter2()  # Use existing function
    least_variant_filtered = smooth2(image_path, least_variant_filter)

    # Step 6: Edge-Aware Filter
    final_filtered = edge_aware_filter(least_variant_filtered)
    cv2.imshow("Final Filtered Image",final_filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f"seq_final_filtered_{image_path}",final_filtered)
    return final_filtered  # Return the final filtered image


def segmentation(gimage, T, path):
    global results
    m, n = gimage.shape
    img_thresh = np.zeros((m, n), dtype=np.uint8)

    for i in range(m):
        for j in range(n):
            if gimage[i, j] < T:
                img_thresh[i, j] = 0
            else:
                img_thresh[i, j] = 255

    # cv2.imshow("Final segmented image", img_thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(f"seq_final_segmented_{path}", img_thresh)

if __name__ == "__main__":
    for i in range(1,323):
        print(i)
        image_path = f"images/mias{i}.jpg"
        image = cv2.imread(image_path)
        gimage = cv2.imread(image_path, 0)
        frequencies = get_frequencies(image_path)
        filter1 = find_filter1(frequencies)
        filter2 = find_filter2()
        brightness_filtered = apply_filter_based_on_brightness(image_path)
        iqr_filtered = iqr_based_smoothing(image)
        most_common_filtered_image = smooth1(iqr_filtered, filter1)
        least_variant_filtered_image = smooth2(most_common_filtered_image, filter2)
        final_filtered = edge_aware_filter(least_variant_filtered_image)
        cv2.imwrite(f"seq_final_filtered_{image_path}",final_filtered)
        image_array = np.array(final_filtered)
        threshold_values = gcd_threshold_segmentation(image_array)
        if len(image_array.shape) == 3 and image.shape[2] == 3:
            gimage = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gimage = image_array
        segmentation(gimage, int(threshold_values['vertical']), image_path)

