import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from pathlib import Path

dataset_path = Path('dataset')
categories = ['easy', 'medium', 'hard']
input_folder = dataset_path / 'images'
output_folder = dataset_path / 'output'
ground_truth_folder = dataset_path / 'ground_truths'

for category in categories:
    os.makedirs(output_folder / category, exist_ok=True)


def convert_color_space(image, space='HSV'):
    if space == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Function to apply noise reduction
def apply_noise_reduction(image, method='Gaussian', kernel_size=5):
    if method == 'Gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return cv2.medianBlur(image, kernel_size)


# Function to segment the image
def segment_image(image, method='Otsu'):
    if method == 'Otsu':
        _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if method == 'Mean' else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        thresholded = cv2.adaptiveThreshold(image, 255, adaptive_method, cv2.THRESH_BINARY, 11, 2)
    return thresholded


# Function to post-process the image
def post_process_image(image):
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)


# Main function to process each image
def process_image(image_path):
    image = cv2.imread(str(image_path))
    hsv_image = convert_color_space(image)
    reduced_noise = apply_noise_reduction(hsv_image[:, :, 2])  # Using the V channel for HSV by default
    segmented = segment_image(reduced_noise)
    final_image = post_process_image(segmented)
    return final_image


# Iterate through categories and images
for category in ['easy', 'medium', 'hard']:
    cat_input_folder = input_folder / category
    cat_output_folder = output_folder / category

    for image_path in cat_input_folder.iterdir():
        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            result = process_image(image_path)
            result_binary = (result > 0).astype(np.uint8) * 255  # Convert to binary image
            output_path = cat_output_folder / (image_path.stem + '.jpg')
            cv2.imwrite(str(output_path), result_binary)
            print(f"Processed {image_path.name} and saved to {output_path.name}")


def show_image(image, title='Image', cmap='gray'):
    if image is None:
        print("Failed to load image. It's None.")
    elif image.dtype == object:
        print("Image data type is object, which indicates a problem in loading.")
    else:
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.show()


def load_and_show_example(category):
    cat_output_folder = output_folder / category

    for example_file in cat_output_folder.iterdir():
        if example_file.is_file() and example_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            example_image = cv2.imread(str(example_file), cv2.IMREAD_GRAYSCALE)
            if example_image is None:
                print(f"Could not read the image from {example_file}. Check the file path.")
            else:
                show_image(example_image, title='Segmented Image')
                return
        else:
            print(f"Skipping non-image file: {example_file}")


load_and_show_example('easy')


def calculate_iou(predicted_binary, ground_truth_binary):
    intersection = np.logical_and(predicted_binary > 0, ground_truth_binary > 0)
    union = np.logical_or(predicted_binary > 0, ground_truth_binary > 0)
    union_sum = np.sum(union)
    if union_sum == 0:
        return float('nan')  # Return NaN if the union is empty
    iou_score = np.sum(intersection) / union_sum
    return iou_score


def convert_ground_truth_to_binary(image_path):
    # Read the image
    ground_truth_color = cv2.imread(str(image_path))

    # Since the flower is red, we look for red values
    # This will create a mask where red parts are white (255)
    lower_red = np.array([0, 100, 100])  # Adjust these values
    upper_red = np.array([10, 255, 255])  # Adjust these values

    # Convert to HSV
    hsv = cv2.cvtColor(ground_truth_color, cv2.COLOR_BGR2HSV)

    # Create a mask for the red color
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # The mask might need to be inverted depending on how your GT is represented
    # If your GT shows the flower as red (and should be white in binary) you might not need to invert
    # binary_ground_truth = cv2.bitwise_not(mask) # Uncomment if you need to invert the mask

    return mask


def evaluate_segmentation(dataset_path, categories):
    iou_scores = []
    for category in categories:
        cat_input_folder = input_folder / category
        cat_ground_truth_folder = ground_truth_folder / category

        for image_path in cat_input_folder.iterdir():
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                ground_truth_path = cat_ground_truth_folder / (image_path.stem + '.png')

                if not ground_truth_path.exists():
                    print(f"Ground truth file does not exist: {ground_truth_path}")
                    continue

                predicted_binary = process_image(image_path)
                if predicted_binary is None or predicted_binary.size == 0:
                    print(f"Failed to process image or image is empty: {image_path}")
                    continue
                predicted_binary = (predicted_binary > 0).astype(np.uint8)

                # Use the function to convert the ground truth to binary
                ground_truth_binary = convert_ground_truth_to_binary(ground_truth_path)

                # Ensuring binary format
                ground_truth_binary = (ground_truth_binary > 0).astype(np.uint8)

                iou_score = calculate_iou(predicted_binary, ground_truth_binary)
                print(f"IoU score for {image_path.name}: {iou_score}")
                iou_scores.append(iou_score)

    # Handle the case when iou_scores list is empty
    if not iou_scores:
        print("No IoU scores were calculated. Returning NaN.")
        return float('nan')

    mIoU = np.mean([score for score in iou_scores if not np.isnan(score)])
    return mIoU


def show_segmentation_results(image_path, predicted_binary, ground_truth_binary):
    print(f"Displaying results for image: {image_path.name}")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    original_image = cv2.imread(str(image_path))
    # Convert BGR to RGB for displaying correctly in matplotlib
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(predicted_binary, cmap='gray')
    plt.title('Segmented Output')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(ground_truth_binary, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.show()


mIoU_score = evaluate_segmentation(dataset_path, categories)
print(f"Mean IoU for the dataset: {mIoU_score}")


# For visual inspection of random result
def display_all_segmented_results(dataset_path, categories):

    segmented_images = []

    for category in categories:
        print(f"Displaying results for category: {category}")
        images_folder_path = dataset_path / 'images' / category
        ground_truths_folder_path = dataset_path / 'ground_truths' / category
        output_folder_path = dataset_path / 'output' / category

        for image_file in images_folder_path.glob('*.jpg'):
            ground_truth_file = ground_truths_folder_path / (image_file.stem + '.png')
            segmented_file = output_folder_path / (image_file.stem + '.jpg')  # Changed to '.jpg'

            # Check if the ground truth and segmented image exists
            if ground_truth_file.exists() and segmented_file.exists():
                original = cv2.imread(str(image_file))
                segmented = cv2.imread(str(segmented_file), cv2.IMREAD_GRAYSCALE)
                ground_truth = cv2.imread(str(ground_truth_file))

                # Show segmentation results
                show_segmentation_results(image_file, segmented, ground_truth)

                # Append segmented image to the list
                segmented_images.append(segmented)

            else:
                print(f"Missing ground truth or segmented image for {image_file.name}")

    # Display all segmented images in a grid
    plt.figure(figsize=(12, 12))
    for i in range(len(segmented_images)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(segmented_images[i], cmap='gray')
        plt.title(f"Segmented Image {i + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Call the function to display all results
display_all_segmented_results(dataset_path, categories)