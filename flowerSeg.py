import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path


def apply_noise_reduction(image, method='Gaussian', kernel_size=5):
    if method == 'Gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return cv2.medianBlur(image, kernel_size)


def convert_color_space(image, space='HSV'):
    if space == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def segment_image(image, method='Otsu'):
    if method == 'Otsu':
        _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if method == 'Mean' else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        thresholded = cv2.adaptiveThreshold(image, 255, adaptive_method, cv2.THRESH_BINARY, 11, 2)
    return thresholded


def post_process_image(image):
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)


class ImageSegmentation:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.categories = ['easy', 'medium', 'hard']
        self.input_folder = dataset_path / 'images'
        self.output_folder = dataset_path / 'output'
        self.ground_truth_folder = dataset_path / 'ground_truths'

        self.setup_output_folders()

    def setup_output_folders(self):
        for category in self.categories:
            os.makedirs(self.output_folder / category, exist_ok=True)

    def process_image(self, image_path):
        image = cv2.imread(str(image_path))
        hsv_image = convert_color_space(image)
        reduced_noise = apply_noise_reduction(hsv_image[:, :, 2])  # Using the V channel for HSV by default
        segmented = segment_image(reduced_noise)
        final_image = post_process_image(segmented)
        return final_image

    def load_and_show_example(self, category):
        cat_output_folder = self.output_folder / category

        for example_file in cat_output_folder.iterdir():
            if example_file.is_file() and example_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                example_image = cv2.imread(str(example_file), cv2.IMREAD_GRAYSCALE)
                if example_image is None:
                    print(f"Could not read the image from {example_file}. Check the file path.")
                else:
                    self.show_image(example_image, title='Segmented Image')
                    return
            else:
                print(f"Skipping non-image file: {example_file}")

    def show_image(self, image, title='Image', cmap='gray'):
        if image is None:
            print("Failed to load image. It's None.")
        elif image.dtype == object:
            print("Image data type is object, which indicates a problem in loading.")
        else:
            plt.imshow(image, cmap=cmap)
            plt.title(title)
            plt.axis('off')
            plt.show()

    def evaluate_segmentation(self):
        iou_scores = []
        for category in self.categories:
            cat_input_folder = self.input_folder / category
            cat_ground_truth_folder = self.ground_truth_folder / category

            for image_path in cat_input_folder.iterdir():
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    ground_truth_path = cat_ground_truth_folder / (image_path.stem + '.png')

                    if not ground_truth_path.exists():
                        print(f"Ground truth file does not exist: {ground_truth_path}")
                        continue

                    predicted_binary = self.process_image(image_path)
                    if predicted_binary is None or predicted_binary.size == 0:
                        print(f"Failed to process image or image is empty: {image_path}")
                        continue
                    predicted_binary = (predicted_binary > 0).astype(np.uint8)

                    ground_truth_binary = self.convert_ground_truth_to_binary(ground_truth_path)
                    ground_truth_binary = (ground_truth_binary > 0).astype(np.uint8)

                    iou_score = self.calculate_iou(predicted_binary, ground_truth_binary)
                    print(f"IoU score for {image_path.name}: {iou_score}")
                    iou_scores.append(iou_score)

        if not iou_scores:
            print("No IoU scores were calculated. Returning NaN.")
            return float('nan')

        mIoU = np.mean([score for score in iou_scores if not np.isnan(score)])
        return mIoU

    def calculate_iou(self, predicted_binary, ground_truth_binary):
        intersection = np.logical_and(predicted_binary > 0, ground_truth_binary > 0)
        union = np.logical_or(predicted_binary > 0, ground_truth_binary > 0)
        union_sum = np.sum(union)
        if union_sum == 0:
            return float('nan')
        iou_score = np.sum(intersection) / union_sum
        return iou_score

    def convert_ground_truth_to_binary(self, image_path):
        ground_truth_color = cv2.imread(str(image_path))
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        hsv = cv2.cvtColor(ground_truth_color, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_red, upper_red)
        return mask

    def display_all_segmented_results(self):
        for category in self.categories:
            print(f"Displaying results for category: {category}")
            images_folder_path = self.input_folder / category
            ground_truths_folder_path = self.ground_truth_folder / category
            output_folder_path = self.output_folder / category

            for image_file in images_folder_path.glob('*.jpg'):
                ground_truth_file = ground_truths_folder_path / (image_file.stem + '.png')
                segmented_file = output_folder_path / (image_file.stem + '.jpg')

                if ground_truth_file.exists() and segmented_file.exists():
                    original = cv2.imread(str(image_file))
                    segmented = cv2.imread(str(segmented_file), cv2.IMREAD_GRAYSCALE)
                    ground_truth = cv2.imread(str(ground_truth_file))

                    self.show_segmentation_results(image_file, segmented, ground_truth)

                else:
                    print(f"Missing ground truth or segmented image for {image_file.name}")

    def show_segmentation_results(self, image_path, predicted_binary, ground_truth_binary):
        print(f"Displaying results for image: {image_path.name}")
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        original_image = cv2.imread(str(image_path))
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
        plt.close()  # Close the current window after displaying the image


def main():
    dataset_path = Path('dataset')
    segmentation = ImageSegmentation(dataset_path)
    mIoU_score = segmentation.evaluate_segmentation()
    print(f"Mean IoU for the dataset: {mIoU_score}")
    segmentation.display_all_segmented_results()


if __name__ == "__main__":
    main()
