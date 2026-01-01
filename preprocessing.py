# Program to pre-process facial images into ELA, Noise Residuals, and DCT coefficients


# input_root is the folder where the facial images are stored
# output_root is the folder where the processed features will be stored


import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


# Function to convert facial images into ELA images with 85% compression
def ela_image(input_root, output_root, quality=85):
    try:

        for root, dirs, files in os.walk(input_root):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg')):
                    input_path = os.path.join(root, file)

                    relative_path = os.path.relpath(root, input_root)
                    output_dir = os.path.join(output_root, relative_path)
                    os.makedirs(output_dir, exist_ok=True)


                    basename, ext = os.path.splitext(file)
                    parts = basename.rsplit('_', 1)
                    new_basename = f"{parts[0]}_ELA_{parts[1]}"
                    output_filename = f"{new_basename}{ext}"
                    ela_path = os.path.join(output_dir, output_filename)

                    original = Image.open(input_path)

                    temp_path = os.path.join(output_dir, 'temp.jpg')
                    original.save(temp_path, 'JPEG', quality=quality)
                    resaved = Image.open(temp_path)
                    os.remove(temp_path)

                    ela = ImageChops.difference(original, resaved)
                    max_diff = max([ex[1] for ex in ela.getextrema()])
                    scale = 255.0 / max_diff if max_diff != 0 else 0
                    ela = ImageEnhance.Brightness(ela).enhance(scale)

                    ela.save(ela_path)

                    print(f"Processed: {input_path} -> {ela_path}")

    except Exception as e:
        print(f" Error during processing: {str(e)}")



def save_srm_image(processed_image, original_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.splitext(os.path.basename(original_path))[0]
    parts = filename.rsplit('_', 1)

    if len(parts) == 2:
        new_filename = f"{parts[0]}_SRM_{parts[1]}.jpg"
    else:
        new_filename = f"{filename}_SRM.jpg"

    output_path = os.path.join(output_dir, new_filename)

    if processed_image.shape[-1] == 3:
        output_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
    else:
        output_image = processed_image

    cv2.imwrite(output_path, output_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return output_path

def apply_srm_filters(image, srm_weights):
    steps = srm_weights.shape[-1]
    filtered_channels = []

    for step in range(steps):
        filtered_channels_step = []
        for channel in range(3):
            filter_kernel = srm_weights[:, :, 0, step]
            filtered_image = cv2.filter2D(image[:, :, channel], -1, filter_kernel)
            filtered_channels_step.append(filtered_image)

        combined_residual = np.stack(filtered_channels_step, axis=-1)
        filtered_channels.append(combined_residual)

    combined_residual = np.mean(filtered_channels, axis=0)
    combined_residual = np.clip(combined_residual, -3, 3)

    combined_residual = ((combined_residual - combined_residual.min()) /
                        (combined_residual.max() - combined_residual.min())) * 255.0

    return combined_residual.astype(np.uint8)

def process_srm_image(image_path, srm_weights, output_dir=None):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Warning: Could not load image from {image_path}")
        return None

    srm_processed = apply_srm_filters(image, srm_weights)

    if output_dir is not None:
        output_path = save_srm_image(srm_processed, image_path, output_dir)
        return output_path

    return srm_processed

# Function to generate noise residuals of the facial images
# srm_weights will be loaded from SRM_Kernels.npy
def process_directory(input_root, output_root, srm_weights):
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                input_path = os.path.join(root, file)

                relative_path = os.path.relpath(root, input_root)
                output_dir = os.path.join(output_root, relative_path)

                output_path = process_srm_image(input_path, srm_weights, output_dir)
                if output_path:
                    print(f"Processed: {input_path} -> {output_path}")


# Function to generate the top 16x16 DCT coefficients of the facial images
def dct_coeffs(input_root, output_root):
    try:

        for root, dirs, files in os.walk(input_root):
            for file in files:

                if file.lower().endswith(('.jpg', '.jpeg')):
                    input_path = os.path.join(root, file)

                    relative_path = os.path.relpath(root, input_root)
                    output_dir_visual = os.path.join(output_root_visual, relative_path)
                    output_dir_coeffs = os.path.join(output_root_coeffs, relative_path)

                    os.makedirs(output_dir_visual, exist_ok=True)
                    os.makedirs(output_dir_coeffs, exist_ok=True)

                    basename, ext = os.path.splitext(file)
                    parts = basename.rsplit('_', 1)

                    if len(parts) == 2:
                        new_basename_visual = f"{parts[0]}_DCT_{parts[1]}"
                        new_basename_coeffs = f"{parts[0]}_COEFF_{parts[1]}"
                    else:

                        new_basename_visual = f"{basename}_DCT"
                        new_basename_coeffs = f"{basename}_COEFF"

                    visual_output_path = os.path.join(output_dir_visual, f"{new_basename_visual}{ext}")
                    coeffs_output_path = os.path.join(output_dir_coeffs, f"{new_basename_coeffs}.npy")

                    img_bgr = cv2.imread(input_path)
                    img_width = 128
                    img_height = 128
                    img_dimensions = (img_width,img_height)
                    img_bgr = cv2.resize(img_bgr,img_dimensions)
                    if img_bgr is None:
                        print(f"Warning: Could not read image, skipping: {input_path}")
                        continue

                    dct_magnitudes = []
                    for i in range(3):
                        channel = np.float32(img_bgr[:, :, i])
                        dct = cv2.dct(channel)
                        dct_magnitudes.append(np.abs(dct))
                    combined_dct_coeffs = np.mean(dct_magnitudes, axis=0)
                    combined_dct_coeffs = combined_dct_coeffs[:16,:16]
                   # log_scaled_dct = 20 * np.log(np.maximum(combined_dct_coeffs, 1e-10))
                   # final_dct_image = cv2.normalize(log_scaled_dct, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                   # cv2.imwrite(visual_output_path, final_dct_image)

                    np.save(coeffs_output_path, combined_dct_coeffs)

                    print(f"✅ Processed: {file}")
                    print(f"  -> Coeffs saved to: {coeffs_output_path}")
                    '''
                    # --- 5. Display Results ---
                    # Convert BGR to RGB for display
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    # Create figure with 3 subplots
                    plt.figure(figsize=(15, 5))

                    # Show original image
                    plt.subplot(1, 3, 1)
                    plt.imshow(img_rgb)
                    plt.title('Original Image')
                    plt.axis('off')

                    # Show DCT result
                    plt.subplot(1, 3, 2)
                    plt.imshow(final_dct_image, cmap='gray')
                    plt.title('DCT Result')
                    plt.axis('off')


                    plt.subplot(1, 3, 3)
                    plt.hist(final_dct_image.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.7)
                    plt.title('DCT Coefficients Histogram')
                    plt.xlabel('Coefficient Value')
                    plt.ylabel('Frequency')
                    plt.grid(True, linestyle='--', alpha=0.5)

                    plt.tight_layout()
                    plt.show()
                    '''

    except Exception as e:
        print(f" Error during processing: {str(e)}")
