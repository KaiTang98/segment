from PIL import Image
import os

def process_images(input_folder, output_folder):
    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')]

    for filename in image_files:
        # Read the image from the input folder
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)

        # Convert image to desired format (e.g., change all non-zero values to 1)
        image = image.point(lambda x: 1 if x != 0 else 0)

        # Save the image to the output folder
        output_path = os.path.join(output_folder, filename)
        image.save(output_path)

# Example usage
input_folder = 'C:/Users/ktang/workspace/sewing2d_database/ufld/train/label'
output_folder = 'C:/Users/ktang/workspace/sewing2d_database/ufld/train/label2'
process_images(input_folder, output_folder)