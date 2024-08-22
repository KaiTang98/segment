import os
import shutil

'''
This program can read images and labels from different source folders and write to the destination folder for training, validation and testing.
The training and validating list will be created automatically.
Please input the source folders and destination folder for training and testing.
'''


def image_read_images(source_folders, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    image_count = 0

    # Iterate through the source folders
    for folder in source_folders:
        # Get the list of files in the current folder
        files = os.listdir(folder)

        # Sort the files to maintain order
        files.sort()

        # Process each image file
        for filename in files:
            # Construct the source and destination paths
            source_path = os.path.join(folder, filename)
            destination_path = os.path.join(destination_folder, f"image_{image_count+1}.bmp")

            # Copy the image to the destination folder
            shutil.copyfile(source_path, destination_path)

            image_count += 1

    print(f"Total images processed: {image_count}")


def label_read_images(source_folders, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    image_count = 0

    # Iterate through the source folders
    for folder in source_folders:
        # Get the list of files in the current folder
        files = os.listdir(folder)

        # Sort the files to maintain order
        # files.sort(key=lambda x: int(x[:-4]))
        files.sort()

        # Process each image file
        for filename in files:
            # Construct the source and destination paths
            source_path = os.path.join(folder, filename)
            destination_path = os.path.join(destination_folder, f"label_{image_count+1}.png")

            # Copy the image to the destination folder
            shutil.copyfile(source_path, destination_path)

            image_count += 1

    print(f"Total labels processed: {image_count}")


    # Function to get all file names in a folder
def get_file_names(folder_path):
    # Get all file names in the folder
    file_names = os.listdir(folder_path)
    # sort the files
    file_names.sort()
    # Return the list of file names
    return file_names

# Function to create a text file and write the file names to it
def create_txt_file(img_folder_path, label_folder_path, save_file_path, train, train_gt, val, val_gt, test_image_path, test_label_path):
    
    # Get all file names in the folder
    img_file_names = get_file_names(img_folder_path)
    label_file_names = get_file_names(label_folder_path)

    # create training set and validation set for image and label
    train_img_file_names = []
    val_img_file_names = []
    train_label_file_names = []
    val_label_file_names = []

    for i in range(len(img_file_names)):
        if i == 10 or i % 10 == 0:
            # Append extracted item to new list
            val_img_file_names.append(img_file_names[i])
            val_label_file_names.append(label_file_names[i])
        elif i == 9 or i % 10 == 9:
            # move the image to test_image folder and label to test_label folder
            shutil.move(os.path.join(img_folder_path, img_file_names[i]), test_image_path)
            shutil.move(os.path.join(label_folder_path, label_file_names[i]), test_label_path)
        else:
            train_img_file_names.append(img_file_names[i])
            train_label_file_names.append(label_file_names[i])

    # Create a new text file in the folder
    train_save_file_path = os.path.join(save_file_path, train)
    train_gt_save_file_path = os.path.join(save_file_path, train_gt)
    val_save_file_path = os.path.join(save_file_path, val)
    val_gt_save_file_path = os.path.join(save_file_path, val_gt)
    
    # Write the file names to the text file
    with open(train_save_file_path, 'w') as f:

        for i, img_name in enumerate(train_img_file_names):
            f.write('/image/' + img_name + '\n')

    with open(train_gt_save_file_path, 'w') as f:

        for i, img_name in enumerate(train_img_file_names):
            f.write('/image/' + img_name + ' /label/' + train_label_file_names[i] + '\n')

    with open(val_save_file_path, 'w') as f:

        for i, img_name in enumerate(val_img_file_names):
            f.write('/image/' + img_name + '\n')

    with open(val_gt_save_file_path, 'w') as f:

        for i, img_name in enumerate(val_img_file_names):
            f.write('/image/' + img_name + ' /label/' + val_label_file_names[i] + '\n')

    print(f"File names written to {save_file_path}")


if __name__ == '__main__':

    # List of source folders
    image_source_folders = ["/home/kaitang/workspace/sewing2d_database/source/230815/final/image"]  
    # Destination folder for saved images
    image_destination_folder = "/home/kaitang/workspace/sewing2d_database/ufld/train/image"  

    # List of source folders, please make sure the order is the same as image_source_folders
    label_source_folders = ["/home/kaitang/workspace/sewing2d_database/source/230815/final/label"]  
    # Destination folder for saved labels
    label_destination_folder = "/home/kaitang/workspace/sewing2d_database/ufld/train/label"  

    # Collect images and labels from different folders
    image_read_images(image_source_folders, image_destination_folder)
    label_read_images(label_source_folders, label_destination_folder)

    # Create .txt list for training and validation and test data
    train = 'train.txt'
    train_gt = 'train_gt.txt'
    val = 'val.txt'
    val_gt = 'val_gt.txt'
    save_list_path = '/home/kaitang/workspace/sewing2d_database/ufld/train/list'
    test_image_path = '/home/kaitang/workspace/sewing2d_database/ufld/test/image'
    test_label_path = '/home/kaitang/workspace/sewing2d_database/ufld/test/label'

    if not os.path.exists(save_list_path):
            os.makedirs(save_list_path)
    if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)
    if not os.path.exists(test_label_path):
            os.makedirs(test_label_path)

    create_txt_file(image_destination_folder, label_destination_folder, save_list_path, 
                    train, train_gt, val, val_gt, test_image_path, test_label_path)
