import math
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def images_square_grid(images):
    """
    Save images as a square grid
    :param images: Images to be used for the grid
    :return: Image of images in a square grid
    """
    # Get maximum size for square grid of images
    save_size = math.floor(np.sqrt(images.shape[0]))
    
    # Scale to 0-255
    images = ((images + 1.0) * 127.5).astype(np.uint8)
    
    # Put images in a square arrangement
    images_in_square = np.reshape(
            images[:save_size*save_size],
            (save_size, save_size, images.shape[1], images.shape[2], images.shape[3]))

    # Combine images to grid image
    new_im = Image.new('RGB', (images.shape[1] * save_size, images.shape[2] * save_size))
    for col_i, col_images in enumerate(images_in_square):
        for image_i, image in enumerate(col_images):
            im = Image.fromarray(image, 'RGB')
            new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))

    return new_im

def output_fig(images_array, file_name="./results"):
    # the shape of your images_array should be (9, width, height, 3),  28 <= width, height <= 112 
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(images_square_grid(images_array))
    plt.axis("off")
    plt.savefig(file_name+'.png', bbox_inches='tight', pad_inches=0)
    plt.show()

def get_image(image_path, width, height, mode):
    """
        Read image from image_path
        :param image_path: Path of image
        :param width: Width of image
        :param height: Height of image
        :param mode: Mode of image
        :return: Image data
        """
    image = Image.open(image_path)
    
    if image.size != (width, height):  # HACK - Check if image is from the CELEBA dataset
        # Remove most pixels that aren't part of a face
        face_width = face_height = 108
        j = (image.size[0] - face_width) // 2
        i = (image.size[1] - face_height) // 2
        image = image.crop([j, i, j + face_width, i + face_height])
        image = image.resize([width, height], Image.BILINEAR)
    
    return np.array(image.convert(mode))


def get_batch(image_files, width, height, mode):
    data_batch = np.array(
                          [get_image(sample_file, width, height, mode) for sample_file in image_files]).astype(np.float32)
    # Make sure the images are in 4 dimensions
    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))

    return data_batch
