# Jason Shawn D' Souza

from skimage import data
from skimage.feature import Cascade
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

def image_plot(image, cmap_type, title='Loaded Image'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('on')
    plt.show()

def plot_comparison(original, filtered, title_filtered):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 6), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(title_filtered)
    ax2.axis('off')

def get_face_coordinates(img, input_image):
    x, y = img['r'], img['c']
    # The width and height of the face rectangle
    width, height = img['r'] + img['width'], img['c'] + img['height']
    # Extract the detected face
    face = input_image[x:width, y:height]
    return face


def original_image_processing(filename):
    input_img = plt.imread(filename)
    trained_file = data.lbp_frontal_face_cascade_filename()

    # Initialize the detector cascade.
    detector = Cascade(trained_file)


    # Detect faces with scale factor to 1.2 and step ratio to 1
    detected = detector.detect_multi_scale(img=input_img,
                                        scale_factor=1.2,
                                        step_ratio=1,
                                        min_size=(50, 50),
                                        max_size=(200, 200))
    original_img = np.copy(input_img)
    original_img = input_img/255
    i = 0
    for d in detected:
        i += 1
        # Obtain the face cropped from detected coordinates
        face = get_face_coordinates(d, input_img)
        print("Face -", i)
        plt.imshow(face)
        plt.show()
        blurred_face = gaussian(face, multichannel=True, sigma = 8)
        print("Blurred Face -", i)
        plt.imshow(blurred_face)
        plt.show()
        resulting_image = merge_with_original(d, original_img, blurred_face)
    
    return resulting_image


def merge_with_original(img, original, gaussian_image):
    x, y  = img['r'], img['c']
    width, height = img['r'] + img['width'],  img['c'] + img['height']
    
    original[ x:width, y:height] =  gaussian_image
    return original


def run_fd():
    result = original_image_processing('groupp1.jpg')
    image_plot(result, 'gray', "Blurred faces")

if __name__ == '__main__':
    run_fd()