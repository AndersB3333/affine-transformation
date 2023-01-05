import numpy as np
import cv2
from PIL import Image

# Load the image, as a np array object
IMAGE = cv2.imread('dogs.jpeg')

# Scaling the matrix by x in height and width
SCALE_FACTOR = 1
scale_matrix = np.array([[SCALE_FACTOR, 0, 0],
                         [0, SCALE_FACTOR, 0],
                         [0, 0, 1]])

# Determining degree of angle, in this case 30 degrees
ANGLE_ROTATION = 30
angle = np.radians(ANGLE_ROTATION)

rotate_matrix = np.array([[np.cos(angle), -1 * np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]])

SHEAR_X = 0.5
SHEAR_Y = 0
# Shear matrix
shear_matrix = np.array([[1, SHEAR_X, 0],
                        [SHEAR_X, 1, 0],
                        [0, 0, 1]])

# Converting the matrices to the transformation matrix in the program
trans_matrix = np.eye(3, dtype=float)
for i in [scale_matrix, rotate_matrix, shear_matrix]:
    trans_matrix = np.dot(trans_matrix, i)

# Declaring the appropriate amount of padding, so the index won't
# exceed the array
PADDING = 1.4

image_transformed = np.empty((round(IMAGE.shape[0] * SCALE_FACTOR * PADDING), round(
    IMAGE.shape[1] * SCALE_FACTOR * PADDING), 3), dtype=np.uint8)

for i, row in enumerate(IMAGE):
    for j, col in enumerate(row):
        pixel = IMAGE[i, j, :]
        input_coordinates = np.array([i, j, 1])
        i_out, j_out, _ = np.dot(trans_matrix, input_coordinates)
        if i_out >= IMAGE.shape[0] or j_out >= IMAGE.shape[1]:
            continue
        else:
            image_transformed[int(i_out), int(j_out), :] = pixel

cv2.imwrite("transformed_image", image_transformed)
cv2.imshow("transformed image", image_transformed)
cv2.waitKey(0)
