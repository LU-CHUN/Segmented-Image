
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image using OpenCV
image_path = 'C:/Users/52748/Desktop/benben.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixels = image.reshape((-1, 3))


pixels = np.float32(pixels)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)


segmented_image = centers[labels.flatten()]

segmented_image = segmented_image.reshape(image.shape)


plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.subplot(122)
plt.imshow(segmented_image)
plt.title('Segmented Image')
plt.axis('off')
plt.show()
