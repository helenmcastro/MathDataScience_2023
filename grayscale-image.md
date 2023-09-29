# Grayscale image using Matplotlib

```import numpy as np
import matplotlib.pyplot as plt

# Define the dimensions of the image (e.g., 100x100)
width, height = 100, 100

# Generate random grayscale pixel values between 0 and 255
random_image = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)

# Create a Matplotlib figure and axis
plt.figure(figsize=(6, 6))
plt.imshow(random_image, cmap='gray', vmin=0, vmax=255)
plt.axis('off')  # Hide the axis
plt.title('Random Grayscale Image')
plt.show()
```


Here's a breakdown of what each part of the code does:

- We specify the dimensions of the image as `width` and `height`.

- We use `np.random.randint(0, 256, size=(height, width), dtype=np.uint8)` to generate random grayscale pixel values between 0 and 255. The `dtype=np.uint8` ensures that the pixel values are stored as 8-bit unsigned integers.

- We create a Matplotlib figure, use `plt.imshow()` to display the random image, specify the colormap as `'gray'` to ensure it's treated as a grayscale image, and set the `vmin` and `vmax` to 0 and 255, respectively, to ensure that the full range of grayscale values is used.

- We use `plt.axis('off')` to hide the axis, add a title to the plot, and finally, display the image with `plt.show()`.

Running this code will generate and display a random grayscale image with the specified dimensions.
