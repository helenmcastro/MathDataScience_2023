# Problem Set 2

## Code Link
Notebook:https://colab.research.google.com/drive/16FhvtK1ATySmJJ-B30-KRiFAob__dv5M?usp=sharing

## Step 1: Import Necessary Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import data, color, io
import imageio as io
from skimage.transform import rescale, resize
```
Here we import necessary libraries for handling numerical operations (NumPy), plotting (Matplotlib), signal processing (SciPy), and image manipulation (SciKit-Image and ImageIO).

## Step 2: Define Plotting Function
```python
def plot(x):
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap = 'gray')
    ax.axis('off')
    fig.set_size_inches(5, 5)
    plt.show()
```
plot(x) is a custom function to visualize an image (x) with a gray colormap and without axes, ensuring clean visual outputs.

## Step 3: Load and Visualize Original Image
```python
image = io.imread("https://www.akc.org/wp-content/uploads/2018/05/Three-Australian-Shepherd-puppies-sitting-in-a-field.jpg")
plot(image)
```

Here, we retrieve and display the original image of puppies from a URL.

![image](https://github.com/helenmcastro/Images/blob/main/puppies.png?raw=true)

## Step 4: Resize Image
```python
image_r = image[131:355, 252:476, :]
plot(image_r)
```
The image is sliced to obtain a smaller, focused section, and this cropped image is visualized.

![image](https://github.com/helenmcastro/Images/blob/main/resize-puppies.png?raw=true)

## Step 5: Convert to Grayscale Using Different Methods
### Averaging Method
```python
image_g = np.mean(image_r, axis = 2)
plot(image_g)
```
The average of the RGB values of each pixel is computed and displayed as a grayscale image.

Technical Explanation: This method simply takes the mean of the RGB values for each pixel, assigning the average to the grayscale intensity.

Qualitative Explanation: This method is straightforward and computationally efficient, but it may not preserve the perceived brightness of the original image since it treats all color channels equally.

Preferences: You might prefer this for speed and simplicity when accuracy is not critical.

![image](https://github.com/helenmcastro/Images/blob/main/average.png?raw=true)

### Channel Mixing Method
```python
red_coefficient = 0.05
green_coefficient = 0.01
blue_coefficient = 0.01

image_c = (
    image_r[:,:, 0] * red_coefficient +
    image_r[:,:, 1] * green_coefficient +
    image_r[:,:, 2] * blue_coefficient
).astype(np.uint8)

plot(image_c)
```
We create a grayscale image using customized RGB coefficients to emphasize or de-emphasize certain color channels.

Technical Explanation: This method forms a weighted sum of the RGB channels. The weights (coefficients) determine the contribution of each channel to the final grayscale image.

Qualitative Explanation: This allows for a bit more control over how much each channel influences the final image and can be tuned to preserve certain features or characteristics.

Preferences: Useful when one channel contains noise or irrelevant information, or when you need to emphasize or deemphasize particular features.
You might choose this when you want control over the weights assigned to each channel, potentially due to domain-specific knowledge or particular image characteristics. 

![image](https://github.com/helenmcastro/Images/blob/main/channel-mixing.png?raw=true)

### Red Channel Method 
```python
image_red = image_r[:,:,0]
plot(image_red)
```
Only the red channel is displayed, ignoring green and blue channels.
Technical Explanation: This method takes only the red channel to form the grayscale image.

Qualitative Explanation: This may preserve or highlight features that are primarily visible in the red channel, but neglects information in the green and blue channels.

Preferences: Suitable when important details are mostly contained in the red channel or for certain scientific applications where the red channel is of primary interest.
This might be your go-to when processing images where the information of interest is predominantly in the red channel (for instance, some astronomical images).

![image](https://github.com/helenmcastro/Images/blob/main/red-channel.png?raw=true)

### Luminance Method
```python
image_l = np.dot(image_r[:, :, :3], [0.299, 0.587, 0.114]).astype(np.uint8)
plot(image_l)
```
A weighted sum of the RGB channels, approximating human perception of color intensity, is computed and displayed.

Technical Explanation: This method also uses a weighted sum of the RGB channels but uses coefficients that are generally agreed upon to align well with human perception of brightness.

Qualitative Explanation: This approach often provides results that are aesthetically pleasing and perceptually similar to the original image in terms of brightness and contrast.

Preferences: When a visually accurate representation of the original image's brightness and contrast is required, such as in professional photography or media production.
Often preferred for a perceptually meaningful conversion, providing grayscale images that retain the luminosity characteristics of the original.

![image](https://github.com/helenmcastro/Images/blob/main/luminance.png?raw=true)

## Step 6: Apply Random Filters
```python
for i in range(10):
  filter_i = np.random.random((i+1,i+1))
  image_i = signal.convolve2d(image_g, filter_i, mode='same')
  print(f"Filter: {i + 1}")
  plot(filter_i)
  print(f"Image: {i + 1}")
  plot(image_i)
```
Ten randomly generated filters of increasing size are applied to the grayscale image, and the resultant filtered images are displayed alongside the random filters.

### Filter 1

Feature Map:

![image](https://github.com/helenmcastro/Images/blob/main/image1.png?raw=true)


Filter:

![image](https://github.com/helenmcastro/Images/blob/main/filter1.png?raw=true)

### Filter 2

Feature Map:

![image](https://github.com/helenmcastro/Images/blob/main/image2.png?raw=true)



Filter:

![image](https://github.com/helenmcastro/Images/blob/main/filter2.png?raw=true)


### Filter 3

Feature Map:

![image](https://github.com/helenmcastro/Images/blob/main/image3.png?raw=true)



Filter:

![image](https://github.com/helenmcastro/Images/blob/main/filter3.png?raw=true)

### Filter 4

Feature Map:

![image](https://github.com/helenmcastro/Images/blob/main/image4.png?raw=true)


Filter:

![image](https://github.com/helenmcastro/Images/blob/main/filter4.png?raw=true)


### Filter 5

Feature Map:

![image](https://github.com/helenmcastro/Images/blob/main/image5.png?raw=true)



Filter:

![image](https://github.com/helenmcastro/Images/blob/main/filter5.png?raw=true)


### Filter 6

Feature Map:

![image](https://github.com/helenmcastro/Images/blob/main/image6.png?raw=true)



Filter:

![image](https://github.com/helenmcastro/Images/blob/main/filter6.png?raw=true)


### Filter 7

Feature Map:

![image](https://github.com/helenmcastro/Images/blob/main/image7.png?raw=true)



Filter:

![image](https://github.com/helenmcastro/Images/blob/main/filter7.png?raw=true)


### Filter 8

Feature Map:

![image](https://github.com/helenmcastro/Images/blob/main/image8.png?raw=true)



Filter:

![image](https://github.com/helenmcastro/Images/blob/main/filter8.png?raw=true)

### Filter 9

Feature Map:

![image](https://github.com/helenmcastro/Images/blob/main/image9.png?raw=true)


Filter:

![image](https://github.com/helenmcastro/Images/blob/main/filter9.png?raw=true)


### Filter 10

Feature Map:

![image](https://github.com/helenmcastro/Images/blob/main/image10.png?raw=true)


Filter:

![image](https://github.com/helenmcastro/Images/blob/main/filter10.png?raw=true)



## Step 7: Apply Hand-Picked Filters
### Mean Removal Filter
```python
mean_removal = np.array([[-1, -1, -1],
                         [-1,  9, -1],
                         [-1, -1, -1]])
image_mean_removal = signal.convolve2d(image_g, mean_removal, mode='same')
plot(image_mean_removal)
```
The mean removal filter enhances edges in the image, adding depth and detail.

Explanation: Mean removal, or high-pass filtering, amplifies or enhances the edges and fine-grained details in the image while suppressing low-frequency components.

Use Case: Employed to accentuate details in the image, making edges and textures more pronounced, which might be particularly useful in scenarios where detail extraction or pattern recognition is essential.

![image](https://github.com/helenmcastro/Images/blob/main/mean-removal.png?raw=true)

### Unsharp Masking Filter
```python
unsharp_masking = np.array([[-1, -2, -1], [-2, 28, -2], [-1, -2, -1]])
image_unsharp_masking = signal.convolve2d(image_g, unsharp_masking, mode='same')
plot(image_unsharp_masking)
```
Unsharp masking sharpens the image, boosting the high-frequency components.

Explanation: This filter tends to sharpen the image by enhancing the edge contrast. By amplifying the differences around edges, the image tends to appear crisper.

Use Case: Utilized in scenarios where clarity and definition are pivotal, like in medical imaging, to aid in the clear visualization of edges, boundaries, and fine details.

![image](https://github.com/helenmcastro/Images/blob/main/unsharp.png?raw=true)

### Gaussian Filter
```python
gaussian_blur = np.array([[1, 2, 1], 
                          [2, 4, 2], 
                          [1, 2, 1]]) / 16
image_gaussian_blur = signal.convolve2d(image_g, gaussian_blur, mode='same')
plot(image_gaussian_blur)
```
Gaussian blur smoothens the image, useful for noise reduction.

Explanation: Gaussian blur is a low-pass filter that suppresses high-frequency components (like noise and fine details) in the image, providing a smoothing or blurring effect.

Use Case: Ideal for noise reduction, it is commonly utilized in scenarios where the smoother gradient transitions and subdued details are preferred, such as in pre-processing steps for object segmentation.

![image](https://github.com/helenmcastro/Images/blob/main/Guassanin.png?raw=true)

### Emboss Filter
```python
emboss = np.array([[-2, -1, 0], 
                   [-1, 1, 1], 
                   [0, 1, 2]])
image_emboss = signal.convolve2d(image_g, emboss, mode='same')
plot(image_emboss)
```
The emboss filter gives a 3D shadow effect, highlighting changes in color and rounding edges.

Explanation: The emboss filter creates a three-dimensional relief effect on the image, by highlighting edges and creating shadows, thereby adding a sense of depth.

Use Case: Often used in graphic design and art for stylistic purposes, or to uncover and explore texture-related details in scientific and medical imaging.

![image](https://github.com/helenmcastro/Images/blob/main/emboss.png?raw=true)

### Sobel Filter
```python
sobel_x = np.array([[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]])
image_sobel_x= signal.convolve2d(image_g, sobel_x, mode='same')
plot(image_sobel_x)
```
The Sobel filter emphasizes the edges, often used for edge detection.

Explanation: The Sobel filter is designed to perform edge detection, enhancing regions where there is a rapid change in intensity. The provided Sobel_x kernel is particularly sensitive to vertical edges.

Use Case: A staple in computer vision applications where edge information is crucial, like in object detection, segmentation, or applications where the identification of patterns, boundaries, or transitions is paramount.

![image](https://github.com/helenmcastro/Images/blob/main/sobel.png?raw=true)
