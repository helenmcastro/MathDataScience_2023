# Plotting y=mx+b

You can use Python's Matplotlib library to plot a straight line with the equation `y = mx + b`. Here's some example code that demonstrates how to do this:

```python
import matplotlib.pyplot as plt
import numpy as np

# Define the values of m and b
m = 2  # Slope of the line
b = 3  # y-intercept

# Generate x values (e.g., from -10 to 10)
x = np.linspace(-10, 10, 100)  # 100 points between -10 and 10

# Calculate y values using the equation y = mx + b
y = m * x + b

# Create the plot
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(x, y, label=f'y = {m}x + {b}', color='blue')  # Plot the line
plt.xlabel('x')  # Label for the x-axis
plt.ylabel('y')  # Label for the y-axis
plt.title('Plot of y = mx + b')  # Title for the plot
plt.grid(True)  # Add a grid
plt.legend()  # Add a legend
plt.axhline(0, color='black',linewidth=0.5)  # Add horizontal line at y=0
plt.axvline(0, color='black',linewidth=0.5)  # Add vertical line at x=0
plt.show()  # Display the plot
```

In this code:

- We specify the values of `m` (slope) and `b` (y-intercept).

- We generate a range of `x` values using NumPy's `linspace` function to create a set of evenly spaced points.

- We calculate the corresponding `y` values using the equation `y = mx + b`.

- We create the plot using Matplotlib, specifying the x and y data for the line, labels for the axes, a title, gridlines, a legend, and more.

- Finally, we use `plt.show()` to display the plot.

You can customize the values of `m`, `b`, and the range of `x` to fit your specific requirements and equations.
