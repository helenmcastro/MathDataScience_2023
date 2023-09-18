# Simple Line Plots

```import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 50)
y = np.sin(x)

# plt.plot(x, y, 'bo', label='Circle')
plt.plot(x, y + 0.5, 'r^', label='Triangle')
# plt.plot(x, y - 0.5, 'gs', label='Square')
# plt.plot(x, y - 1.0, 'kd', label='Diamond')
# plt.plot(x, y - 1.5, 'mp', label='Pentagon')
# plt.plot(x, y - 2.0, 'cH', label='Hexagon')
# plt.plot(x, y - 2.5, 'y+', label='Plus')
# plt.plot(x, y - 3.0, 'x', label='Cross')
# plt.plot(x, y - 3.5, 'r.', label='Point')
```

* Save this for later maybe plt.legend(). Add a legend to identify marker types.

The code you've provided uses the NumPy and Matplotlib libraries to create a simple line plot. Here's an explanation of the code:

1. `import numpy as np`: Imports the NumPy library and gives it the alias `np`.

2. `import matplotlib.pyplot as plt`: Imports the `pyplot` module from Matplotlib and gives it the alias `plt`.

3. `x = np.linspace(0, 10, 50)`: Generates an array of 50 evenly spaced values between 0 and 10 and assigns it to the variable `x`.

4. `y = np.sin(x)`: Calculates the sine of each element in the `x` array and assigns it to the variable `y`.

5. `plt.plot(x, y + 0.5, 'r^', label='Triangle')`: Plots the data using Matplotlib. It creates a line plot with red triangle markers ('r^') and labels it as 'Triangle'.

6. The other `plt.plot` lines are commented out. They are similar to the first line but use different marker styles (e.g., 'bo' for blue circles, 'gs' for green squares, etc.). You can uncomment these lines to see the different marker styles or save them for later use.

7. The line `plt.legend()` is also commented out. This line would be used to add a legend to the plot to identify the different marker types. You can uncomment it when you want to include a legend in your plot.

To visualize the plot, you can run the code with the desired plot style uncommented. For example, if you want to see the plot with red triangles, you can remove the comment from the line `plt.plot(x, y + 0.5, 'r^', label='Triangle')`, and then run the code.
