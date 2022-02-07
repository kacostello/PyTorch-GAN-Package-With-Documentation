## Data Input

* Accept numerical/tabular data: numpy matrix (2d) (rows = observations, columns = attributes)

* If labeled with a class (0,n), another numpy array (1d) (integer index of the class, ie: 1, 2, 0, 3, ....). The size of this array must equal the number of rows in the matrix above

* Accept image data: single channel (grayscale) numpy tensor (3d) (pixels in the x-axis of image, pixels in the y-axis of image, index of images)

* If labeled with a class (0, n), another numpy array (1d) (integer index of the class, ie: 1, 2, 0, 3, ....). The size of this array must equal the size of the 3rd dimension in the tensor above.
