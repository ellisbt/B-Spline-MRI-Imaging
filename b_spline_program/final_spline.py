import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mayavi.mlab as my
from tvtk.api import tvtk
import math

# References
# https://stackoverflow.com/questions/19349904/displaying-true-colour-2d-rgb-textures-in-a-3d-plot
# https://stackoverflow.com/questions/34803197/fast-b-spline-algorithm-with-numpy-scipy
# https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/PARA-knot-generation.html
# https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
# http://www2.cs.uregina.ca/~anima/408/Notes/Interpolation/UniformBSpline.htm

# Imports for CSV file and reading into different arrays.
csv = np.genfromtxt(r'smooth_path.csv', delimiter=",", dtype=float)
x_axis = csv[:, 0]
y_axis = csv[:, 1]
z_axis = csv[:, 2]

# Import radius from CSV
csv2 = np.genfromtxt(r'GCorridor.csv', delimiter=",", dtype=float)
radius_data = csv[:, 1]

# Algorithm to find the control points
# coord is the array of values, u is the weight and index is the location in array


def find_c_points(coord, u, index):
    p1 = (((1 - u) ** 3) / 6) * coord[index]
    p2 = (((3 * u ** 3) - (6 * u ** 2) + 4) / 6) * coord[index + 1]
    p3 = (((-3 * u ** 3) + (3 * u ** 2) + (3 * u + 1)) / 6) * coord[index + 2]
    p4 = (u ** 3 / 6) * coord[index + 4]
    p5 = p1 + p2 + p3 + p4
    return p5

# initialize empty arrays


arr_count = 0
control_vector_x = []
control_vector_y = []
control_vector_z = []

# Runs the for loop in range of the control points, doing one axis at a time and a weight of 0.2
for position in range(1735):
    control_vector_x.append(find_c_points(x_axis, 0.2, position))
    control_vector_y.append(find_c_points(y_axis, 0.2, position))
    control_vector_z.append(find_c_points(z_axis, 0.2, position))

# cv is one large array containing all the control points.
cv = np.empty((1735, 3))
for j in range(1735):
    cv[j] = control_vector_x[j], control_vector_y[j], control_vector_z[j]


# degree is taken from the function and num is the number of control points
def find_knots(degree, num):

    # This find uniformly spaced knots with padding at the end
    # The first loop adds 0 to the first 0-degree of knots, the send loops is the uniform space and the last loop
    # is the padding for the last four knots.
    # algorithm similar to (https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/PARA-knot-generation.html)
    m = num + degree + 1
    knot = []
    k = 0
    # Padding the first p=degree values
    for a in range(degree):
        knot.append(0)
    for i in range(degree + 1, num + 1):
        knot.append(k)
        k = k + 1
    # m is defined as number of control points + degree + 1
    # Padding for the last m+1 values = number of control points-3+1 to control points +1
    for b in range(m - degree, m + 1):
        knot.append(num - degree)
    return knot


# x_t is the query range and is defined as element belonging to (ti, ti+k)
# t_i is the index of the knot
# knot is passing the knot vector
# degree is the degree of the function
def cox_de_boors(x_t, t_i, degree, knot):
    # Implemented (https://en.wikipedia.org/wiki/De_Boor%27s_algorithm) under the local support section
    # Local Support

    if degree == 0:
        if knot[t_i] <= x_t < knot[t_i + 1]:
            return 1
        else:
            return 0
    # Check DeBoors recursion for 0 Denominator (A/B * Recursion + C/D * Recursion)
    equation_a = 0
    equation_b = 0
    b = knot[t_i + degree] - knot[t_i]
    d = knot[t_i + degree + 1] - knot[t_i + 1]
    if b > 0:
        # if the denominator is not zero we follow the Cox-deBoors formula
        equation_a = ((x_t - knot[t_i]) / b) * cox_de_boors(x_t, t_i, degree - 1, knot)
    if d > 0:
        # if the denominator is not zero we follow the Cox-deBoors formula
        equation_b = ((knot[t_i + degree + 1] - x_t) / d) * cox_de_boors(x_t, t_i + 1, degree - 1, knot)
    # Finally we combine both equations or add 0 to one of the equation or return 0
    final_eq = equation_a + equation_b
    return final_eq


def b_spline(c_points, n=1739, degree=3, periodic=False):
    # Create a range of u values
    num_of_c_points = len(c_points)

    # function to find knots
    knots = find_knots(degree, num_of_c_points)

    # Calculate query range
    u = np.linspace(periodic, (num_of_c_points - degree), n)

    # Sample the curve at each u value
    segment_data = np.zeros((n, 3))
    for i in range(n):
        if not periodic:
            # returns the cox-De Boors recursion for the control point and saves them in segment data
            # The return of recursion is multiplies by control point to get the value of the spline
            # Final the multiplication is save in segment data
            if u[i] == num_of_c_points - degree:
                segment_data[i] = np.array(c_points[-1])
            else:
                for k in range(num_of_c_points):
                    segment_data[i] += cox_de_boors(u[i], k, degree, knots) * c_points[k]
    return segment_data


p = b_spline(cv, periodic=False)
x, y, z = p.T

# This is the function for calculate the RMS as the professor said during the last day of class


def percent_error():
    p = round((len(x_axis)/len(x)))
    de = 0
    i = 0
    for j in range(0, len(x_axis), p):
        de = (x[i]-x_axis[j])**2 + (y[i]-y_axis[j])**2 + (z[i] - z_axis[j])**2 + de
        i = i+1
    de1 = math.sqrt(de/len(x))
    return de1


print('Percentage Error: ', percent_error())

# surface data
fig = plt.figure()
# Red is the spline and Blue is the smooth_path.csv
ax = fig.add_subplot(111, projection='3d')
Axes3D.plot(ax, xs=x, ys=y, zs=z, color='red')
Axes3D.plot(ax, xs=x_axis, ys=y_axis, zs=z_axis, color='blue')
# Purple is the control points
Axes3D.scatter3D(ax, xs=control_vector_x, ys=control_vector_y, zs=control_vector_z, s=0.25, color='purple')
plt.show()


# s contains all the radius data
# tube radius is varied by a factor 0.1 * radius for a cleaner 3D output
s = radius_data / 2

user_using = True
while user_using == True:
    # User input for vessel representation
    print()
    print('Vessel representations:')
    print('[1]: Given Smooth Path')
    print('[2]: B-Spline Values')
    vessel_type = input('Select which representation of the vessel you would like to use: ')

    # Error check user input
    while int(vessel_type) > 2 or int(vessel_type) < 1:
        print('ERROR: Invalid Choice')
        vessel_type = input('Select which representation of the vessel you would like to use: ')

    # Assign plot values based on user choice
    if vessel_type == '1':
        x_plot = x_axis
        y_plot = y_axis
        z_plot = z_axis

    if vessel_type == '2':
        x_plot = x
        y_plot = y
        z_plot = z

    # User input for splice images
    print()
    print('Slice Representations: ')
    print('[1]: Single slice')
    print('[2]: Multiple individual slices')
    print('[3]: Slice Range')
    print('[4]: No slices, vessel only')

    # Used to get MRI to overlay on the 3D Spline
    j = 0

    slice_type = input('Input what representation type of slices you would like to see: ')
    # Error check user input
    while int(slice_type) > 4 or int(slice_type) < 1:
        print('ERROR: Invalid Choice')
        slice_type = input('Input what representation type of slices you would like to see: ')

    if slice_type == '1':
        slice_num = int(input('Input single slice number(0-143): '))

        # Error check user input
        while slice_num < 0 or slice_num > 143:
            print('ERROR: Invalid input')
            slice_num = int(input('Input single slice number(0-143): '))

        # 3D Plot with radius data
        t = my.plot3d(x_plot, y_plot, z_plot, s, tube_radius=0.1, color=(1, 0, 0))
        t.parent.parent.filter.vary_radius = 'vary_radius_by_scalar'

        # Assign posistion based on ratio of slice images to total points along the path
        j = int(slice_num * 11.805556666)
        # Reads the image data into and variable
        im = plt.imread('test' + str(slice_num) + '.png', format='png') * 255
        colors = tvtk.UnsignedCharArray()
        # converts the PNG color data to variables Mayavi could understand
        colors.from_array(im.transpose((1, 0, 2)).reshape(-1, 3))
        m_image = my.imshow(np.ones(im.shape[:2]))
        # Sets the color data on a 2D plane in 3D space
        m_image.actor.input.point_data.scalars = colors
        # position in 3D space of the slice
        m_image.actor.position = [x_axis[j], y_axis[j], z_axis[j]]


    if slice_type == '2':
        slice_num = input('Input slice numbers(0-143) separated by a space: ')
        slice_num_list = slice_num.split()

        # Error check user input
        while int(min([int(i) for i in slice_num_list])) < 0 or int(max([int(i) for i in slice_num_list])) > 143:
            print('ERROR: Invalid input')
            slice_num = input('Input slice numbers(0-143) separated by a space: ')
            slice_num_list = slice_num.split()

        # 3D Plot with radius data
        t = my.plot3d(x_plot, y_plot, z_plot, s, tube_radius=0.1, color=(1, 0, 0))
        t.parent.parent.filter.vary_radius = 'vary_radius_by_scalar'

        for k in slice_num_list:
            # Assign posistion based on ratio of slice images to total points along the path
            j = int(int(k) * 11.805556666)
            # Reads the image data into and variable
            im = plt.imread('test' + str(k) + '.png', format='png') * 255
            colors = tvtk.UnsignedCharArray()
            # converts the PNG color data to variables Mayavi could understand
            colors.from_array(im.transpose((1, 0, 2)).reshape(-1, 3))
            m_image = my.imshow(np.ones(im.shape[:2]))
            # Sets the color data on a 2D plane in 3D space
            m_image.actor.input.point_data.scalars = colors
            # position in 3D space of the slice
            m_image.actor.position = [x_axis[j], y_axis[j], z_axis[j]]

    if slice_type == '3':
        slice_num = input('Input slice range(0-143) separated by a space: ')
        slice_num = slice_num.split()
        slice_num = range(int(slice_num[0]), int(slice_num[1]))

        while int(min([int(i) for i in slice_num])) < 0 or int(max([int(i) for i in slice_num])) > 143:
            print('ERROR: Invalid input')
            slice_num = input('Input slice range(0-143) separated by a space: ')
            slice_num = slice_num.split()
            slice_num = range(int(slice_num[0]), int(slice_num[1]))

        # 3D Plot with radius data
        t = my.plot3d(x_plot, y_plot, z_plot, s, tube_radius=0.1, color=(1, 0, 0))
        t.parent.parent.filter.vary_radius = 'vary_radius_by_scalar'

        for k in slice_num:
            # Assign posistion based on ratio of slice images to total points along the path
            j = int(int(k) * 11.805556666)
            # Reads the image data into and variable
            im = plt.imread('test' + str(k) + '.png', format='png') * 255
            colors = tvtk.UnsignedCharArray()
            # converts the PNG color data to variables Mayavi could understand
            colors.from_array(im.transpose((1, 0, 2)).reshape(-1, 3))
            m_image = my.imshow(np.ones(im.shape[:2]))
            # Sets the color data on a 2D plane in 3D space
            m_image.actor.input.point_data.scalars = colors
            # position in 3D space of the slice
            m_image.actor.position = [x_axis[j], y_axis[j], z_axis[j]]

    if slice_type == '4':
        # 3D Plot with radius data
        t = my.plot3d(x_plot, y_plot, z_plot, s, tube_radius=0.1, color=(1, 0, 0))
        t.parent.parent.filter.vary_radius = 'vary_radius_by_scalar'

    my.draw()
    my.show()

    # Check if user wants to run the graphs again
    print()
    again = input('Go again?(Y/N) ')
    while again != 'Y' and again != 'N':
        print('ERROR: Invalid input')
        again = input('Go again?(Y/N) ')
    if again == 'N':
        user_using = False

