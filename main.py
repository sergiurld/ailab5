import csv
import random
import matplotlib.pyplot as plt


file = 'v1_world-happiness-report-2017.csv'


def loadData(fileName, inputVariabName, outputVariabName):
    '''
    Load data from a CSV file based on the specified input and output variable names.
    Parameters:
    - fileName: Name of the CSV file to read.
    - inputVariabName: Name of the input variable column.
    - outputVariabName: Name of the output variable column.

    Returns:
    - inputs: List of input values.
    - outputs: List of output values.
    '''
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariable = dataNames.index(inputVariabName)
    inputs = [float(data[i][selectedVariable]) for i in range(len(data))]
    selectedOutput = dataNames.index(outputVariabName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]

    return inputs, outputs


# Load data from the specified file, extracting 'Family' as input and 'Happiness.Score' as output
inputs, outputs = loadData(file, 'Family', 'Happiness.Score')
# Print the first 5 input values
print('in:  ', inputs[:5])
# Print the first 5 output values
print('out: ', outputs[:5])

import matplotlib.pyplot as plt


def plotDataHistogram(x, variableName):
    '''
    Plot a histogram for the given data.

    Parameters:
    - x: List of data values.
    - variableName: Name of the variable to display in the title.
    '''
    # Create a histogram with 10 bins
    n, bins, patches = plt.hist(x, 10)

    # Set the title of the histogram
    plt.title('Histogram of ' + variableName)

    # Display the histogram
    plt.show()


# Plot histogram for 'Family' input data
plotDataHistogram(inputs, 'Family')

# Plot histogram for 'Happiness score' output data
plotDataHistogram(outputs, 'Happiness score')

# Plot scatter plot for 'Family' vs 'Happiness score'
plt.plot(inputs, outputs, 'ro')
plt.xlabel('Family')
plt.ylabel('Happiness score')
plt.title('Family vs. Happiness score')
plt.show()


def calculate_coefficients(inputs, outputs):
    '''
    Calculate the coefficients (intercept and slope) for a linear regression model.

    Parameters:
    - inputs: List of input values.
    - outputs: List of output values.

    Returns:
    - w0: Intercept of the linear regression model.
    - w1: Slope of the linear regression model.
    '''
    # Get the number of data points
    n = len(inputs)

    # Calculate the sum of inputs and outputs
    sum_x = sum(inputs)
    sum_y = sum(outputs)

    # Calculate the sum of the product of inputs and outputs
    sum_xy = sum(x * y for x, y in zip(inputs, outputs))

    # Calculate the sum of the squared inputs
    sum_xx = sum(x ** 2 for x in inputs)

    # Calculate the slope (w1) using the formula
    w1 = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)

    # Calculate the intercept (w0) using the formula
    w0 = (sum_y - w1 * sum_x) / n

    # Return the calculated coefficients
    return w0, w1


random.seed(5)

indexes = list(range(len(inputs)))

trainSample = random.sample(indexes, int(0.8 * len(inputs)))

validationSample = [i for i in indexes if i not in trainSample]

trainInputs = [inputs[i] for i in trainSample]
trainOutputs = [outputs[i] for i in trainSample]

validationInputs = [inputs[i] for i in validationSample]
validationOutputs = [outputs[i] for i in validationSample]

plt.scatter(trainInputs, trainOutputs, color='red', marker='o', label='training data')

plt.scatter(validationInputs, validationOutputs, color='green', marker='^', label='validation data')

plt.title('Train and Validation Data')
plt.xlabel('Family')
plt.ylabel('Happiness')

plt.legend()

plt.show()

w0, w1 = calculate_coefficients(trainInputs, trainOutputs)

print('w0 = ', w0, ' w1 = ', w1)

noOfPoints = 1000
xref = []
val = min(trainInputs)
step = (max(trainInputs) - min(trainInputs)) / noOfPoints
for i in range(1, noOfPoints):
    xref.append(val)
    val += step

all_equal = all(element == inputs[0] for element in inputs)
if all_equal:
    plt.scatter(trainInputs, trainOutputs, color='red', marker='o', label='training data')  # train data points
    plt.axvline(x=inputs[0], color='b', label='learnt model')  # k is for black, -- is for dashed line style
    plt.title('Train Data and the Learnt Model')
    plt.xlabel('Family')
    plt.ylabel('Happiness')
    plt.legend()
    # Optional: define limits for y-axis
    plt.ylim(0, 10)  # for example, set the y-axis to range from 0 to 10

    plt.show()
else:
    yref = [w0 + w1 * el for el in xref]
    plt.scatter(trainInputs, trainOutputs, color='red', marker='o', label='training data')  # train data points
    plt.plot(xref, yref, 'b-', label='learnt model')  # model reference line

    plt.title('Train Data and the Learnt Model')
    plt.xlabel('Family')
    plt.ylabel('Happiness')
    plt.legend()
    plt.show()


def loadData(fileName, inputVariabName1, inputVariabName2, outputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            elif '' not in row:
                data.append(row)
            line_count += 1
    selectedVariable1 = dataNames.index(inputVariabName1)
    inputs1 = [float(data[i][selectedVariable1]) for i in range(len(data))]
    selectedVariable2 = dataNames.index(inputVariabName2)
    inputs2 = [float(data[i][selectedVariable2]) for i in range(len(data))]
    selectedOutput = dataNames.index(outputVariabName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]

    return inputs1, inputs2, outputs


inputs1, inputs2, outputs = loadData(file, 'Freedom', 'Economy..GDP.per.Capita.',
                                     'Happiness.Score')
print('in1:  ', inputs1[:5])
print('in2:  ', inputs2[:5])
print('out: ', outputs[:5])

plotDataHistogram(inputs1, 'Freedom')
plotDataHistogram(inputs2, 'Economy..GDP.per.Capita.')
plotDataHistogram(outputs, 'Happiness score')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for 3D data
ax.scatter(inputs1, inputs2, outputs, c='r', marker='o')

ax.set_xlabel('Freedom')
ax.set_ylabel('Economy..GDP.per.Capita.')
ax.set_zlabel('Happiness.Score')
ax.view_init(elev=30, azim=45)
plt.title('Freedom vs. Economy..GDP.per.Capita. vs. Happiness.Score')
plt.show()


def transpose(matrix):
    return list(map(list, zip(*matrix)))


# Function to multiply two matrices
def multiply_matrices(A, B):
    return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]


# Function to find the inverse of a matrix
# For a 3x3 matrix this is doable by hand as shown below
def inverse_matrix(matrix):
    det = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
           matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
           matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))

    if det == 0:
        raise ValueError('The matrix is not invertible')

    inv_det = 1 / det
    return [
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inv_det,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inv_det,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inv_det,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inv_det,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inv_det,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inv_det,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inv_det,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inv_det,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inv_det,
        ],
    ]


# Assume inputs1, inputs2, and outputs are already defined lists of data

# Prepare matrix X by adding a column of ones for the intercept
X = [[1, x1, x2] for x1, x2 in zip(inputs1, inputs2)]
Y = [[y] for y in outputs]

# Transpose X to get X^T
XT = transpose(X)

# Multiply X^T with X to get X^T * X
XTX = multiply_matrices(XT, X)

# Invert X^T * X to get (X^T * X)^(-1)
XTX_inv = inverse_matrix(XTX)

# Multiply (X^T * X)^(-1) with X^T to get (X^T * X)^(-1) * X^T
XTX_inv_XT = multiply_matrices(XTX_inv, XT)

# Finally, multiply (X^T * X)^(-1) * X^T with Y to get the coefficients w
w = multiply_matrices(XTX_inv_XT, Y)

# w contains the coefficients [w0, w1, w2]
w0, w1, w2 = w[0][0], w[1][0], w[2][0]

print(f'w0 = {w0}, w1 = {w1}, w2 = {w2}')

predicted_outputs = [w0 + w1 * x + w2 * y for x, y in zip(inputs1, inputs2)]

# Create a figure for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the original data points
ax.scatter(inputs1, inputs2, outputs, color='r', marker='o', label='Actual data')

# Plot the predicted outputs (plane) as a scatter plot, you could also plot this as a surface
ax.scatter(inputs1, inputs2, predicted_outputs, color='b', marker='^', label='Predicted data')

# Label the axes
ax.set_xlabel('Freedom')
ax.set_ylabel('Economy..GDP.per.Capita.')
ax.set_zlabel('Happiness.Score')

# Set the viewing angle for better visibility
if(file!='v2_world-happiness-report-2017.csv'):
    ax.view_init(elev=30, azim=0)
else:
    ax.view_init(elev=30,azim=45)
# Add a title and a legend
plt.title('Freedom vs. Economy..GDP.per.Capita. vs. Happiness.Score')
ax.legend()

# Show the plot
plt.show()
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for actual data points
ax.scatter(inputs1, inputs2, outputs, color='r', marker='o', label='Actual data')

# Create a grid of values
x1_range = np.linspace(min(inputs1), max(inputs1), 10)
x2_range = np.linspace(min(inputs2), max(inputs2), 10)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# Calculate corresponding y values for the grid using the regression plane equation
y_grid = w0 + w1 * x1_grid + w2 * x2_grid

# Plot the surface
ax.plot_surface(x1_grid, x2_grid, y_grid, color='b', alpha=0.5, label='Prediction Plane')

ax.set_xlabel('Freedom')
ax.set_ylabel('Economy..GDP.per.Capita.')
ax.set_zlabel('Happiness.Score')
ax.view_init(elev=30., azim=30)  # You can change the view angle as needed
plt.title('Freedom vs. Economy..GDP.per.Capita. vs. Happiness.Score')
ax.legend()
plt.show()