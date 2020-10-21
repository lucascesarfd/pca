import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

@dataclass
class context:
    data: []
    data_adjust: []
    avg: []
    num_of_elements: int = 0
    num_of_entry: int = 0

max_entrys = 3


def get_data(ctx, file_name):
    '''
    This function read all data from the file "file_name" and fill the variables of the context "ctx" with the correct interpretation. 
    It returns the dataframe of the pd object.
    
    Example:
        ctx = context(data, data_adjust,avg)
        data_frame = get_data(ctx, "text_file.txt"):
    '''
    data_frame = pd.read_csv(file_name, sep=";", header=None)

    ctx.num_of_elements  = data_frame.shape[0]  # gives number of rows
    ctx.num_of_entry = data_frame.shape[1]  # gives number of columns

    for var in range(max_entrys):
        if var < ctx.num_of_entry:
            ctx.data[var] = data_frame[var].to_list()
            ctx.data_adjust[var] = data_frame[var].to_list()
        else:
            del ctx.data[-1]
            del ctx.data_adjust[-1]

    return data_frame

def get_average(ctx, data_frame):
    '''
    This function returns the simple mean (average value) of each component of the data_frame using the context variable.

    Example:
        ctx = context(data, data_adjust,avg)
        data_frame = pd.read_csv("text_file.txt", sep=";", header=None)
        get_average(ctx, data_frame)
    '''
    sum_of_axis = data_frame.sum(axis=0) # gives the sum of determinated axis in a list

    for column in range(ctx.num_of_entry):
        ctx.avg[column] = sum_of_axis[column]/ctx.num_of_elements

def get_adjusted_by_mean(ctx):
    '''
    This function fill the arrays of the context with the values of each component adjusted by mean.

    Example:
        ctx = context(data, data_adjust,avg)
        get_adjusted_by_mean(ctx)
    '''
    for data_group in range(ctx.num_of_entry):
        for element in range(ctx.num_of_elements):
            ctx.data_adjust[data_group][element] = ctx.data[data_group][element] - ctx.avg[data_group]
    

def evaluate_covariance(ctx, first_dimension, first_dimension_avg, second_dimension, second_dimension_avg):
    '''
    This function use the two given numerical arrays to evaluate the covariance according the formula:
        1. 𝑐𝑜𝑣(𝑥,𝑦)=(∑(𝑥𝑖−𝑥_i)(𝑦𝑖−𝑦_𝑖 ))/(𝑁−1)
    Where:
        xi = first_dimension
        xi_ = first_dimension_avg
        yi = second_dimension
        yi_ = second_dimension_avg
    The result will be returned.

    Example:
        first_dimension = [1,2,3]
        first_dimension_avg = [2]
        second_dimension = [4,5,6]
        second_dimension_avg = [5]
        ctx = context(data, data_adjust, avg)
        covariance = evaluate_covariance(ctx, first_dimension, first_dimension_avg, second_dimension, second_dimension_avg)
    '''
    covariance = 0
    for element in range(ctx.num_of_elements):
        covariance += (first_dimension[element]-first_dimension_avg) * (second_dimension[element]-second_dimension_avg)

    covariance = covariance/(len(first_dimension)-1)

    return covariance

def generate_covariance_matrix(ctx, matrix):
    '''
    This function genarates a covariance matrix "matirx" using the values from the context "ctx".
    The matrix must be square!

    Example:
        ctx = context(data, data_adjust, avg)
        matrix = np.zeros(3,3)
        generate_covariance_matrix(ctx, matrix)
    '''
    order = ctx.num_of_entry

    for row in range(order):
        for column in range(order):
            matrix[row][column] = evaluate_covariance(ctx, ctx.data[row], ctx.avg[row], ctx.data[column], ctx.avg[column])

def draw_vector(v0, v1, ax=None):
    '''
    This function receive two points and draw a vector of them.

    Example:
        v0 = [1,2]
        v1 = [3,4]
        draw_vector(v0, v1)
    '''
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

def draw_adjusted_point(v0, v1, ax=None):
    '''
    This function receive two points and draw a vector of them.

    Example:
        v0 = [1,2]
        v1 = [3,4]
        draw_vector(v0, v1)
    '''
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='|-|',
                    linewidth=1)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

def dimension_reduction(order, eigen_values, eigen_vectors, eigen_values_red, eigen_vectors_red):
    '''
    This function orders the eigenvalues and eigenvectors according to the order predefined in "order" and returns the result in "eigen_values_red" and "eigen_vectors_red"
    
    Example:
        order = 2
        eigen_values = [1,2,3]
        eigen_vectors = [[1,2,3], [4,5,6], [7,8,9]]
        eigen_values_red = np.zeros(2,2)
        eigen_vectors_red = np.zeros(2,3)
        dimension_reduction(order, eigen_values, eigen_vectors, eigen_values_red, eigen_vectors_red)
    '''
    index = np.zeros(order)
    eigen_values_aux = eigen_values.copy()

    for step in range(len(eigen_values)):
        if step < order:
            eigen_values_red[step] = np.amax(eigen_values_aux)
            index[step] = np.where(eigen_values == eigen_values_red[step])[0]
            eigen_values_aux = np.delete(eigen_values_aux, np.where(eigen_values_aux == eigen_values_red[step]))

    for column in range(len(eigen_vectors[0])):
        if column < order:
            eigen_vectors_red[:, column] = eigen_vectors[:,int(index[column])]

def transpose_matrix(input_matrix, result):
    '''
    This function resturns the transpose of the original matrix on the "result"

    Example:
        input_matrix = [[1,2,3], [4,5,6]]
        result = np.zeros(3,2)
        transpose_matrix(first, result)
    '''
    for row in range(len(input_matrix)):
        for column in range(len(input_matrix[0])):
            result[column][row] = input_matrix[row][column]

def multiply_matrices(first, second, result):
    '''
    This function does the multiplication of two matrices and return into "result" matrix

    Example:
        first = [[1,2,3], [4,5,6], [7,8,9]]
        second = [[1,2,3], [4,5,6], [7,8,9]]
        result = np.zeros(3,3)
        multiply_matrices(first, second, result)
    '''
    sum = 0.0
    #iterate for each column of the second matrix
    for i in range(len(second[0])):
        #iterate for each row of the first matrix
        for j in range(len(first)):
            sum = 0.0
            #iterate for each column of the first matrix (equal to line on second matrix)
            for h in range(len(first[0])):
                sum += first[j][h]*second[h][i]
            result[j][i] = sum


def main():
    data_1 = []
    data_2 = []
    data_3 = []
    data_adj_1 = []
    data_adj_2 = []
    data_adj_3 = []
    avg = [0,0,0]
    order = 1

    file_name=".\\data\\Books_attend_grade.txt"

    ctx = context([data_1, data_2, data_3], [data_adj_1, data_adj_2, data_adj_3], avg)
    
    data_frame = get_data(ctx, file_name)

    get_average(ctx, data_frame)

    get_adjusted_by_mean(ctx)

    cov_matrix = np.zeros((ctx.num_of_entry, ctx.num_of_entry))

    generate_covariance_matrix(ctx, cov_matrix)

    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

    # declare the matrix of EigenValues and EigenVectors of the reduced dimension
    eigen_values_red = np.zeros(order)
    eigen_vectors_red = np.zeros((len(eigen_vectors), order))

    dimension_reduction(order, eigen_values, eigen_vectors, eigen_values_red, eigen_vectors_red)

    new_dataset = np.zeros((len(eigen_values_red), ctx.num_of_elements))
    eigen_vector_transpose = np.zeros((len(eigen_vectors_red[0]), len(eigen_vectors_red)))

    transpose_matrix(eigen_vectors_red, eigen_vector_transpose)

    multiply_matrices(eigen_vector_transpose, ctx.data_adjust, new_dataset)

    # Plot the original dataset
    plt.scatter(ctx.data[0], ctx.data[1]) 

    # Plot the Vectors from PCA
    vector = np.array([eigen_vectors[0][0],eigen_vectors[1][0]])
    v = vector * 100
    draw_vector([ctx.avg[0], ctx.avg[1]], [ctx.avg[0], ctx.avg[1]] + v)
    vector = np.array([eigen_vectors[0][1],eigen_vectors[1][1]])
    v = vector * 100
    draw_vector([ctx.avg[0], ctx.avg[1]], [ctx.avg[0], ctx.avg[1]] + v)

        # Plot the Vectors from PCA
    for element in new_dataset[0]:
        vector = np.array([eigen_vectors_red[0][0],eigen_vectors_red[1][0]])
        v = vector * element
        draw_adjusted_point([ctx.avg[0], ctx.avg[1]], [ctx.avg[0], ctx.avg[1]] + v)
    plt.axis('equal')
    plt.show()

    print()
    print("Dataset from file %s is:" % file_name)
    print(data_frame)
    print()
    print("Elements: " + str(ctx.num_of_elements) + " - Entry's: " + str(ctx.num_of_entry))
    print()
    print("Value 0:")
    print(ctx.data[0])
    print("Value 1:")
    print(ctx.data[1])
    print("Value 2:")
    print(ctx.data[2])
    print()
    print("Average 0")
    print(ctx.avg[0])
    print("Average 1")
    print(ctx.avg[1])
    print("Average 2")
    print(ctx.avg[2])
    print()
    print("Value Adjusted 0:")
    print(ctx.data_adjust[0])
    print("Value Adjusted 1:")
    print(ctx.data_adjust[1])
    print("Value Adjusted 2:")
    print(ctx.data_adjust[2])
    print()
    print("Covariance Matrix:")
    print(cov_matrix)
    print()
    print("EigenValues:")
    print(eigen_values)
    print()
    print("EigenVectors")
    print(eigen_vectors)
    print()
    print("EigenValues Reduced:")
    print(eigen_values_red)
    print()
    print("EigenVectors Reduced")
    print(eigen_vectors_red)
    print()
    print("New Dataset")
    print(new_dataset)

############################
# Main program starts here #
############################
main()


