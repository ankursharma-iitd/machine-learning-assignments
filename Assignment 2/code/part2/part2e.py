import numpy as np
import csv
import matplotlib.pyplot as plt
i=0
with open('mnist/test.csv', 'r') as csv_file:
    content = csv_file.readlines()
    something = [33, 613, 4990]
    actual = [4, 2, 3]
    predicted = [6, 8, 2]
    for i in range(3):
        pixels = list(map(int, content[something[i]].split(',')))
        num_features = len(pixels) - 1
        # This array will be of 1D with length 784
        # The pixel intensity values are integers from 0 to 255
        pixels = np.array(pixels[:num_features])

        # Reshape the array into 28 x 28 array (2-dimensional array)
        pixels = pixels.reshape((28, 28))
        # Plot
        plt.title('Actual Label : ' + str(actual[i]) + ', Predicted Label : ' + str(predicted[i]))
        plt.imshow(pixels, cmap='gray')
        plt.savefig(str(i) + '.png')
        plt.close()
        i +=1
