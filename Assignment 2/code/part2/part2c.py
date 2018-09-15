import sys
import numpy as np

train_x_data = []
train_y_date = []
num_examples = 0
input_train_file = sys.argv[1]
output_train_file = sys.argv[2]

if __name__ == '__main__':
    input_train_csv = input_train_file
    print('\nREADING THE CSV FILE...')
    with open(input_train_csv) as f:
        content = f.readlines()
        for i in range(len(content)):
            x = content[i]
            pixels = list(map(int, x.split(',')))
            num_features = len(pixels) - 1
            y = pixels[num_features]
            train_y_date.append(y)
            train_x_data.append(pixels[:num_features])
    print('CSV FILE HAS BEEN READ!\n')
    num_examples = len(train_x_data)

    output_train_csv = output_train_file
    print('\nPRINTING THE NEW CSV FILE...')
    with open(output_train_csv, 'a') as f:
        for i in range(num_examples):
            curr_ex = train_x_data[i]
            curr_label = train_y_date[i]
            string = str(curr_label) + ' '
            count = 1
            for j in range(len(curr_ex)):
                string += str(count) + ':' + str(curr_ex[j]) + ' '
                count += 1
            string += '\n'
            f.write(string)
        f.close()
    print('CSV FILE HAS BEEN CREATED!\n')
