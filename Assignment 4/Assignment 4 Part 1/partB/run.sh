#!/bin/sh
#!/usr/bin/env python

mkdir svm
python partB.py 0 0 0 0
./svm-scale -l 0 -u 1 svm/new_train > svm/scaled_train
./svm-scale -l 0 -u 1 svm/new_test > svm/scaled_test
./svm-train -t 0 -h 0 -c 5 svm/scaled_train svm/linear_model
./svm-predict svm/scaled_train svm/linear_model svm/linear_kernel_output_train
./svm-predict svm/scaled_test svm/linear_model svm/linear_kernel_output_test
python partB.py 1 svm/linear_kernel_output_train svm/linear_kernel_output_test svm/submission.csv
