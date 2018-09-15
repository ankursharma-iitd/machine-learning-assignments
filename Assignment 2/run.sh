#!/bin/sh
#!/usr/bin/env python

if [ $1 -eq 1 ] 
then
    if [ $2 -eq 1 ]
    then
        #1 NB model corresponding to part-a i.e. without stemming and stopword removal
        echo $1 : NB, $2 : Model
        python3 part1.py $2 $3 $4
    elif [ $2 -eq 2 ]
    then
        #2 NB model corresponding to part-d i.e. with stemming and stopword removal.
        echo $1 : NB, $2 : Model
        python3 part1.py $2 $3 $4
        rm new_test_text
    elif [ $2 -eq 3 ]
    then
        #3 NB model corresponding to part-e, i.e. your best model.
        echo $1 : NB, $2 : Model
        python3 part1.py $2 $3 $4
        rm new_test_text
    fi
elif [ $1 -eq 2 ]
then
    if [ $2 -eq 1 ]
    then
        #1 Pegasos model corresponding to part b.
        echo $1 : SVM, $2 : Model
        python part2.py $2 $3 $4
    elif [ $2 -eq 2 ]
    then
        #2 Libsvm model (linear kernel) corresponding to part c.
        echo $1 : SVM, $2 : Model
        python part2.py $2 $3 new_test.csv
        ./svm-scale -l 0 -u 1 new_test.csv > scaled_test
        ./svm-predict scaled_test models_2/model2.p $4
        rm new_test.csv
        rm scaled_test
    elif [ $2 -eq 3 ]
    then
        #3 Best Libsvm model (rbf kernel) corresponding to part d.
        echo $1 : SVM, $2 : Model
        python part2.py $2 $3 new_test.csv
        ./svm-scale -l 0 -u 1 new_test.csv > scaled_test
        ./svm-predict scaled_test models_2/model3.p $4
        rm new_test.csv
        rm scaled_test
    fi
else 
    echo Unknown option.
fi
