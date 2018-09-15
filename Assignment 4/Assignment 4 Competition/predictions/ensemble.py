import csv
import sys
from collections import Counter

num_files = int(sys.argv[1])
allpreds = [[] for i in range(100000)]

def csv_dict_reader(file_obj):
	global allpreds
	reader = csv.DictReader(file_obj, delimiter=',')
	for line in reader:
		allpreds[int(line["ID"])].append(line["CATEGORY"])

if __name__ == '__main__':
	for i in range(num_files):
		with open('./' + str(i + 1) + '.csv') as f_obj:
			csv_dict_reader(f_obj)
	finalpreds = []
	for i in range(100000):
		somelist = allpreds[i]
		counts = Counter(somelist).most_common(1)
		finalpreds.append(counts[0])
	f = open('ensemble.csv', 'w')
	f.write("ID,CATEGORY\n")
	for i in range(len(finalpreds)):
	    f.write(str(i) + "," + finalpreds[i][0] + "\n")
	f.close()
