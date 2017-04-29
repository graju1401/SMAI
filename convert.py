import csv

with open("test.csv", 'r') as fp:
    mat = []
    for r in csv.reader(fp, delimiter=","):
        mat.append(r)
    with open("test.out", 'w') as tp:
        for i in range(1, len(mat) + 1):
            for j in range(1, i):
                if mat[i - 1][j - 1] == "1":
                    tp.write(str(i) + " " + str(j) + "\n")