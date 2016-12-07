# encoding=utf-8
import csv
import numpy as np

embeddings = np.loadtxt('embeddings.csv', delimiter=',')

print np.sum(embeddings, axis=1)
print '============================='
print np.sum(np.square(embeddings), axis=1)
