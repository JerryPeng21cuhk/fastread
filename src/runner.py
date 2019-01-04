from __future__ import division, print_function

import numpy as np
from pdb import set_trace
from demos import cmd, demo
import pickle
import matplotlib.pyplot as plt
from mar import MAR
from sk import rdivDemo

import random

from collections import Counter

def active_learning(filename, query='', stop='true', stopat=1.00, error='none', interval = 100000, seed=0):
    stopat = float(stopat)
    thres = 0
    starting = 1
    counter = 0
    pos_last = 0
    np.random.seed(seed)

    read = MAR()
    read = read.create(filename)
    # random sampling or by querying similar documents
    # self.bm is provided with a list or a view of a dict's value which is not sorted
    read.BM25(query.strip().split('_'))

    # get the rest #pos documents
    num2 = read.get_allpos()
    target = int(num2 * stopat) # stopat is 1. Is it the minum num of pos to activate svm training ?
    if stop == 'est': # stop = 'true'
        read.enable_est = True
    else:
        read.enable_est = False # will excute this line

    while True:
        pos, neg, total = read.get_numbers()
        try:
            print("%d, %d, %d" %(pos,pos+neg, read.est_num)) # what is est_num ?
        except:
            print("%d, %d" %(pos,pos+neg)) # execute this line

        if pos + neg >= total: # do not go inside
            if stop=='knee' and error=='random':
                coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                seq = coded[np.argsort(read.body['time'][coded])]
                part1 = set(seq[:read.kneepoint * read.step]) & set(
                    np.where(np.array(read.body['code']) == "no")[0])
                part2 = set(seq[read.kneepoint * read.step:]) & set(
                    np.where(np.array(read.body['code']) == "yes")[0])
                for id in part1 | part2:
                    read.code_error(id, error=error)
            break

        if pos < starting or pos+neg<thres: # the second condition doesn't work
            for id in read.BM25_get(): # select a set of candidates from self.pool
                read.code_error(id, error=error) # simulate human labeling error, default is no error
        else:
            a,b,c,d =read.train(weighting=True,pne=True)
            if stop == 'est':
                if stopat * read.est_num <= pos:
                    break
            elif stop == 'soft':
                if pos>=10 and pos_last==pos:
                    counter = counter+1
                else:
                    counter=0
                pos_last=pos
                if counter >=5:
                    break
            elif stop == 'knee':
                if pos>=10:
                    if read.knee():
                        if error=='random':
                            coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                            seq = coded[np.argsort(np.array(read.body['time'])[coded])]
                            part1 = set(seq[:read.kneepoint * read.step]) & set(
                                np.where(np.array(read.body['code']) == "no")[0])
                            part2 = set(seq[read.kneepoint * read.step:]) & set(
                                np.where(np.array(read.body['code']) == "yes")[0])
                            for id in part1|part2:
                                read.code_error(id, error=error)
                        break
            else:
                if pos >= target:
                    break
            if pos < 10:
                for id in a:
                    read.code_error(id, error=error)
            else:
                for id in c:
                    read.code_error(id, error=error)
    return read


if __name__ == "__main__":
    # eval(cmd())
    active_learning('Hall.csv', query='software')
