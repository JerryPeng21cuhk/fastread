import numpy as np
from pdb import set_trace
from demos import cmd
import pickle
import matplotlib.pyplot as plt
from sk import rdivDemo
import random
from mar import MAR
import pdb
from collections import Counter
# from xlrd import open_workbook
from xlrd import *
import csv


def active_learning(filename, query='', stop='true', stopat=0.95, error='none', interval=100000, seed=0):
    stopat = float(stopat)
    thres = 0
    starting = 1
    counter = 0
    pos_last = 0
    np.random.seed(seed)
    read = MAR()
    read = read.create(filename)
    read.interval = interval

    read.BM25(query.strip().split('_'))

    num2 = read.get_allpos()
    target = int(num2 * stopat)
    print("number of target, true/close here:", target)

    if stop == 'est':
        read.enable_est = True
    else:
        read.enable_est = False

    while True:
        pos, neg, total = read.get_numbers()
        try:
            print("%d, %d, %d" % (pos, pos+neg, read.est_num))
        except:
            print("%d, %d" % (pos, pos+neg))

        if pos + neg >= total:
            if stop == 'knee' and error == 'random':
                coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                seq = coded[np.argsort(read.body['time'][coded])]
                part1 = set(seq[:read.kneepoint * read.step]) & set(
                    np.where(np.array(read.body['code']) == "no")[0])
                part2 = set(seq[read.kneepoint * read.step:]) & set(
                    np.where(np.array(read.body['code']) == "yes")[0])
                for id in part1 | part2:
                    read.code_error(id, error=error)
            break

        if pos < starting or pos+neg < thres:
            for id in read.BM25_get():
                read.code_error(id, error=error)
        else:
            a,b,c,d =read.train(weighting=True, pne=True)
            if stop == 'est':
                if stopat * read.est_num <= pos:
                    break
            elif stop == 'soft':
                if pos>= 10 and pos_last == pos:
                    counter = counter+1
                else:
                    counter = 0
                pos_last = pos
                if counter >= 5:
                    break
            elif stop == 'knee':
                if pos >= 10:
                    if read.knee():
                        if error == 'random':
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
    set_trace()                
    return read


def csv_from_excel():

    wb = open_workbook('MySpreadSheet.xlsx')
    sh = wb.sheet_by_name('Sheet1')
    your_csv_file = open('output.csv', 'w', encoding='utf8')
    wr = csv.writer(your_csv_file, quoting = csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    your_csv_file.close()


if __name__ == "__main__":

    wb = open_workbook('mydataset.xlsx')
    sh = wb.sheet_by_name('Sheet1')
    your_csv_file = open('output.csv', 'w', encoding='utf8')
    wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)
    housing_map = {'open': 'no', 'close': 'yes'}

    for rownum in range(sh.nrows):

        # print(type(sh))
        #if sh.nrows == 'open':
        #    rownum.label = 'no'
        #else:
         #   rownum.label ='yes'

        wr.writerow(sh.row_values(rownum))
    your_csv_file.close()

#    with open_workbook('mydataset.xlsx') as wb:
#        sh = wb.sheet_by_index(0)  # or wb.sheet_by_name('name_of_the_sheet_here')
#    with open('a_file.csv', 'wb') as f:
#        c = csv.writer(f)
#        for r in range(sh.nrows):
#            c.writerow(sh.row_values(r))

    # active_learning('output.csv')
    active_learning('Hall.csv')
    # eval(cmd())
