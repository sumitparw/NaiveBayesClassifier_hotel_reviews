import sys
import glob
import os
import collections
import operator
import ast
import math
import re
import json

review_classes = ['positive_polaritytruthful_from_TripAdvisor', 'positive_polaritydeceptive_from_MTurk', 'negative_polaritytruthful_from_Web', 'negative_polaritydeceptive_from_MTurk']


def classify(root):

    all_files = glob.glob(os.path.join(root, '*/*/*/*.txt'))
    #print(len(all_files))

    file_read_dics = []
    with open('nbmodel.txt', 'r') as fmodel:
        for line in fmodel:
            file_read_dics.append(line)
    class_counts_total = ast.literal_eval(file_read_dics[1])
    vocab_counts_total = ast.literal_eval(file_read_dics[3])
    stopwords_counts_total = ast.literal_eval(file_read_dics[5])
    probab_prior = ast.literal_eval(file_read_dics[7])
    probab_conditional = ast.literal_eval(file_read_dics[9])

    for fi in all_files:
        probab_fourway = {}
        with open(fi, 'r') as fi_n:
            text = fi_n.read()
            text = text.lower()
            text = ''.join([i for i in text if not i.isdigit()])
            for c in text:
                if c in '`~!@#$%^&*()-_=+[]{}\|;:",.<>/?':
                    text = text.replace(c, ' ')
            processed_text = text.split()

            for c in review_classes:
                probab_text = 0.0
                for word in processed_text:
                    if word not in stopwords_counts_total:
                        if word not in probab_conditional[c]:
                            probab_conditional[c][word] = math.log(0 + 1) - math.log(class_counts_total[c] + vocab_counts_total + 1)

                        probab_text += probab_conditional[c][word]
                probab_text += probab_prior[c]
                probab_fourway[c] = probab_text

            c_probab = max(probab_fourway.items(), key=operator.itemgetter(1))[0]

            with open('nboutput.txt', 'a+') as fout:
                if c_probab == review_classes[0]:
                    #print("truthful positive")
                    fout.write('truthful positive ' + fi + '\n')
                elif c_probab == review_classes[1]:
                    #print("deceptive positive")
                    fout.write('deceptive positive ' + fi + '\n')
                elif c_probab == review_classes[2]:
                    #print("truthful negative")
                    fout.write('truthful negative ' + fi + '\n')
                elif c_probab == review_classes[3]:
                    #print("deceptive negative")
                    fout.write('deceptive negative ' + fi + '\n')



root_dir = sys.argv[1]
nbclassify = classify(root_dir)
