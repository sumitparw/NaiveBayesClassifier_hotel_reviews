import sys
import glob
import os
import collections
import operator
import ast
import math
import re
# import json


class NBClassify:

    def __init__(self, root):

        self.classes = ['positive_polaritytruthful_from_TripAdvisor', 'positive_polaritydeceptive_from_MTurk', 'negative_polaritytruthful_from_Web', 'negative_polaritydeceptive_from_MTurk']
        self.classify(root)

    def classify(self, root):
        # List all files, given the root of training data.
        all_files = glob.glob(os.path.join(root, '*/*/*/*.txt'))

        dics = []
        with open('nbmodel.txt', 'r') as fmodel:
            for line in fmodel:
                dics.append(line)
        class_counts = ast.literal_eval(dics[1])
        vocab = ast.literal_eval(dics[3])
        stopwords = ast.literal_eval(dics[5])
        prior = ast.literal_eval(dics[7])
        conditional = ast.literal_eval(dics[9])
        # print(conditional)

        # print(prior)
        # print(conditional)

        for f in all_files:
            # class1, class2, fold, fname = f.split("\\")[-4:]  # CHANGE!
            # if fold == 'fold1':
            prob_fourway = {}
            with open(f, 'r') as fin:
                text = fin.read()
                processed_text = self.tokenize(text)
                for cl in self.classes:
                    prob_text = 0.0
                    for word in processed_text:
                        if word not in stopwords:
                            if word not in conditional[cl]:
                                conditional[cl][word] = math.log(0 + 1) - math.log(class_counts[cl] + vocab + 1)
                                # print(word)
                                # print(class_counts[cl])
                                # print(conditional[cl][word])
                            prob_text += conditional[cl][word]
                    prob_text += prior[cl]
                    prob_fourway[cl] = prob_text
                # print(prob_fourway)
                c = max(prob_fourway.items(), key=operator.itemgetter(1))[0]
                # print(c)
                with open('nboutput.txt', 'a+') as fout:
                    if c == self.classes[0]:
                        fout.write('truthful positive ' + f + '\n')
                    elif c == self.classes[1]:
                        fout.write('deceptive positive ' + f + '\n')
                    elif c == self.classes[2]:
                        fout.write('truthful negative ' + f + '\n')
                    elif c == self.classes[3]:
                        fout.write('deceptive negative ' + f + '\n')

    def isspecialchar(self, ch):
        if ch in '`~!@#$%^&*()-_=+[]{}\|;:",.<>/?':
            return True
        return False

    def tokenize(self, text):
        text = text.lower()
        text = ''.join([i for i in text if not i.isdigit()])
        for c in text:
            if self.isspecialchar(c):
                text = text.replace(c, ' ')
        split_text = text.split()
        return split_text


root_dir = sys.argv[1]
# root_dir = "op_spam_training_data"
nbclassify = NBClassify(root_dir)
