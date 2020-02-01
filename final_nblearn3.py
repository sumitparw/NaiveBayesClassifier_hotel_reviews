import sys
import glob
import os
import collections
import math

bag_of_awords = collections.Counter()
bag_of_words = {}
bag_of_stop_words = {}

file_count_s = {}
probab_prior = {}  # In log
probab_conditional = {}
class_counts_total = {}


def Learn(root_dir):
    file_count_total = 0
    vocab_count_total = 0
    all_files = glob.glob(os.path.join(root_dir, '*/*/*/*.txt'))

    class_test = collections.defaultdict(list)
    class_train = collections.defaultdict(list)

    for f in all_files:
        # if ('fold1' not in f):
        class_1, class_2, fold, file_name = f.split("\\")[-4:]

        class_train[class_1 + class_2].append(f)
    for cl in class_train:
        bag_of_words[cl] = put_bag_of_words(class_train[cl])
        bag_of_awords.update(bag_of_words[cl])
        file_count_s[cl] = len(class_train[cl])
        file_count_total += file_count_s[cl]

    sorted_dict = sorted(bag_of_awords, key=bag_of_awords.get, reverse=True)

    for word in sorted_dict:
        if bag_of_awords[word] >= 1000:
            bag_of_stop_words[word] = bag_of_awords[word]

    vocab_count_total = len(bag_of_awords)
    for cl in bag_of_words:
        class_count = 0
        for word in bag_of_words[cl]:
            class_count += bag_of_words[cl][word]
        class_counts_total[cl] = class_count
        probab_prior[cl] = math.log(file_count_s[cl]) - math.log(file_count_total)
        probab_conditional[cl] = {}
        # print('class count = ' + str(class_count))
        for word in bag_of_words[cl]:
            probab_conditional[cl][word] = math.log(bag_of_words[cl][word] + 1) - math.log(class_count + vocab_count_total + 1)
    with open('nbmodel.txt', 'w') as fout:
        fout.write('Class counts\n')
        fout.write(str(class_counts_total))
        fout.write('\n')
        fout.write('Vocabulary\n')
        fout.write(str(vocab_count_total))
        fout.write('\n')
        fout.write('Stop words\n')
        fout.write(str(bag_of_stop_words))
        fout.write('\n')
        fout.write('Prior\n')
        fout.write(str(probab_prior))
        fout.write('\n')
        fout.write('Conditional\n')
        fout.write(str(probab_conditional))
        fout.write('\n')

# def special_char_check(ch):
#     if ch in '`~!@#-_=+[]{}\|;$%^&*():",.<>/?':
#         return True
#     return False

# def text_token(text):
#     review_text = text.lower()
#     #print(review_text)
#
#     review_text = ''.join([i for i in review_text if not i.isdigit()])
#
#     for dat in review_text:
#         if dat in '`~!@#-_=+[]{}\|;$%^&*():",.<>/?':
#             review_text = review_text.replace(dat, ' ')
#     text_split = review_text.split()
#     return text_split

def put_bag_of_words(class_list):
    new_bag_of_words = collections.Counter()
    for file_path in class_list:
        with open(file_path, 'r') as f:
            review_text = f.read()
            review_text = review_text.lower()
            review_text = ''.join([i for i in review_text if not i.isdigit()])

            for dat in review_text:
                if dat in '`~!@#-_=+[]{}\|;$%^&*():",.<>/?':
                    review_text = review_text.replace(dat, ' ')
            text_processed = review_text.split()
            new_bag_of_words.update(text_processed)
    return new_bag_of_words



if __name__ == "__main__":
    root_dir = sys.argv[1]
    nblearn = Learn(root_dir)
