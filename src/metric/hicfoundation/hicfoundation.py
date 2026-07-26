import os
import os.path
import re
import hashlib
import random
import string
from datetime import datetime
import os
import sys
import numpy as np
import pickle
from collections import defaultdict

current_datetime = datetime.now()
current_datetime_str = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
rand_letters = string.ascii_lowercase
rand_letters = ''.join(random.choice(rand_letters) for i in range(20))
output_dir = "/content/"


print("Please uploading your input files")
os.chdir("/content/")
root_dir = os.getcwd()
upload_dir = os.path.join(root_dir, rand_letters)
if not os.path.exists(upload_dir):
    os.mkdir(upload_dir)
os.chdir(upload_dir)
map_input = files.upload()
for fn in map_input.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(map_input[fn])))
    hic_input_path1 = os.path.abspath(fn)
    print("The input save to %s" % hic_input_path1)
os.chdir(root_dir)


os.chdir(upload_dir)
map_input = files.upload()
for fn in map_input.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(map_input[fn])))
    hic_input_path2 = os.path.abspath(fn)
    print("The input save to %s" % hic_input_path2)
os.chdir(root_dir)


input_pickle1 = hic_input_path1
input_pickle2 = hic_input_path2


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


input1 = load_pickle(input_pickle1)
input2 = load_pickle(input_pickle2)


def find_key(chr, loc, key_list):
    key1 = chr+":"+loc
    if key1 in key_list:
        return key1
    key1 = "chr"+chr+":"+loc
    if key1 in key_list:
        return key1
    key1 = chr+"_"+chr+":"+loc
    if key1 in key_list:
        return key1
    key1 = "chr"+chr+"_chr"+chr+":"+loc
    if key1 in key_list:
        return key1
    return None


def calculate_similarity(input1, input2):
    similarity_dict = defaultdict(list)
    for key in input1.keys():

        split_chromosome = key.split(":")[0]
        split_loc = key.split(":")[1]
        combine_key = split_chromosome + ":" + split_loc
        chr = split_chromosome.split("_")[0]
        chr = chr.replace("chr", "")
        if combine_key not in input2.keys():
            combine_key = find_key(chr, split_loc, input2.keys())
            if combine_key is None:
                continue

        embedding1 = input1[key]
        embedding2 = input2[combine_key]

        similarity = np.dot(embedding1, embedding2) / \
            (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        if np.isnan(similarity):
            continue
        similarity_dict[chr].append(similarity)

    similarity_list = []
    for chrom in similarity_dict:
        if "Y" in chrom or "M" in chrom or "Un" in chrom or "Alt" in chrom:
            continue
        mean_val = np.mean(similarity_dict[chrom])
        similarity_list.append(mean_val)
    similarity = np.mean(similarity_list)
    return similarity


similarity = calculate_similarity(input1, input2)
print("The reproducibility score between the two Hi-C is: ", similarity)
