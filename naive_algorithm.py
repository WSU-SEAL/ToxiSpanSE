# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2023
# Authors: Jaydeb Sarker <jaydebsarker@wayne.edu> and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

#This is naive toxic token matching model

import pandas as pd
from nltk.tokenize import word_tokenize
from ast import literal_eval
import numpy as np
from preprocessing import split_paragraph_to_sentences
from nltk.tokenize import WhitespaceTokenizer

import tokenizations
import textspan
import nltk
from pandas import *
from metric import compute_performance
from nltk.stem import PorterStemmer


##read 167 top toxic tokens
new_token_data = read_csv("models/toxic_token_list.csv")
new_token_list=new_token_data['toxic_words'].tolist()


#data=pd.read_csv("models/CR_full_span_dataset.csv")
data=pd.read_excel('models/CR_full_span_dataset.xlsx')
data["spans"] = data["spans"].apply(literal_eval)

data_process=split_paragraph_to_sentences(data)


print(len(data_process))

tokens_list=[]
offset_mapping=[]
for i in range(0,len(data_process)):
    toks=word_tokenize(data_process.text[i])
    tokens_list.append(word_tokenize(data_process.text[i]))
    offset_mapping.append(textspan.get_original_spans(toks, data_process.text[i]))

maxlen=0
for i in range(0,len(offset_mapping)):
    #print(len(offset_mapping[i]))
    a=len(offset_mapping[i])
    if(maxlen<int(a)):
        maxlen=a

print(maxlen)



def ground_truth(data, tokens, max_length,offset_mapping):
    output = np.zeros((data.shape[0], max_length))

    for i in range(data.shape[0]):
        isToxic = np.zeros(len(data.text[i]))
        for idx in data.spans[i]:
            isToxic[idx - data.base_offset[i]] = 1
        for j in range(0,len(offset_mapping[i])):
            if(offset_mapping[i][j]==[]):
                continue
            start = offset_mapping[i][j][0][0]
            end = offset_mapping[i][j][0][1]
            cnt = 0
            for pos in range(start, end):
                if isToxic[pos]:
                    cnt += 1
            if cnt >= (end - start + 1) // 2:
                output[i][j] = 1
    return output

##generate ground truth
output_vec=ground_truth(data_process, tokens_list, maxlen, offset_mapping)

ps = PorterStemmer()
def naive_classifier_output(data, tokens,list_of_selected_tok,max_len):
    output = np.zeros((data.shape[0], max_len))
    for i in range(data.shape[0]):
        for j in range(0, len(tokens[i])):
            for k in range(0,len(list_of_selected_tok)):
                stem=tokens[i][j].lower()
                if(stem==list_of_selected_tok[k]):
                    output[i][j]=1

    return output

##classifier output
naive_op=naive_classifier_output(data_process, tokens_list, new_token_list,maxlen)

##results
## the following 3 are unused arguments for naive algorithm
test_texts=""
tokenizer=""
retro_output=[]


avg_precision_0, avg_recall_0, avg_f1_0, avg_precision_1, avg_recall_1, avg_f1_1,retro_output = compute_performance(
        output_vec, naive_op,test_texts, tokenizer, maxlen, False)

print("Precision non-toxic: " + str(avg_precision_0))
print("Recall non-toxic" + str(avg_recall_0))
print("F1 non-toxic: " + str(avg_f1_0))
print("Precision toxic: " + str(avg_precision_1))
print("Recall toxic" + str(avg_recall_1))
print("F1 toxic: " + str(avg_f1_1))
