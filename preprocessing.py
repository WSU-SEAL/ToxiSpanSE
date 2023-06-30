# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2023
# Authors: Jaydeb Sarker <jaydebsarker@wayne.edu> and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

from sklearn.model_selection import train_test_split, KFold
import numpy as np
import numpy
import en_core_web_sm

nlp = en_core_web_sm.load()
import pandas as pd


##partition of the data using dictionary
def partition_input_dict(inputDict, dataIndex):
    returnDict = dict()
    returnDict["input_ids"] = inputDict["input_ids"][dataIndex]
    returnDict["token_type_ids"] = inputDict["token_type_ids"][dataIndex]
    returnDict["attention_mask"] = inputDict["attention_mask"][dataIndex]
    return returnDict

##test train ratio
def generate_test_train_index( length, ratio):
    rand_array=np.arange(0,length,1)
    train, test =train_test_split(rand_array,test_size=ratio, shuffle=True )
    train=numpy.sort(train)
    test=numpy.sort(test)
    return train, test

# spliting each text (Code Rreview comment) to sentences
def split_paragraph_to_sentences(data):
    base_offset = []
    sentences = []
    sentence_spans = []
    sent_start = []

    print("Shape: ", data.shape[0])

    for i in range(len(data)):
        # print( data['text'][i])
        text = str(data['text'][i])
        spans = data['spans'][i]
        left_ptr = 0
        right_ptr = 0
        started = False
        doc = nlp(text)
        for sent in doc.sents:
            start = doc[sent.start].idx
            end = doc[sent.end - 1].idx + len(doc[sent.end - 1])
            if end - start <= 1:
                continue
            sentences.append(text[start:end])
            base_offset.append(start)
            if started == False:
                started = True
                sent_start.append(True)
            else:
                sent_start.append(False)
            while left_ptr < len(spans) and spans[left_ptr] < start:
                left_ptr += 1
            right_ptr = left_ptr
            cur_spans = []
            while right_ptr < len(spans) and spans[right_ptr] < end:
                cur_spans.append(spans[right_ptr])
                right_ptr += 1
            sentence_spans.append(cur_spans)
            left_ptr = right_ptr
        if started == False:
            sentences.append(text)
            base_offset.append(0)
            sentence_spans.append(spans)
            sent_start.append(True)
    data_processed = pd.DataFrame(
        data={"text": sentences, "spans": sentence_spans, "base_offset": base_offset, "sent_start": sent_start})
    return data_processed


##preprocess each sentence
def sentence_preprocessing(data, max_length, tokenizer):
        data_processed = split_paragraph_to_sentences(data)
        print(len(data_processed['text']))
        input_texts =data_processed['text']
        input_texts=input_texts.to_numpy()
        tokens = tokenizer(list(data_processed.text), max_length=max_length, padding="max_length",
                           truncation=True, return_offsets_mapping=True, return_special_tokens_mask=True,
                           return_token_type_ids=True)

        data_input = {"input_ids": np.array(tokens.input_ids),
                      "token_type_ids": np.array(tokens.token_type_ids),
                      "attention_mask": np.array(tokens.attention_mask)}

        data_output = generate_output(data_processed, tokens, max_length,tokenizer)

        #count=0
        #for i in range(len(data_output)):
            #a=sum(data_output[i])
            #if(a>=1):
                #count=count+1
        #print("total 1 is:", count)

        inital_weights = 1 - np.array(tokens.special_tokens_mask)
        return data_input, input_texts, data_output, inital_weights,tokens

# prepare the output toxic/non toxic labels
def generate_output(data, tokens, max_length,tokenizer):
        output = np.zeros((data.shape[0], max_length))
        for i in range(data.shape[0]):
            isToxic = np.zeros(len(data.text[i]))
            for idx in data.spans[i]:
                isToxic[idx - data.base_offset[i]] = 1
            for j in range(max_length):
                if tokens.special_tokens_mask[i][j]:
                    continue

                sp_tok=tokenizer.convert_ids_to_tokens(tokens.input_ids[i][j])
                ##this portion is added to making some unwanted tokens as 0 (non-toxic)
                # these tokens are generated for pretrained tokenizers
                if(sp_tok=="_" or sp_tok=="s" or sp_tok=="D"):
                    continue
                if (sp_tok == "Ġ" or  sp_tok=="Ġ," or sp_tok=="Ġ." or sp_tok=="Ġ/" or sp_tok=="Ġs" or sp_tok=="Ġ[" or sp_tok=="Ġy" or sp_tok=="Ġd" or sp_tok=="Ġ@" or sp_tok=="Ġa"):
                    continue
                if (sp_tok=="ĠI" or sp_tok=="Ġ(" or sp_tok=="Ġi" or sp_tok=="Ġ*" or sp_tok=="Ġ_" or sp_tok=="Ġ:" or sp_tok=="Ġ=" or sp_tok=="Ġ-"):
                    continue
                if(sp_tok=="ĠU" or sp_tok=="Ġ#" or sp_tok=="ĠD" or sp_tok=="ĠA" or sp_tok=="ĠS" or sp_tok=="Ġ\"" or sp_tok=="Ġ," or sp_tok=="Ġ>"):
                    continue
                start = tokens.offset_mapping[i][j][0]
                end = tokens.offset_mapping[i][j][1]
                cnt = 0
                for pos in range(start, end):
                    if isToxic[pos]:
                        cnt += 1
                if cnt >= (end - start + 1) // 2:
                    output[i][j] = 1
        return output


