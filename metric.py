# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2023
# Authors: Jaydeb Sarker <jaydebsarker@wayne.edu> and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

import numpy as np
from post_processings import convert_to_original_text_with_toxicity

def offset_precision_recall_fscore(gold,predictions):

    num_correct = len(set(predictions).intersection(set(gold)))
    num_predicted = len(set(predictions))
    correct_span_length = len(set(gold))


    if num_predicted == 0:
        if correct_span_length==0:
            precision =1 #non-toxic instance correctly predicted
        else:
            precision =0
    else:
        precision =num_correct/num_predicted

    if correct_span_length==0:
        if num_predicted ==0:
            recall = 1 # non-toxic correctly predicted
            f1=1  # non-toxic correctly predicted
        else:
            recall =0
            f1=0
    else:
        recall =num_correct /correct_span_length
        f1 = (2*num_correct) / (correct_span_length+ num_predicted )
    return precision, recall, f1

def compute_performance(ground_truth, predicted_output, test_texts,tokenizer, maxlen, error_analysis=False):
    instance_f1_0=[]
    instance_prec_0=[]
    instance_recall_0=[]
    instance_f1_1=[]
    instance_prec_1=[]
    instance_recall_1=[]
    retro_output=[]

    for i in range(0, len(ground_truth)):
        pred_offset=[]
        test_base_offset=[]

        for j in range(0, maxlen):
            if(ground_truth[i][j]==1):
                test_base_offset.append(j)        ##ground truth offset
            if(predicted_output[i][j]==1):
                pred_offset.append(j)             ##token offset for prediction

        if len(test_base_offset)==0: #non-toxic instance
            (precision_0, recall_0, f1_0) = offset_precision_recall_fscore(test_base_offset, pred_offset)
            instance_prec_0.append(precision_0)
            instance_recall_0.append(recall_0)
            instance_f1_0.append(f1_0)
        else:  #toxic instances
            (precision_1, recall_1, f1_1) = offset_precision_recall_fscore(test_base_offset, pred_offset)
            instance_prec_1.append(precision_1)
            instance_recall_1.append(recall_1)
            instance_f1_1.append(f1_1)
        if error_analysis:
            gold ='[' +','.join( str(v) for v in test_base_offset) +']'
            predicted ='[' +','.join(str(w) for w in pred_offset)+']'
            original_text =test_texts[i]

            ##added code

            list_decoded_text_toxicity=[]

            only_decoded_text = []

            if(len(original_text)==0):
                only_decoded_text=[]
            else:
                encoded_text = tokenizer.encode(original_text)
                only_decoded_text = tokenizer.convert_ids_to_tokens(encoded_text)
                decoded_actual_text=[tokenizer.convert_tokens_to_string([i]) for i in
                              tokenizer.convert_ids_to_tokens(encoded_text)]

                if(len(pred_offset)==0):
                    list_decoded_text_toxicity=decoded_actual_text
                else:
                    list_decoded_text_toxicity=convert_to_original_text_with_toxicity(encoded_text,pred_offset,tokenizer)

                only_decoded_text_str = ' '.join(str(x) for x in only_decoded_text)
                only_decoded_text_list=[only_decoded_text_str]
                decoded_text_toxicity_to_string=' '.join(str(y) for y in list_decoded_text_toxicity)

                list_decoded_text_toxicity_to_string=[decoded_text_toxicity_to_string]


            retro_output.append([original_text,list_decoded_text_toxicity_to_string, gold, predicted])


    avg_precision_0 = np.mean(np.array(instance_prec_0))
    avg_recall_0 = np.mean(np.array(instance_recall_0))
    avg_f1_0 = np.mean(np.array(instance_f1_0))

    avg_precision_1 = np.mean(np.array(instance_prec_1))
    avg_recall_1 = np.mean(np.array(instance_recall_1))
    avg_f1_1 = np.mean(np.array(instance_f1_1))

    return avg_precision_0, avg_recall_0, avg_f1_0, avg_precision_1, avg_recall_1, avg_f1_1, retro_output