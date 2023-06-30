# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2023
# Authors: Jaydeb Sarker <jaydebsarker@wayne.edu> and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.


##importing all libraries
import random
import timeit
import argparse

import numpy
from numpy.random import seed
from tensorflow.python.framework.random_seed import set_seed
import os
from tqdm import tqdm
import tensorflow as tf
from  functools import reduce

from keras.layers import Dense,Flatten, Dropout
from  keras import Input, Model
import keras.backend as K

import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split, KFold
import numpy as np

import en_core_web_sm
nlp = en_core_web_sm.load()

seed(1)
set_seed(2)
from pandas import *
##done with import


## import from defined functions from other files
from preprocessing import partition_input_dict, split_paragraph_to_sentences, sentence_preprocessing, generate_output,generate_test_train_index
from post_processings import classify_by_threshold
from metric import compute_performance


from Transformer_Models import tokenizer_collect
from Transformer_Models import get_distilbert,get_bert,get_albert,get_roberta,get_xlnet

##loss function
## we used only binary_loss() for our experiment
def binary_loss(y_true, y_pred):
  loss = -1 * y_true * K.log( K.clip(y_pred + K.epsilon(), 0, 1.0) )
  loss += -1 * (1-y_true) * K.log( K.clip(1-y_pred+K.epsilon(), 0, 1.0) )
  return loss

def weighted_mse(y_true, y_pred):
    mse= tf.keras.losses.MeanSquaredError()
    loss =mse(y_true, y_pred).numpy()
    return loss


## this is for a full token list with 239 tokens
## these tokens are used to add with pre-trained tokenizers
## after trained with the tokens, pretrained tokenizers can provide a full word tokenization and improvs expalinability
new_token_data = read_csv("models/full_token_list.csv")
new_token_list=new_token_data['words'].tolist()

## Main Class of ToxiSpanSE

class ToxiSpanSE:
    def __init__(self, ALGO="BERT", Tokenizer="bert",
                 model_file="models/CR_full_span_dataset.xlsx",
                 load_pretrained=False, max_len=70):
        self.classifier_model = None
        self.modelFile = model_file
        self.Tokenizer=Tokenizer
        self.ALGO = ALGO
        self.data = self._read_data_from_file(model_file)
        self.load_pretrained = load_pretrained
        self.max_length=max_len

    def _read_data_from_file(self, model_file):
        dataframe =pd.read_excel(model_file)
        dataframe.sample(frac=1).reset_index(drop=True) # call a random shuffle
        return  dataframe

    ##get the DL models
    def get_model(self, params, max_length):
        ALGO = self.ALGO.upper()
        training = params['training']
        if(ALGO=="BERT"):
            return get_bert(training, max_length)
        elif(ALGO=="DBERT"):
            return get_distilbert(training, max_length)
        elif (ALGO == "ALBERT"):
            return get_albert(training, max_length)
        elif (ALGO == "ROBERTA"):
            return get_roberta(training, max_length)
        elif (ALGO == "XLNET"):
            return get_xlnet(training, max_length)
        else:
            print("Unknown algorithm: " + ALGO)
            exit(1)

    ##define tokenizer
    def get_tokenizer(self):
        tokenizer = tokenizer_collect(self.Tokenizer)
        return tokenizer


    def get_training_data(self):
        return self.data


## define the 10-fold CV
def ten_fold_cross_validation(toxicClassifier, rand_state=42, max_length=70, threshold =None):
        dataset = toxicClassifier.get_training_data()
        if threshold:
            threshold=float(threshold)
        random_folding = KFold(n_splits=10, shuffle=True, random_state=rand_state)

        dataset["spans"] = dataset["spans"].apply(literal_eval)
        tokenizer =toxicClassifier.get_tokenizer()

        ## add a large list of token to each tokenizer for more accurtate tokenization
        ##one can skip this part by commenting out next line
        tokenizer.add_tokens(new_token_list)

        data_input, input_texts, data_output, initial_weights,tokens = sentence_preprocessing(dataset, max_length, tokenizer)


        toxic_words = []
        for i in range(data_output.shape[0]):
            for j in range(max_length):
                if data_output[i, j] == 1 and initial_weights[i, j] == 1:
                    toxic_words.append(tokenizer.convert_ids_to_tokens(tokens.input_ids[i][j]))
        ##one can check top toxic words by executing following 2 commented lines
        #toxic_words_freq = pd.Series(toxic_words).value_counts()
        #print(toxic_words_freq[:15])

        #starting of 10-fold from here
        count = 1
        error_analysis_output =[]
        results =""
        for train_index, test_index in random_folding.split(data_output):
            start = timeit.default_timer()
            print("Using split-" + str(count) + " as test data..")


            train_input_set = partition_input_dict(data_input, train_index)
            test_input = partition_input_dict(data_input, test_index)
            test_texts =input_texts[test_index,]
            train_output = data_output[train_index,]
            test_output = data_output[test_index,]

            params = {
                "training": True,
                "learning_rate": 1e-5,
                "pre_training_epoch": 0,
                "initial_epoch": 0,
                "epochs": 30,
                "silver": False,
                     }

            training_index, validation_index = generate_test_train_index(train_output.shape[0], 0.1)
            val_texts=input_texts[validation_index,]
            model_training_input = partition_input_dict(train_input_set, training_index)
            validation_input = partition_input_dict(train_input_set, validation_index)
            model_training_output = train_output[training_index]
            validation_output = train_output[validation_index]

            model = toxicClassifier.get_model(params, max_length)
            optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

            model.compile(optimizer=optimizer, loss=binary_loss)

            #save the model by following name
            checkpoint_path = str(toxicClassifier.ALGO)+".h5"
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True,
                                                            save_weights_only=True, verbose=1)

            es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
            sample_weight = initial_weights[training_index]

            model.fit(model_training_input, model_training_output, epochs=params['epochs'],
                      initial_epoch=params['initial_epoch'],
                      validation_data=(validation_input, validation_output),
                      sample_weight=sample_weight,
                      callbacks=[es_callback])


            #finding the test prediction probability
            test_pred = model.predict(test_input, verbose=1, batch_size=64)
            stop = timeit.default_timer()
            time_elapsed = stop - start


            # this is for validation set prediction
            #when threshold will be None from the argument, it will predic the validation set
            if threshold is None:
                val_pred = model.predict(validation_input, verbose=1, batch_size=64)
                best_threshold=0.05
                best_fscore=0.00
                for probablity in np.arange(0.01,0.99, 0.01):
                    prediction_label = classify_by_threshold(val_pred, probablity)
                    avg_precision_0, avg_recall_0, avg_f1_0, avg_precision_1, avg_recall_1, avg_f1_1, retro_output\
                        = compute_performance(
                    validation_output, prediction_label, test_texts,tokenizer, max_length, False)

                    results = results + str(count) + "," + str(probablity)+ "," + str(toxicClassifier.ALGO)+","
                    results = results + str(avg_precision_0) + "," + str(avg_recall_0) + "," + str(avg_f1_0)
                    results = results + "," + str(avg_precision_1) + "," + str(avg_recall_1) + "," + str(avg_f1_1) + ","+ str(time_elapsed)+"\n"

                    if avg_f1_1>best_fscore:
                        best_threshold=probablity
                        best_fscore=avg_f1_1
                print("Best f1 toxic: "+ str(best_fscore)+ "at threshold: "+str(best_threshold))

            #this is for test set when user put a threshold value in argument
            else:
                prediction_label = classify_by_threshold(test_pred, threshold)
                avg_precision_0, avg_recall_0, avg_f1_0, avg_precision_1, \
                avg_recall_1, avg_f1_1, retro_output = compute_performance(test_output, prediction_label, test_texts,tokenizer, max_length, True)
                error_analysis_output.append(retro_output)


                print("Precision non-toxic: " + str(avg_precision_0))
                print("Recall non-toxic: " + str(avg_recall_0))
                print("F1 non-toxic: " + str(avg_f1_0))
                print("Precision toxic: " + str(avg_precision_1))
                print("Recall toxic: " + str(avg_recall_1))
                print("F1 toxic: " + str(avg_f1_1))

                results = results + str(count) + "," +str(threshold)+","+ str(toxicClassifier.ALGO)+","
                results = results + str(avg_precision_0) + "," + str(avg_recall_0) + "," + str(avg_f1_0)
                results = results + "," + str(avg_precision_1) + "," + str(avg_recall_1) + "," + str(avg_f1_1) + ","+ str(time_elapsed)+"\n"
            count = count + 1

        return results, error_analysis_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ToxiSpanSE: An Explainable Toxicity detector for code review comments')

    parser.add_argument('--algo', type=str,
                        help='Classification algorithm. Choices are: BERT| ALBERT| DBERT|ROBERTA |XLNET',
                        default="BERT")
    parser.add_argument('--tokenizer', type=str,
                        help='Choices are: bert| albert| dbert|roberta|xlnet',
                        default="bert")

    parser.add_argument('--repeat', type=int, help='Iteration count', default=1) #default 1
    parser.add_argument('--threshold', help='Set a Threshold value', default=0.1)  # default 0.1
    parser.add_argument('--vary', help='Varying threshold',
                        action='store_true', default=False)  # default False, will not predict on test set
    parser.add_argument('--retro', help='Print missclassifications',
                        action='store_true', default=False)  # default False, will not write

    parser.add_argument('--mode', type=str,
                        help='Execution mode. Choices are: eval | pretrain | tuning',
                        default="eval")                      # not implemented yet
    args = parser.parse_args()

    print(args)
    ALGO = str(args.algo).upper()
    tokenizer = str(args.tokenizer)
    THRESHOLD=float(args.threshold)
    REPEAT = args.repeat

    mode = args.mode

    toxicClassifier = ToxiSpanSE(ALGO=ALGO, Tokenizer=tokenizer)


    filename = "cross-validation-" + ALGO + "-" + str(args.tokenizer) + ".csv"
    debug_file_name = "error-analysis-" + ALGO + "-" + str(args.tokenizer) + ".xlsx"
    training_log = open(filename, 'w')
    training_log.write("Fold,Threshold,Algo,precision_0,recall_0,f-score_0,precision_1,recall_1,f-score_1,time\n")

    random.seed(999)
    for k in range(0, REPEAT):
        print(".............................")
        print("Run# {}".format(k))
        if args.vary:
            results, debug_info=ten_fold_cross_validation(toxicClassifier, random.randint(1, 10000))
        else:
            results, debug_info =ten_fold_cross_validation(toxicClassifier, random.randint(1, 10000),threshold=THRESHOLD)
        training_log.write(results)  ##writing results on file


        ##when retro is true, it will print misclassification
        if args.retro:
            debug_info = reduce(lambda x,y: x+y, debug_info)
            debug_df=pd.DataFrame(debug_info, columns=["text","explainable_toxicity","labeled","predicted"])
            debug_df.to_excel(debug_file_name)

    ##########################
    training_log.close()
