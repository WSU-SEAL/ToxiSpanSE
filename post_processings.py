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
import  pandas as pd
from itertools import chain

##making binary (0 or 1) value using a threshold (float 0~1) value
def classify_by_threshold(prediction, thresh):
  predictions = np.zeros(prediction.shape)
  for i in range(prediction.shape[0]):
    for j in range(prediction.shape[1]):
      if prediction[i,j]>=thresh:
        predictions[i,j]=1
  return predictions


## this function is for text encoding which provides the explainable output
def convert_to_original_text_with_toxicity(encoded_text,pred,tokenizer):
  ##list_decoded_text_toxicity = []
  only_decoded_text = tokenizer.convert_ids_to_tokens(encoded_text)
  decode = [tokenizer.convert_tokens_to_string([i]) for i in tokenizer.convert_ids_to_tokens(encoded_text)]
  decoded_new = []
  prev = 0

  if (len(pred) == 0):
    decoded_new = decode
  else:
    #if (pred[0] == 0):
      #decoded_new.append(['<toxic>'])
    for i in range(0, len(pred)):
      pred_val = int(pred[i])

      ##for handling exception
      if (pred_val >= len(decode)):
        print("Missing here", only_decoded_text, pred)
        break
      if (pred_val > prev and prev > 0):
        decoded_new.append(['</toxic>'])
      decoded_new.append(decode[prev:pred_val])

      if (pred_val - prev > 0):
        decoded_new.append(['<toxic>'])

      decoded_new.append([decode[pred_val]])

      #decoded_new.append([decode[pred_val]])
      # if(pred_val - prev >0): decoded_new.append(['</toxic>'])
      prev = pred_val + 1
    decoded_new.append(['</toxic>'])
    decoded_new.append(decode[prev:len(decode)])

  decoded_new = list(chain.from_iterable(decoded_new))
  return decoded_new


## if we want to use BIO for prediction
# we do not use it in our experiment
def classify_bio_output(prediction):
  num_samples = prediction.shape[0]*prediction.shape[1]
  predictions = np.zeros((prediction.shape[0],prediction.shape[1]))
  for i in range(prediction.shape[0]):
    for j in range(prediction.shape[1]):
      if prediction[i,j] != 0:
        predictions[i,j] = 1
  return predictions





