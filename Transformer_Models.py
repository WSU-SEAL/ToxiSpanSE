# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2023
# Authors: Jaydeb Sarker <jaydebsarker@wayne.edu> and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.


from transformers import AutoConfig, AutoModel,AutoTokenizer
from transformers import DistilBertTokenizer, TFDistilBertModel
from keras.layers import Dense,Flatten, Dropout
from  keras import Input, Model
import keras.backend as K
from transformers import TFBertModel
from transformers import TFRobertaModel,  TFAlbertModel, TFXLNetModel


##BERT-base model
def get_bert(training, max_length):
  input_ids = Input(shape=(max_length),name="input_ids",dtype="int32")
  token_type_ids = Input(shape=(max_length),name="token_type_ids")
  attention_mask = Input(shape=(max_length),name="attention_mask")

  encoder = TFBertModel.from_pretrained('bert-base-uncased')

  embeddings = encoder({"input_ids":input_ids,"token_type_ids":token_type_ids,"attention_mask":attention_mask}, training=training)[0]

  dense_layer2=Dense(1,activation='sigmoid') (embeddings, training=training)
  outputs = Flatten()(dense_layer2)


  return Model(inputs = [input_ids, token_type_ids, attention_mask ] ,outputs = outputs)

##DistilBERT model
def get_distilbert(training, max_length):
  input_ids = Input(shape=(max_length),name="input_ids",dtype="int32")
  token_type_ids = Input(shape=(max_length),name="token_type_ids")
  attention_mask = Input(shape=(max_length),name="attention_mask")

  encoder = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

  embeddings = encoder({"input_ids":input_ids,"token_type_ids":token_type_ids,"attention_mask":attention_mask}, training=training)[0]

  dense_layer2=Dense(1,activation='sigmoid') (embeddings, training=training)
  outputs = Flatten()(dense_layer2)


  return Model(inputs = [input_ids, token_type_ids, attention_mask ] ,outputs = outputs)

##RoBERTa model
def get_roberta(training, max_length):
  input_ids = Input(shape=(max_length),name="input_ids",dtype="int32")
  token_type_ids = Input(shape=(max_length),name="token_type_ids")
  attention_mask = Input(shape=(max_length),name="attention_mask")

  encoder = TFRobertaModel.from_pretrained("roberta-base")

  embeddings = encoder({"input_ids":input_ids,"token_type_ids":token_type_ids,"attention_mask":attention_mask}, training=training)[0]

  dense_layer2=Dense(1,activation='sigmoid') (embeddings, training=training)
  outputs = Flatten()(dense_layer2)
  return Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=outputs)

#ALBERT model
def get_albert(training, max_length):
  input_ids = Input(shape=(max_length),name="input_ids",dtype="int32")
  token_type_ids = Input(shape=(max_length),name="token_type_ids")
  attention_mask = Input(shape=(max_length),name="attention_mask")

  encoder = TFAlbertModel.from_pretrained("albert-base-v2")

  embeddings = encoder({"input_ids":input_ids,"token_type_ids":token_type_ids,"attention_mask":attention_mask}, training=training)[0]

  dense_layer2=Dense(1,activation='sigmoid') (embeddings, training=training)
  outputs = Flatten()(dense_layer2)
  return Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=outputs)


##xlnet model
def get_xlnet(training, max_length):
  input_ids = Input(shape=(max_length),name="input_ids",dtype="int32")
  token_type_ids = Input(shape=(max_length),name="token_type_ids")
  attention_mask = Input(shape=(max_length),name="attention_mask")

  encoder = TFXLNetModel.from_pretrained("xlnet-base-cased")

  embeddings = encoder({"input_ids":input_ids,"token_type_ids":token_type_ids,"attention_mask":attention_mask}, training=training)[0]

  dense_layer2=Dense(1,activation='sigmoid') (embeddings, training=training)
  outputs = Flatten()(dense_layer2)
  return Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=outputs)

##collecting the pretrained tokenizer
def tokenizer_collect(Tokenizer):
    Tokenizer=Tokenizer.lower()
    if (Tokenizer == 'bert'):
        return AutoTokenizer.from_pretrained("bert-base-uncased")
    elif (Tokenizer == 'roberta'):
        return AutoTokenizer.from_pretrained("roberta-base",add_prefix_space=True)
    elif (Tokenizer == 'dbert'):
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")
    elif (Tokenizer == 'albert'):
        return AutoTokenizer.from_pretrained("albert-base-v2")
    elif (Tokenizer == 'deberta'):
        return AutoTokenizer.from_pretrained("kamalkraj/deberta-base")
    elif (Tokenizer == 'xlnet'):
        return AutoTokenizer.from_pretrained("xlnet-base-cased")
    else:
        print("Wrong tokenizer selected")

