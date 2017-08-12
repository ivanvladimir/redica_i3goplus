#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import numpy as np

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers.noise import AlphaDropout
from keras.layers.merge import concatenate
import sys

STATES={}

DANY_CODE_SIZE=8960
IMAGE_CODE_SIZE=4096
WORD_CODE_SIZE=200

# Función principal (interfaz con línea de comandos)
if __name__ == '__main__':
    p = argparse.ArgumentParser("redica_i3go+")

    p.add_argument("JSON",
            help="JSON with data")
    p.add_argument("--model",default="model.h5",type=str,
                action="store", dest="model",
                            help="Weights model [model.h5]")
    p.add_argument("-v", "--verbose",
            action="store_true", dest="verbose",
            help="Verbose mode [Off]")

    opts = p.parse_args()

    data=[]
    clines=0
    # Train
    with open(opts.JSON) as data_file:   
        for line in data_file:
            clines+=1

    dany_code=np.zeros((clines,DANY_CODE_SIZE))
    image_code=np.zeros((clines,IMAGE_CODE_SIZE))
    word_code=np.zeros((clines,WORD_CODE_SIZE))

    salida=np.zeros(clines)

    i=0
    with open(opts.JSON) as data_file:   
        for line in data_file:
            j_=json.loads(line)
            dany_code[i,:]=np.array(j_['dany_code'],dtype=np.float32)
            image_code[i,:]=np.array(j_['image_code'],dtype=np.float32)
            word_code[i,:]=np.array(j_['word_code'],dtype=np.float32)
            salida[i]=j_['klass']
            i+=1


    nsplit=4000
    dany_code_val=dany_code[:nsplit,]
a    image_code_val=image_code[:nsplit,]
    word_code_val=word_code[:nsplit,]
    salida_val=salida[:nsplit]
    dany_code=dany_code[nsplit:,]
    image_code=image_code[nsplit:,]
    word_code=word_code[nsplit:,]
    salida=salida[nsplit:]


    n_dense=3
    dense_units=64
    dense_units_dany= 128
    dense_units_word=64
    dense_units_image=128
    activation='selu'
    dropout = AlphaDropout
    dropout_rate = 0.2
    kernel_initializer="lecun_normal"
    optimizer='adam'

    input_dany= Input(shape=(DANY_CODE_SIZE,))
    input_word= Input(shape=(WORD_CODE_SIZE,))
    input_image= Input(shape=(IMAGE_CODE_SIZE,))


    layer_dany = Dense(dense_units_dany,
                    kernel_initializer=kernel_initializer)(input_dany)

    layer_word = Dense(dense_units_word,
                    kernel_initializer=kernel_initializer)(input_word)

    #layer_image = Dense(dense_units_image,
    #                kernel_initializer=kernel_initializer)(input_image)

    #merged = concatenate([layer_dany,layer_word,layer_image])
    merged = concatenate([layer_dany,layer_word])

    dense_1=Dense(64, kernel_initializer=kernel_initializer,
                activation="relu"
            )(merged)
    dropout_1=AlphaDropout(dropout_rate)(dense_1)
    dense_2=Dense(64, kernel_initializer=kernel_initializer,
                activation="relu"
            )(dropout_1)
    dropout_2=AlphaDropout(dropout_rate)(dense_2)
    dense_3=Dense(32, kernel_initializer=kernel_initializer,
                activation="relu"
            )(dropout_2)
    dropout_3=AlphaDropout(dropout_rate)(dense_3)

    output=Dense(1,activation="sigmoid")(dropout_3)
    #model=Model( inputs=[input_dany,input_word,input_image],
    model=Model( inputs=[input_dany,input_word],
                 outputs=output)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print(model.summary())

    #model.fit([dany_code, word_code, image_code], salida, batch_size=100,epochs=20)
    model.fit([dany_code, word_code], salida, 
            batch_size=100,epochs=220,validation_data=([dany_code_val,word_code_val],salida_val))

    model.save_weights(opts.model, overwrite=True)

