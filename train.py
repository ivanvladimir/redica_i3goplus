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

    p.add_argument("-v", "--verbose",
            action="store_true", dest="verbose",
            help="Verbose mode [Off]")

    opts = p.parse_args()

    data=[]
    clines=0
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

    n_dense=3
    dense_units=64
    dense_units_dany= 256
    dense_units_word=64
    dense_units_image=128
    activation='selu'
    dropout = AlphaDropout
    dropout_rate = 0.1
    kernel_initializer="lecun_normal"
    optimizer='adam'

    input_dany= Input(shape=(DANY_CODE_SIZE,))
    input_word= Input(shape=(WORD_CODE_SIZE,))
    input_image= Input(shape=(IMAGE_CODE_SIZE,))


    layer_dany = Dense(dense_units_dany,
                    kernel_initializer=kernel_initializer)(input_dany)

    layer_word = Dense(dense_units_word,
                    kernel_initializer=kernel_initializer)(input_word)

    layer_image = Dense(dense_units_image,
                    kernel_initializer=kernel_initializer)(input_image)

    merged = concatenate([layer_dany,layer_word,layer_image])

    dense_1=Dense(dense_units, kernel_initializer=kernel_initializer,
                activation="selu"
            )(merged)
    dropout_1=AlphaDropout(dropout_rate)(dense_1)
    dense_2=Dense(dense_units, kernel_initializer=kernel_initializer,
                activation="selu"
            )(dropout_1)
    dropout_2=AlphaDropout(dropout_rate)(dense_2)
    dense_3=Dense(dense_units, kernel_initializer=kernel_initializer,
                activation="selu"
            )(dropout_2)
    dropout_3=AlphaDropout(dropout_rate)(dense_3)

    output=Dense(1,activation="sigmoid")(dropout_3)
    model=Model( inputs=[input_dany,input_word,input_image],
                 outputs=output)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print(model.summary())

    model.fit([dany_code, word_code, image_code], salida, batch_size=100,epochs=20)






