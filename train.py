#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import numpy as np

import keras
from keras.models import Sequential
from keras.layers.core import Reshape, Flatten
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers.noise import AlphaDropout
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
import sys

STATES={}

CODE_SIZE={
    'dany_code':8960,
    'image_code':4096,
    'word_code':200
}


layers=[('dany_code',[64]),('image_code',[64] ),('word_code',[64])]
layers_deep=[32]
    #model.fit([dany_code, word_code, image_code], salida, batch_size=100,epochs=20)

# Función principal (interfaz con línea de comandos)
if __name__ == '__main__':
    p = argparse.ArgumentParser("redica_i3go+")
    p.add_argument("JSON",
            help="JSON with data")
    p.add_argument("--split",default=20.0,type=float,
        action="store", dest="split",
        help="Percentage of valadation data [20]")
    p.add_argument("--max",default=None,type=int,
        action="store", dest="max",
        help="Maximum number of instances [20000]")
    p.add_argument("--model",default="model.h5",type=str,
                action="store", dest="model",
                            help="Weights model [model.h5]")
    p.add_argument("-v", "--verbose",
            action="store_true", dest="verbose",
            help="Verbose mode [Off]")

    opts = p.parse_args()

    clines=0
    # Train
    with open(opts.JSON) as data_file:   
        for line in data_file:
            clines+=1


    if opts.max and clines>opts.max:
        clines=opts.max

    data=[]
    for ln,lu in layers:
        data.append(np.zeros((clines,CODE_SIZE[ln])))
    data_output=np.zeros(clines)

    i=0

    with open(opts.JSON) as data_file:   
        for line in data_file:
            if i>=clines:
                break
            info=json.loads(line)
            for j,(ln,lu) in enumerate(layers):
                data[j][i,:]=np.array(info[ln],dtype=np.float32)
            data_output[i]=info['klass']
            i+=1

    
    nsplit=int(clines*opts.split/1000)*10
    data_val=[]
    data_output_val=[]
    for i,(ln,lu) in enumerate(layers):
        data_val.append(data[i][:nsplit,]) 
        data[i]=data[i][nsplit:,] 
    data_output_val=data_output[:nsplit]
    data_output=data_output[nsplit:]

    activation='selu'
    #dropout = Dropout
    dropout = AlphaDropout
    dropout_rate = 0.2
    kernel_initializer="lecun_normal"
    optimizer='adam'

    layers_middle=[]
    inputs=[]
    for ln,lus in layers:
        input_layer=Input(shape=(CODE_SIZE[ln],))
        inputs.append(input_layer)
        conn=input_layer
        for units in lus:

            conn = Dense(units,
                    kernel_initializer=kernel_initializer)(conn)
        layers_middle.append(conn)

    merged = concatenate(layers_middle)
    rs=0
    for ln,lu in layers:
        rs+=lu[-1]

    reshape = Reshape((rs,1))(merged)
    conv1=Conv1D(32,3)(reshape)
    maxp1=MaxPooling1D()(conv1)

    conv2=Conv1D(32,3)(maxp1)
    maxp2=MaxPooling1D()(conv2)


    flat=Flatten()(maxp2)
    conn=flat
    for units in layers_deep:
        dense=Dense(units, kernel_initializer=kernel_initializer,
                activation="relu"
            )(conn)
        conn=dropout(dropout_rate)(dense)
    output=Dense(1,activation="sigmoid")(conn)
    model=Model( inputs=inputs,
                 outputs=output)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print(model.summary())

    model.fit(data, data_output, 
            batch_size=100,epochs=220,validation_data=(data_val,data_output_val))

    model.save_weights(opts.model, overwrite=True)

