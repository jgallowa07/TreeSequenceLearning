'''
Authors: Jeff Adrion, Andrew Kern, Jared Galloway

These are some network architectures that we are using for the
inference of Rho through images of the genotype matrices.

'''

from imports import *

def CNN1D(inputShape):

    img_1_inputs = Input(shape=(inputShape[0][1],inputShape[0][2]))

    h = layers.Conv1D(1250, kernel_size=2, activation='relu', name='conv1_1')(img_1_inputs)
    h = layers.Conv1D(512, kernel_size=2, dilation_rate=1, activation='relu')(h)
    h = layers.AveragePooling1D(pool_size=2)(h)
    h = layers.Dropout(0.25)(h)
    h = layers.Conv1D(512, kernel_size=2, activation='relu')(h)
    h = layers.AveragePooling1D(pool_size=2)(h)
    h = layers.Dropout(0.25)(h)
    h = layers.Flatten()(h)

    loc_input = Input(shape=(inputShape[1][1],))
    m2 = layers.Dense(64,name="m2_dense1")(loc_input)
    m2 = layers.Dropout(0.1)(m2)

    h =  layers.concatenate([h,m2])
    h = layers.Dense(128,activation='relu')(h)
    h = Dropout(0.2)(h)
    output = layers.Dense(1,kernel_initializer='normal',name="out_dense",activation='linear')(h)

    model = Model(inputs=[img_1_inputs,loc_input], outputs=[output])
    model.compile(loss='mse', optimizer='adam')
    model.summary()

    return model
    

