'''
Authors: Jeff Adrion, Andrew Kern, Jared Galloway

These are some network architectures that we are using for the
inference of Rho through images of the genotype matrices.

'''

from imports import *

def CNN2D(x,y):

    batch_size = x.shape[0]
    num_nodes = x.shape[1]
    window_size = x.shape[2]
    filters = x.shape[3]

    img_1_inputs = Input(shape=(num_nodes,window_size,filters))

    h = layers.Conv2D(64, kernel_size=(4,4), activation='relu', name='conv1_1')(img_1_inputs)
    #h = layers.Conv1D(256, kernel_size=2, dilation_rate=1, activation='relu')(h)
    h = layers.AveragePooling2D(pool_size=2)(h)
    h = layers.Dropout(0.25)(h)
    h = layers.Flatten()(h)
    h = layers.Dense(32)(h)
    h = layers.Dropout(0.25)(h)
    output = layers.Dense(1,kernel_initializer='normal',name="out",activation='linear')(h)

    model = Model(inputs=[img_1_inputs], outputs=[output])
    model.compile(loss='mse', optimizer='adam')
    model.summary()

    return model
    

def CNN2D_S(x,y):
        
    batch_size = x.shape[0]
    num_nodes = x.shape[1]
    window_size = x.shape[2]
    filters = x.shape[3]

    model = Sequential()
    model.add(layers.Conv2D(64, kernel_size=(3, 3),
    #model.add(layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     #activation='linear',
                     input_shape=(num_nodes,window_size,filters)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2)) #less dropout
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    #model.add(layers.Dense(64, activation='linear'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(1,kernel_initializer='normal',name="out",activation='linear'))

    model.compile(loss='mse', optimizer='adam')
    model.summary()

    return model    
    


