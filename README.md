coloncancerneuralnetwork
======

This repository contains the source code for a machine learning 
algorithm using keras and the SEER cancer data. 
The app asks for input from a user and outputs the predicted survival curve based on the input data.

The model in the COLONPICKLNN folder is a keras
 nerual network that implements the transformation of censored
 data to a form appropriate for machine learning algorithms as described here:

http://www.benkuhn.net/survival-trees


modelcolon = Sequential()
modelcolon.add(Dense(114, input_shape=(102,) ,init='normal'))
modelcolon.add(Activation('relu'))
modelcolon.add(Dropout(0.05))
modelcolon.add(Dense(50, init='normal'))
modelcolon.add(Activation('relu'))
modelcolon.add(Dropout(0.05))


modelcolon.add(Dense(35, init='normal'))
modelcolon.add(Activation('relu'))
modelcolon.add(Dropout(0.05))


modelcolon.add(Dense(2, init='normal'))
modelcolon.add(Activation('softmax'))



rms = RMSprop(lr=0.001)


modelcolon.compile(loss='binary_crossentropy', optimizer=rms, class_mode="binary")





# coloncancerneuralnetwork
# coloncancerneuralnetwork 
"# coloncancernnerrors" 
"# testtestcolon" 
