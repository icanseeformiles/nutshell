import numpy as np
import pandas as pd

from keras.layers import Dense, Embedding, Dropout, Reshape, Merge, Input, LSTM, concatenate
from keras.layers import TimeDistributed
from keras.models import Sequential, Model 
from keras.optimizers import Adam, Adamax
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras import utils

class LearningData:
    
    def __init__(self, inputdata = pd.DataFrame()):
        self.inputdata = inputdata 
        self.prepdata = pd.DataFrame()
        self.trainingfeatures = []
        self.traininglabels = []
        self.validationfeatures = []
        self.validationlabels = []
        self.labelcolumn = ''
        self.categorycolumns = [] #columns are either category or numeric but not both
        self.numericcolumns = []
        self.sequencecolumns = [] # sequence columns are also in category or numeric column list
        self.sequencelength = 1 # max/pad length of sequences for training - all sequential cols have same length
        self.valueindex = {} # dictionary of dictionaries - key is column name
        self.indexvalue = {} # above only in reverse
        self.validationsplit = .10
        self.maxvalidation = 100000
    
    def unique_column_values(self, dataSeries, isSequence=False):
        # return a list of all unique values in series/column
        # if each value is actually a list of values, handle that
        
        uniqueValues = []

        # add two addition values for unknown and padding values
        # unknown will only be used during inference, not training
        uniqueValues.append('<pad>') # should be 0 index
        uniqueValues.append('<unk>') # should be 1 index
        
        if isSequence:
            seq = []
            for r in dataSeries.iteritems():
                for v in r[1]:
                    seq.append(v)
            uniqueValues = uniqueValues + list(set(seq))
        else:
            uniqueValues = uniqueValues + list(set(dataSeries))
            
        return uniqueValues
    
    def column_values_to_index(self, dataSeries, columnName, isSequence=False):
        # take values in one column and changes them all to their zero-based index equivalent
        # if each value is actually a list of values, handle that
        # return a list of converted values
        
        indexList = []
        if isSequence: # create a list of lists
            for l in dataSeries:
                seq = []
                for v in l:
                    seq.append(self.valueindex[columnName][v])
                indexList.append(seq)
        else:
            for v in dataSeries:
                indexList.append(self.valueindex[columnName][v])
                
        return indexList
    
    def add_false_rows(self, defaceColumns):
        # add negative samples to training data by defacing specific columns with false values
        # label the new rows false (0)
        
        #self.prepdata[self.labelcolumn] = 1 # label all true examples - dont assume this
        
        dfFalse = self.prepdata.copy(deep=True) # copy all true examples as a starting point (dont use pandas copy)
        dfFalse[self.labelcolumn] = 0 # label all false examples
        
        for colName in defaceColumns:
            dfFalse[colName] = self.deface_column(dfFalse[colName], colName in self.sequencecolumns, .15)
                
        # add false rows to training data
        self.prepdata = pd.concat([self.prepdata, dfFalse], ignore_index=True)
        
    
    def deface_column(self, dataSeries, isSequence=False, modifySequence=.10):
        # to create negative/false samples with value distribution similar to input set
        # for sequences, change a random x% of values in the sequence - where x = modifySequence

        false_list = []
        
        # shuffle true value column to select random specimin values
        # using real values from a random row will maintain proper distribution of values
        indices = np.arange(0,len(dataSeries))
        np.random.shuffle(indices)
                
        for i in range(0,len(indices)):
            s = dataSeries[indices[i]] # random specimin from within true data set
            r = dataSeries[i].copy() # must make a copy, otherwise we are updating the original list object reference
            if isSequence:
                # calc number of sequence values to modify (at least 1)
                for m in range(0, max(int(len(r) * modifySequence), 1)):
                    rpos = np.random.randint(0,len(r)) # choose a random position in seq to modify
                    spos = np.random.randint(0,len(s)) # choose random position in specimin to use for mod
                    r[rpos] = s[spos]
                false_list.append(r) # append true sequence with modifications
            else:
                false_list.append(s) # append the specimin value overwriting true value
        
        return false_list      
    
    def prepare_data(self):

        #initialize prepared data frame - this is the set that will be split into train and validation parts
        self.prepdata = pd.DataFrame()
        
        #convert category column values to index values
        print("Converting category columns...")
        for colname in self.categorycolumns:
            print(colname)
            uniqueValues = self.unique_column_values(self.inputdata[colname], colname in self.sequencecolumns)            
            self.valueindex[colname] = dict((c,i) for i, c in enumerate(uniqueValues))
            self.indexvalue[colname] = dict((i,c) for i, c in enumerate(uniqueValues))
            # write new column to training data set
            self.prepdata[colname] = self.column_values_to_index(self.inputdata[colname], colname, colname in self.sequencecolumns)
        
        # normalize numeric columns
        print("Normalizing numeric columns...")
        for colname in self.numericcolumns:
            # don't normalize label column even if it is in the list
            if colname == self.labelcolumn:
                continue
            
            print(colname)
            self.prepdata[colname] = utils.normalize(self.inputdata[colname].values) ## wont work if there are any nan values
        
        # add label column to training set
        self.prepdata[self.labelcolumn] = self.inputdata[self.labelcolumn]
        
        print('Done preparing data')
        
    def dataframe_to_input(self, dataframe):
        # convert each dataframe column to a seperate numpy array in a list suitable for keras input tensor
        # for sequence columns, truncate/pad each value to uniform length 
        # pad value is 0 because we defined that in our valueindex dictionary (0 = padding; 1 = unknown)
        # dont forget to set sequencelength before calling this!
        
        input_list = []
        for col in dataframe:
            if col in self.sequencecolumns:
                padseq = sequence.pad_sequences(dataframe[col].values, maxlen=self.sequencelength, \
                                  padding='post', truncating='post', value=0)
                input_list.append(padseq)

            else:
                input_list.append(dataframe[col].values)
            
        return input_list 

    
    def split_data(self, shuffle=False):
        
        # split prepdata into training and validation set
        # it's best to shuffle the order of the data rows so validation set is random sample
        #  unless there is some reason not to e.g. train on jan-may, validate on june
        
        # build list of columns in same order as model inputs will be added
        featureNames = []
        trainingData = None
        validationData = None
        for c in self.categorycolumns:
            if c != self.labelcolumn:
                featureNames.append(c)
        for c in self.numericcolumns:
            if c != self.labelcolumn:
                featureNames.append(c)
              
        validationRows = min(int(self.validationsplit * len(self.prepdata)), self.maxvalidation)

        if shuffle:
            indices = np.array(self.prepdata.index.values.copy())
            np.random.shuffle(indices)
            trainIndices = indices[0:-validationRows]
            valIndices = indices[-validationRows:]
            trainingData = self.prepdata.iloc[trainIndices]
            validationData = self.prepdata.iloc[valIndices]
        else:
            trainingData = self.prepdata[:-validationRows]
            validationData = self.prepdata[-validationRows:]
            
        # convert dataframe values to keras input
        self.trainingfeatures = self.dataframe_to_input(trainingData[featureNames])
        self.traininglabels = self.dataframe_to_input(trainingData[[self.labelcolumn]])
        self.validationfeatures = self.dataframe_to_input(validationData[featureNames])
        self.validationlabels = self.dataframe_to_input(validationData[[self.labelcolumn]])
        
        print('Training examples:',len(self.trainingfeatures[0]))
        print('Validation examples:',len(self.validationfeatures[0]))

        
class DeepLearner:
    
    def __init__(self, learningdata):       
        self.learningdata = learningdata
        self.hiddenlayers = 2 # number of hidden dense/lstm layer sets to add not including rep/2d layers - min=1
        self.labeltype = 'binary' # or category
        self.model = None
        self.categoryinputs = []
        self.categoryfactors = {} # embedding factors - each value corresponds to categoryinputs value 
        self.numericinputs = []
        self.outputfactors = 50 # number of factors in the representational output vector
        self.batchsize = 32 # number of rows to process in a batch during training or inference. high number = faster + more GPU memory usage
        self.lstmunits = 24 # number of LSTM memory cells
        self.dropoutrate = .20 # default dropout percentage for hidden layers
        self.GPU = False # whether or not GPU batching is being used - LSTM setting
        #self.sequentialinput = len(self.learningdata.sequencecolumns) > 0

        # populate category and numeric input names from learning data - not including label column
        d = self.learningdata
        for c in d.categorycolumns:
            if c != d.labelcolumn:
                self.categoryinputs.append(c)
                # determine default number of embedding factors for each category 
                # default is the smaller of cardinality / 2 or 50 - but at least 1
                cardinality = len(self.learningdata.indexvalue[c])
                self.categoryfactors[c] = max(min(int(cardinality/2), 50),1)
        for c in d.numericcolumns:
            if c!= d.labelcolumn:
                self.numericinputs.append(c)
                
    def build_model(self):

        # paths vary for category vs numeric data as well as for sequential vs non-squential data
        # so building layer flow is a bit tricky
        
        d = self.learningdata
        inputColumns = self.categoryinputs + self.numericinputs
        
        inputLayers = {}
        embeddingLayers = {}
        factorLayers = {}
        #factorLayersReshape = {}
        
        # add input layers
        for c in inputColumns:
            inputLayers[c] = Input(shape=(d.sequencelength if c in d.sequencecolumns else 1,), name='input_' + c)
        
        # add embedding layers for all categorical inputs
        for c in self.categoryinputs:
            embeddingLayers[c] = Embedding(input_dim = len(d.indexvalue[c]), \
                                                output_dim = self.categoryfactors[c], \
                                                input_length = d.sequencelength, \
                                                name='embed_' + c)
    
        # add factor layers - joining category input layer to category embedding layer + Reshape
        for c in self.categoryinputs:
            #factorLayers[c] = Reshape((d.sequencelength, self.categoryfactors[c],)) (embeddingLayers[c](inputLayers[c]))
            factorLayers[c] = embeddingLayers[c](inputLayers[c])
           
        # merge inputs into two seperate merge layers that will go down seperate paths (sequential & non-sequential)
                            
        # merge sequential inputs (factor layers for categorical and input layers for numeric)
        sequentialLayers = []
        for c in d.sequencecolumns:
            if c in self.categoryinputs:
                sequentialLayers.append(factorLayers[c])
            if c in self.numericinputs:
                sequentialLayers.append(inputLayers[c])
        
        if len(sequentialLayers) > 1:
            mergeSequentialLayer = concatenate(sequentialLayers)
        elif len(sequentialLayers) == 1:
            mergeSequentialLayer = sequentialLayers[0]
        else:
            mergeSequentialLayer = None

        if mergeSequentialLayer is not None:
            print('Sequential Merge Layer Shape: ', mergeSequentialLayer.shape)
            
        # merge non-sequential inputs
        nonSequentialLayers = []
        for c in self.categoryinputs:
            if c not in d.sequencecolumns:
                nonSequentialLayers.append(factorLayers[c])
        for c in self.numericinputs:
            if c not in d.sequencecolumns:
                nonSequentialLayers.append(inputLayers[c])

        if len(nonSequentialLayers) > 1:
            mergeNonSequentialLayer = concatenate(nonSequentialLayers)
        elif len(nonSequentialLayers) == 1:
            mergeNonSequentialLayer = nonSequentialLayers[0]
        else:
            mergeNonSequentialLayer = None
        
        if mergeNonSequentialLayer is not None:
            print('Non-Sequential Merge Layer Shape:', mergeNonSequentialLayer.shape)

        # build LSTM layer set for sequential data
        if mergeSequentialLayer is not None:
            
            lstmLayers = [] 
            lstmDropoutLayers = []
            
            # add n sets of LSTM and Dropout layers
            impl = 2 if self.GPU else 1 
            for i in range(0, self.hiddenlayers):
                    
                lstmLayers.append(LSTM(self.lstmunits, return_sequences=True, \
                                     dropout=self.dropoutrate, recurrent_dropout=self.dropoutrate, \
                                     implementation=impl, activation='relu', \
                                     name='lstm_' + str(i))(mergeSequentialLayer if i==0 else lstmDropoutLayers[i-1]))
                
                lstmDropoutLayers.append(Dropout(self.dropoutrate, name='lstm_dropout_' + str(i))(lstmLayers[i]))
            
            lstmTimeDistLayer = TimeDistributed(Dense(self.lstmunits, name='lstm_dense'), name='lstm_timedist')(lstmDropoutLayers[self.hiddenlayers-1])
            lstmReshapeLayer = Reshape((d.sequencelength * self.lstmunits,), name='lstm_reshape')(lstmTimeDistLayer)

        # build Dense layer set for non-sequential layers
        if mergeNonSequentialLayer is not None:
            
            denseLayers = []
            denseDropoutLayers = []
            
            # add n sets of Dense and Dropout layers
            for i in range(0, self.hiddenlayers):
                
                denseLayers.append(Dense(self.outputfactors, activation='relu', name='dense_' + str(i)) \
                                   (mergeNonSequentialLayer if i==0 else denseDropoutLayers[i-1]))
                
                denseDropoutLayers.append(Dropout(self.dropoutrate, name='dense_dropout_' + str(i)) \
                                         (denseLayers[i]))
            
            # conclude stack with reshape layer
            denseReshapeLayer = Reshape((self.outputfactors,), name='dense_reshape') \
                                       (denseDropoutLayers[self.hiddenlayers-1])
                    
        # merge sequential and non-sequental path results
        if mergeSequentialLayer is not None and mergeNonSequentialLayer is not None:
            mergeFinal = concatenate([lstmReshapeLayer, denseReshapeLayer])    
        elif mergeSequentialLayer is not None and mergeNonSequentialLayer is None:
            mergeFinal = lstmReshapeLayer
        elif mergeSequentialLayer is None and mergeNonSequentialLayer is not None:
            mergeFinal = denseReshapeLayer
        else:
            raise NameError('No learning columns defined. Cannot continue.')
                            
        denseRepresentation = Dense(self.outputfactors, name='dense_representation') (mergeFinal)
        dense2D = Dense(2, name='dense_2d') (denseRepresentation)
        
        # output layer size depends on whether label is binary or category
        if self.labeltype == 'binary':
            denseOutput = Dense(1, name='dense_output') (dense2D)
        else:
            denseOutput = Dense(len(self.learningdata.indexvalue[d.labelcolumn]), name='dense_output') (dense2D) # output values = cardinality of label column

        self.model = Model(inputs=list(inputLayers.values()), outputs=denseOutput)
        self.model.compile(loss='mse', optimizer=Adam(), metrics=['acc'] )
        print(self.model.summary())
        
    def train_model(self, fileName='', epochs=1, superEpochs=1, learningRate=.001, learningRateDivisor=10):
        # use learning data set to train model
        # super epoch is another round of n epochs with the learning rate divided by the divisor
        
        # to-do if user passes filename, load the existing weights first if file exists

        if fileName == '':
            callbacks = [EarlyStopping('val_loss', patience=2)]
        else:
            callbacks = [EarlyStopping('val_loss', patience=2), \
                     ModelCheckpoint(fileName + '_weights.h5', save_best_only=True)]

        d = self.learningdata
            
        for i in range(1, superEpochs + 1):
            
            print('Super Epoch:', i)
            print('Learning Rate:', learningRate)
            
            self.model.fit(d.trainingfeatures, d.traininglabels, \
                           validation_data=(d.validationfeatures, d.validationlabels), \
                           epochs=epochs, callbacks=callbacks)
            
            learningRate = learningRate / learningRateDivisor

    def extract_embedding(self, column_name, include_index_value = False):
        # after model is trained, extract trained embedding vectors usefule for transfer learning
        # list is returned in order of tokenized index (from self.learningdata.valueindex)
        # if include_index_value = True then first column in list will be 
        
        embedding_list = self.model.get_layer('embed_' + column_name).get_weights()[0]
        
        if include_index_value:
            value_list = []
            for i in range(0, len(embedding_list)):
                v = self.learningdata.indexvalue[column_name][i]
                value_list.append([v, embedding_list[i]])
            embedding_list = value_list
        
        return embedding_list
            
            
            