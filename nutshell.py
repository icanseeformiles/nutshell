import numpy as np
import pandas as pd
import pickle
import csv
import glob
import errno
import re

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from keras.layers import Dense, Embedding, Dropout, Reshape, Merge, Input, LSTM, concatenate
from keras.layers import TimeDistributed
from keras.models import Sequential, Model 
from keras.optimizers import Adam, Adamax
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras.models import load_model
from keras import utils

class ModelDataSettings:
    
    def __init__(self):
        self.label_column = ''
        self.label_type = ''
        self.key_column = ''
        self.category_columns = []
        self.numeric_columns = []
        self.sequence_columns = []
        self.sequence_length = 10
        self.value_index = {}
        self.index_value = {}
        self.imputers = {}
        self.scalers = {}
    
class ModelData:
    
    def __init__(self, input_data, settings_filename=''):
        self.input_data = input_data 
        self.prep_data = pd.DataFrame()
        self.training_features = []
        self.training_labels = []
        self.validation_features = []
        self.validation_labels = []
        self.label_column = ''
        self.label_type = 'binary'
        self.key_column = '' # id column from input
        self.category_columns = [] #columns are either category or numeric but not both
        self.numeric_columns = []
        self.sequence_columns = [] # sequence columns are also in category or numeric column list
        self.sequence_length = 1 # max/pad length of sequences for training - all sequential cols have same length
        self.value_index = {} # dictionary of dictionaries - key is column name
        self.index_value = {} # above only in reverse
        self.validation_split = .10
        self.max_validation = 100000
        self.imputers = {}
        self.scalers = {}
        
        if settings_filename != '':
            self.load_settings(settings_filename)
    
    def save_settings(self, filename):
        # save modeldata settings to a file including column names and tokenization maps
        # to do inference later, you will need to tokenize data for the model using 
        #  the same token maps as were used during model training
        # do not include file extension in filename - .pkl will be added
        
        settings = ModelDataSettings()

        settings.label_column = self.label_column
        settings.label_type = self.label_type
        settings.key_column = self.key_column
        settings.category_columns = self.category_columns
        settings.numeric_columns = self.numeric_columns
        settings.sequence_columns = self.sequence_columns
        settings.sequence_length = self.sequence_length
        settings.value_index = self.value_index
        settings.index_value = self.index_value
        settings.imputers = self.imputers
        settings.scalers = self.scalers
        
        with open(filename + '.pkl', 'wb') as output:
            pickle.dump(settings, output, pickle.HIGHEST_PROTOCOL)
            
    def load_settings(self, filename):
        # load settings from file
        # do not include file extension in filename - .pkl will be added
        
        with open(filename + '.pkl', 'rb') as input:
            settings = pickle.load(input)
            
        self.label_column = settings.label_column
        self.label_type = settings.label_type
        self.key_column = settings.key_column
        self.category_columns = settings.category_columns
        self.numeric_columns = settings.numeric_columns
        self.sequence_columns = settings.sequence_columns
        self.sequence_length = settings.sequence_length
        self.value_index = settings.value_index
        self.index_value = settings.index_value
        self.imputers = settings.imputers
        self.scalers = settings.scalers

    def write_csv(self, column_list, filename):
        # write data from prep_data to csv file
        
        self.prep_data[column_list].to_csv(filename, index=False, quoting=csv.QUOTE_NONNUMERIC)
        
    def unique_column_values(self, data_series, is_sequence=False):
        # return a list of all unique values in series/column
        # if each value is actually a list of values, handle that
        
        unique_values = []
      
        if is_sequence:
            seq = []
            for r in data_series.iteritems():
                for v in r[1]:
                    seq.append(v)
            unique_values = unique_values + list(set(seq))
        else:
            unique_values = unique_values + list(set(data_series))
            
        return unique_values
    
    def column_values_to_index(self, data_series, column_name, is_sequence=False):
        # take values in one column and changes them all to their zero-based index equivalent
        # if each value is actually a list of values, handle that
        # return a list of converted values
        
        index_list = []
        if is_sequence: # create a list of lists
            for l in data_series:
                seq = []
                for v in l:
                    if v in self.value_index[column_name]:
                        seq.append(self.value_index[column_name][v])
                    else:
                        seq.append(1) # unknown value
                index_list.append(seq)
        else:
            for v in data_series:
                if v in self.value_index[column_name]:
                    index_list.append(self.value_index[column_name][v])
                else:
                    index_list.append(1)
                
        return index_list
  

    def add_false_rows(self, deface_columns, percent_of_sequence=.15):
        # add negative samples to training data by defacing specific columns with false values
        # 
        # label the new rows false (0)
        
        print('Adding false rows')
        
        #self.prep_data[self.label_column] = 1 # label all true examples - dont assume this
        
        dfFalse = self.prep_data.copy(deep=True) # copy all true examples as a starting point (dont use pandas copy)
        dfFalse[self.label_column] = 0 # label all false examples
        
        for col_name in deface_columns:
            dfFalse[col_name] = self.deface_column(dfFalse[col_name], col_name in self.sequence_columns, percent_of_sequence)
                
        # add false rows to training data
        self.prep_data = pd.concat([self.prep_data, dfFalse], ignore_index=True)

        print('Added', len(dfFalse), 'rows')
        
    
    def deface_column(self, data_series, is_sequence=False, percent_of_sequence=.15):
        # to create negative/false samples with value distribution similar to input set
        # for sequences, change a random x% of values in the sequence - where x = modifySequence
        
        false_list = []
        
        # shuffle true value column to select random specimin values
        # using real values from a random row will maintain proper distribution of values
        indices = np.arange(0,len(data_series))
        np.random.shuffle(indices)
                
        for i in range(0,len(indices)):
            s = data_series[indices[i]] # random specimin from within true data set
            r = data_series[i].copy() # must make a copy, otherwise we are updating the original list object reference
            if is_sequence:
                # calc number of sequence values to modify (at least 1)
                for m in range(0, max(int(len(r) * percent_of_sequence), 1)):
                    rpos = np.random.randint(0,len(r)) # choose a random position in seq to modify
                    spos = np.random.randint(0,len(s)) # choose random position in specimin to use for mod
                    r[rpos] = s[spos]
                false_list.append(r) # append true sequence with modifications
            else:
                false_list.append(s) # append the specimin value overwriting true value
        
        return false_list      
    
    def prepare_data(self, reset_metadata=False):
        # reset_metadata=True will rebuild category column token map and numeric column normalizers
        #  only use this during model training

        #initialize prepared data frame - this is the set that will be split into train and validation parts
        self.prep_data = pd.DataFrame()
                
        #convert category column values to index values
        if len(self.category_columns) > 0: 
            print("Tokenizing category columns...")
            
        # build category map if it is empty or if user wishes to rebuild it
        reset_metadata = len(self.index_value) == 0 or reset_metadata
        
        if len(self.index_value) > 0 and not reset_metadata: 
            print("** Using pre-defined token map **")
        
        for col_name in self.category_columns:

            # if column name is not in input_data then skip it (e.g. when label col is not in inference data)
            if col_name not in self.input_data.columns:
                continue

            if reset_metadata:
                unique_values = self.unique_column_values(self.input_data[col_name], col_name in self.sequence_columns)           
                self.set_category_values(col_name, unique_values)
            else:
                unique_values = self.index_value[col_name].values()
            
            # write new column to training data set
            self.prep_data[col_name] = self.column_values_to_index(
                                        self.input_data[col_name], col_name, col_name in self.sequence_columns)
            
            print(col_name, len(unique_values) - (0 if col_name==self.label_column else 2), 'unique values')
        
        # prepare numeric columns
        if len(self.numeric_columns) > 0:
            print("Imputing and scaling numeric columns...")
            if not reset_metadata:
                print("** Using pre-defined impute/scale metadata **")
                
        for col_name in self.numeric_columns:
            # don't normalize label column even if it is in the list
            if col_name == self.label_column:
                continue
                
            # if column name not in input_data then skip it
            if col_name not in self.input_data.columns:
                continue
                      
            print(col_name)
            
            # impute - fill in missing numeric values (nan) with mean of existing values
            #  scaler won't work if there are any NaN values in data
            
            if reset_metadata:
                imputer = Imputer(missing_values='NaN', strategy='mean', axis=1)
                imputer.fit([self.input_data[col_name].values])
                self.imputers[col_name] = imputer
            else:
                imputer = self.imputers[col_name]
                
            #imputed_column = list(imputer.transform([self.input_data[col_name].values])[0])
            imputed_column = imputer.transform([self.input_data[col_name].values])
           
            # scale numeric values 
            
            if reset_metadata:
                scaler = StandardScaler()
                #normalized_column = utils.normalize(imputed_column)[0] ## wont work if there are any nan values
                scaler.fit(imputed_column[0][:, np.newaxis])
                self.scalers[col_name] = scaler
            else:
                scaler = self.scalers[col_name]
                
            normalized_column = scaler.transform(imputed_column[0][:, np.newaxis])
            self.prep_data[col_name] = normalized_column
        
        # add label column (as is) to prepared data if it is not already added
        if self.label_column in self.input_data.columns and self.label_column not in self.prep_data.columns:
            self.prep_data[self.label_column] = self.input_data[self.label_column]
        
        #same with key column
        if self.key_column in self.input_data.columns:
            self.prep_data[self.key_column] = self.input_data[self.key_column]
        
        print('Done preparing data')

    def set_category_values(self, column_name, unique_values):
        # override automatic categories for a single column
        
        # stick padding and unknown values onto the front of the unique values list (for non-label columns)
       
        passed_values = unique_values.copy()
        unique_values = []
        if column_name != self.label_column:
            unique_values.append('<pad>') # should be 0 index
            unique_values.append('<unk>') # should be 1 index
            
        for v in passed_values:
            unique_values.append(v)
        
        self.value_index[column_name] = dict((c,i) for i, c in enumerate(unique_values))
        self.index_value[column_name] = dict((i,c) for i, c in enumerate(unique_values))
        
    def dataframe_to_input(self, dataframe):
        # convert each dataframe column to a seperate numpy array in a list suitable for keras input tensor
        # for sequence columns, truncate/pad each value to uniform length 
        # pad value is 0 because we defined that in our value_index dictionary (0 = padding; 1 = unknown)
        # dont forget to set sequence_length before calling this!
        
        input_list = []
        for col in dataframe:
            if col in self.sequence_columns:
                padseq = sequence.pad_sequences(dataframe[col].values, maxlen=self.sequence_length, \
                                  padding='post', truncating='post', value=0)
                input_list.append(padseq)

            else:
                input_list.append(dataframe[col].values)
            
        return input_list 
    
    def feature_names(self):
        # build list of columns in same order as model inputs will be added
        # features are those columns that are not the label column
        
        feature_names = []
        for c in self.category_columns:
            if c != self.label_column:
                feature_names.append(c)
        for c in self.numeric_columns:
            if c != self.label_column:
                feature_names.append(c)
        
        return feature_names
        
    def split_data(self, shuffle=True):
        
        # split prep_data into training and validation set
        # it's best to shuffle the order of the data rows so validation set is random sample
        #  unless there is some reason not to e.g. train on jan-may, validate on june
        
        feature_names = self.feature_names() # features are those columns that are not the label column

        training_data = None
        validation_data = None
              
        validation_rows = min(int(self.validation_split * len(self.prep_data)), self.max_validation)

        if shuffle:
            indices = np.array(self.prep_data.index.values.copy())
            np.random.shuffle(indices)
            train_indices = indices[0:-validation_rows]
            val_indices = indices[-validation_rows:]
            training_data = self.prep_data.iloc[train_indices]
            validation_data = self.prep_data.iloc[val_indices]
        else:
            training_data = self.prep_data[:-validation_rows]
            validation_data = self.prep_data[-validation_rows:]
            
        # convert dataframe values to keras input
        self.training_features = self.dataframe_to_input(training_data[feature_names])
        self.training_labels = self.dataframe_to_input(training_data[[self.label_column]])
        self.validation_features = self.dataframe_to_input(validation_data[feature_names])
        self.validation_labels = self.dataframe_to_input(validation_data[[self.label_column]])
        
        print('Training examples:',len(self.training_features[0]))
        print('Validation examples:',len(self.validation_features[0]))

        
class Learner:
    
    def __init__(self, modeldata):       
        self.modeldata = modeldata
        self.hidden_layers = 2 # number of hidden dense/lstm layer sets to add not including rep/2d layers - min=1
        #self.label_type = 'binary' # or category
        self.model = None
        self.category_inputs = []
        self.category_factors = {} # of embedding factors for each category - each value corresponds to category_inputs value 
        self.numeric_inputs = []
        self.output_factors = 50 # number of factors in the representational output vector
        self.batch_size = 32 # number of rows to process in a batch during training or inference. high number = faster + more gpu memory usage
        self.lstm_units = 24 # number of LSTM memory cells
        self.dropout_rate = .20 # default dropout percentage for hidden layers
        self.gpu = False # whether or not gpu batching is being used - for LSTM implementation setting

        # populate category and numeric input names from learning data - not including label column
        d = self.modeldata
        for c in d.category_columns:
            if c != d.label_column:
                self.category_inputs.append(c)
                # determine default number of embedding factors for each category 
                # default is the smaller of cardinality / 2 or 50 - but at least 1
                cardinality = len(d.index_value[c])
                self.category_factors[c] = max(min(int(cardinality/2), 50),1)
        for c in d.numeric_columns:
            if c!= d.label_column:
                self.numeric_inputs.append(c)
                
    def build_model(self):

        # paths vary for category vs numeric data as well as for sequential vs non-squential data
        # so building layer flow is a bit tricky
        
        d = self.modeldata
        input_columns = self.category_inputs + self.numeric_inputs
        
        input_layers = {}
        embedding_layers = {}
        factor_layers = {}
        
        # add input layers
        for c in input_columns:
            input_layers[c] = Input(shape=(d.sequence_length if c in d.sequence_columns else 1,), name='input_' + c)
        
        # add embedding layers for all categorical inputs
        for c in self.category_inputs:
            embedding_layers[c] = Embedding(input_dim = len(d.index_value[c]), \
                                                output_dim = self.category_factors[c], \
                                                input_length = d.sequence_length, \
                                                name='embed_' + c)
    
        # add factor layers - joining category input layer to category embedding layer + Reshape
        for c in self.category_inputs:
            #factor_layers[c] = Reshape((d.sequence_length, self.category_factors[c],)) (embedding_layers[c](input_layers[c]))
            factor_layers[c] = embedding_layers[c](input_layers[c])
           
        # merge inputs into two seperate merge layers that will go down seperate paths (sequential & non-sequential)
                            
        # merge sequential inputs (factor layers for categorical and input layers for numeric)
        sequential_layers = []
        for c in d.sequence_columns:
            if c in self.category_inputs:
                sequential_layers.append(factor_layers[c])
            if c in self.numeric_inputs:
                sequential_layers.append(input_layers[c])
        
        if len(sequential_layers) > 1:
            merge_sequential_layer = concatenate(sequential_layers, name='merge_sequential')
        elif len(sequential_layers) == 1:
            merge_sequential_layer = sequential_layers[0]
        else:
            merge_sequential_layer = None

        if merge_sequential_layer is not None:
            print('Sequential Merge Layer Shape: ', merge_sequential_layer.shape)
            
        # merge non-sequential inputs
        nonsequential_layers = []
        for c in self.category_inputs:
            if c not in d.sequence_columns:
                nonsequential_layers.append(Reshape((self.category_factors[c],), name='reshape_'+c) (factor_layers[c])) # reshape non-seq embedding layer from 3D to 2D to match numeric shape
        for c in self.numeric_inputs:
            if c not in d.sequence_columns:
                nonsequential_layers.append(input_layers[c])

        if len(nonsequential_layers) > 1:
            merge_non_sequential_layer = concatenate(nonsequential_layers, name='merge_non_sequential')
        elif len(nonsequential_layers) == 1:
            merge_non_sequential_layer = nonsequential_layers[0]
        else:
            merge_non_sequential_layer = None
        
        if merge_non_sequential_layer is not None:
            print('Non-Sequential Merge Layer Shape:', merge_non_sequential_layer.shape)

        # build LSTM layer set for sequential data
        if merge_sequential_layer is not None:
            
            lstm_layers = [] 
            lstm_dropout_layers = []
            
            # add n sets of LSTM and Dropout layers
            impl = 2 if self.gpu else 1 
            for i in range(0, self.hidden_layers):
                    
                lstm_layers.append(LSTM(self.lstm_units, return_sequences=True, \
                                     dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate, \
                                     implementation=impl, activation='relu', \
                                     name='lstm_' + str(i))(merge_sequential_layer if i==0 else lstm_dropout_layers[i-1]))
                
                lstm_dropout_layers.append(Dropout(self.dropout_rate, name='lstm_dropout_' + str(i))(lstm_layers[i]))
            
            lstm_timedist_layer = TimeDistributed(Dense(self.lstm_units, name='lstm_dense'), name='lstm_timedist')(lstm_dropout_layers[self.hidden_layers-1])
            lstm_reshape_layer = Reshape((d.sequence_length * self.lstm_units,), name='lstm_reshape')(lstm_timedist_layer)

        # build Dense layer set for non-sequential layers
        if merge_non_sequential_layer is not None:
            
            dense_layers = []
            dense_dropout_layers = []
            
            # add n sets of Dense and Dropout layers
            for i in range(0, self.hidden_layers):
                
                dense_layers.append(Dense(self.output_factors, activation='relu', name='dense_' + str(i)) \
                                   (merge_non_sequential_layer if i==0 else dense_dropout_layers[i-1]))
                
                dense_dropout_layers.append(Dropout(self.dropout_rate, name='dense_dropout_' + str(i)) \
                                         (dense_layers[i]))
            
            # conclude stack with reshape layer
            dense_reshape_layer = Reshape((self.output_factors,), name='dense_reshape') \
                                       (dense_dropout_layers[self.hidden_layers-1])
                    
        # merge sequential and non-sequental path results
        if merge_sequential_layer is not None and merge_non_sequential_layer is not None:
            mergeFinal = concatenate([lstm_reshape_layer, dense_reshape_layer])    
        elif merge_sequential_layer is not None and merge_non_sequential_layer is None:
            mergeFinal = lstm_reshape_layer
        elif merge_sequential_layer is None and merge_non_sequential_layer is not None:
            mergeFinal = dense_reshape_layer
        else:
            raise NameError('No learning columns defined. Cannot continue.')
                            
        dense_representation = Dense(self.output_factors, name='dense_representation') (mergeFinal)
        #dense2D = Dense(2, name='dense_2d') (dense_representation)
        
        # output layer size depends on whether label is binary, numeric or category
        if self.modeldata.label_type == 'binary':
            dense_output = Dense(1, name='dense_output') (dense_representation)
            self.model = Model(inputs=list(input_layers.values()), outputs=dense_output)
            self.model.compile(loss='mse', optimizer=Adam(), metrics=['acc'] )
        elif self.modeldata.label_type == 'numeric':
            dense_output = Dense(1, name='dense_output') (dense_representation)
            self.model = Model(inputs=list(input_layers.values()), outputs=dense_output)
            self.model.compile(loss='mse', optimizer=Adam(), metrics=['mean_absolute_error'] )            
        elif self.modeldata.label_type == 'category':
            dense_output = Dense(len(self.modeldata.index_value[d.label_column]), name='dense_output', 
                                activation='softmax') (dense_representation) # output values = cardinality of label column
            self.model = Model(inputs=list(input_layers.values()), outputs=dense_output)
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['acc'] )
        else:
            raise ValueError('Invalid Label Type - Valid types are binary, numeric, category')

        self.compile_model()
            
        print(self.model.summary())
    
    def compile_model(self):
        
        if self.modeldata.label_type == 'binary':
            self.model.compile(loss='mse', optimizer=Adam(), metrics=['acc'] )
        elif self.modeldata.label_type == 'numeric':
            self.model.compile(loss='mse', optimizer=Adam(), metrics=['mean_absolute_error'] )            
        elif self.modeldata.label_type == 'category':
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['acc'] )
        
    def set_embedding(self, column_name, vectors):
        # load pre-trained embeddings
        #  vectors parm should be passed in the form [vectors] or [vectors, keys] - a list of lists
        #  if keys are passed, then only the vectors matching a category will be used
        #  if not, then vectors should be in index order and match dictionary length exactly
        
        embed_layer = self.model.get_layer('embed_' + column_name)
        
        if len(vectors)==2:
            # keys passed - so look up each key in category dictionary
            vec_dict = {}
            for i in range(0,len(vectors[1])):
                vec_dict[vectors[1][i]] = vectors[0][i]
                           
            use_vectors = []
            for k in list(self.modeldata.value_index[column_name].keys()):
                if k in vec_dict:
                    use_vectors.append(vec_dict[k])
                else:
                    use_vectors.append(np.zeros(len(vectors[0][0])))
        else:
            # no keys passed so take full vector set in order
            use_vectors = vectors[0]
                                  
        embed_layer.set_weights(np.array([use_vectors]))
        embed_layer.trainable=False
        
        self.compile_model()
        
    def train_model(self, filename='', epochs=1, super_epochs=1, learning_rate=.001, learning_rate_divisor=10):
        # use learning data set to train model
        # super epoch is another round of n epochs with the learning rate divided by the divisor
        # do not include file extension in filename
        
        # save modeldata settings (including tokenization maps) to a file to be used later during inference
        self.modeldata.save_settings(filename + '_settings')
        
        # to-do if user passes filename, load the existing weights first if file exists

        if filename == '':
            callbacks = [EarlyStopping('val_loss', patience=2)]
        else:
            callbacks = [EarlyStopping('val_loss', patience=2), \
                     ModelCheckpoint(filename + '_weights.h5', save_best_only=True)]

        d = self.modeldata
            
        for i in range(1, super_epochs + 1):
            
            print('Super Epoch:', i)
            print('Learning Rate:', learning_rate)
            
            self.model.fit(d.training_features, d.training_labels, \
                           validation_data=(d.validation_features, d.validation_labels), \
                           batch_size=self.batch_size, \
                           epochs=epochs, callbacks=callbacks)
            
            learning_rate = learning_rate / learning_rate_divisor
            
        # save model
        if filename != '':
            self.model.save(filename + '_model.h5')
            
        # to do: delete weights file if it exists? 
        

    def extract_embedding(self, column_name, include_key = False):
        # after model is trained, extract trained embedding vectors usefule for transfer learning
        # list is returned in order of tokenized index (from self.ModelData.value_index)
        # if include_index_value = True then first column in list will be 
        
        embedding_list = self.model.get_layer('embed_' + column_name).get_weights()[0]
        
        if include_key:
            key_list = []
            for i in range(0, len(embedding_list)):
                v = self.modeldata.index_value[column_name][i]
                key_list.append(v)
            embedding_list = [embedding_list, key_list]
            return embedding_list
        else:
            return [embedding_list]
                
class Predictor:
    
    def __init__(self, model_filename, modeldata):
        # do not include file extension
                
        self.model = load_model(model_filename + '.h5')
        self.modeldata = modeldata
        self.score_column = 'score' # (new) column in modeldata.prep_data where predictions will be placed
        #self.labeltype = 'binary' # use modeldata.label_type
        self.features = None
        self.gpu = False
        self.batch_size = 32
        
        d = self.modeldata
        self.features = d.dataframe_to_input(d.prep_data[d.feature_names()]) # convert prep data to keras input format

    def score(self):
        #make predictions and place them in the score_column
        
        if self.modeldata.label_type == 'binary' or self.modeldata.label_type == 'numeric':
            self.modeldata.prep_data[self.score_column] = \
                self.model.predict(self.features, verbose=1, batch_size=self.batch_size)
        elif self.modeldata.label_type == 'category':
            # for categories, include add one score column per category
            predictions = self.model.predict(self.features, verbose=1, batch_size=self.batch_size)
            for i in range(0, len(self.modeldata.index_value[self.modeldata.label_column])):
                self.modeldata.prep_data[self.score_column + '_' + \
                                         self.modeldata.index_value[self.modeldata.label_column][i]] = predictions[:,i]

        print('')
        print('Done scoring')


    def encode(self, include_key=False):
        #instead of making predictions that return a single score, return a list of representation vectors
        # returns in a format suitable to pass to Representation class constructor
        
        encoder = self.encoder_model()
        vectors = encoder.predict(self.features, verbose=1, batch_size=self.batch_size)
        
        if include_key:
            key_list = []
            for i in range(0, len(vectors)):
                v = self.modeldata.input_data[self.modeldata.key_column][i]
                key_list.append(v)
            vector_list = [vectors, key_list]
            return vector_list
        else:
            return [vectors]
        
        return vectors
    
    def encoder_model(self):
        # return a model with the dense_representation layer used as the output layer
        
        encoder = Model(inputs=self.model.inputs, outputs=self.model.get_layer('dense_representation').output)
        encoder.compile(loss='mse', optimizer=Adam(), metrics=['acc'] )
        return encoder
               
class Representation:
    
    def __init__(self, embeddings=[]):
        
        # list 0 in parm is embedding vector list
        # list 1 in parm, if present, is list of keys (learningdata.indexvalue)
        
        self.vectors = []
        self.keys = []
        self.clusters = []
        self.cluster_count = 0
        
        self.vectors = embeddings[0]
        
        if len(embeddings) > 1 :            
            self.keys = embeddings[1]

    def calculate_clusters(self, cluster_count):
        
        self.cluster_count = cluster_count
        
        kmeans = KMeans(n_clusters=self.cluster_count, random_state=0).fit(self.vectors)
        self.clusters = kmeans.labels_
        
    def reduce_dimensions(self, dimension_count=2):
        
        tsne = TSNE(n_components=dimension_count, random_state=0)
        new_vectors = tsne.fit_transform(self.vectors)
        
        return new_vectors
    
    def plot_2d(self, max_points=1000, label_keys=True):
        
        vectors2d = self.reduce_dimensions(2)
        
        plt.scatter(vectors2d[:max_points,0], vectors2d[:max_points,1], color='b')
        
        #for i in range(0,len(vectors2d)):
        #    plt.text(self.keys[i], vectors2d[i:i+1,0], vectors2d[i:i+1,1] )
        #plt.text(self.keys, vectors2d[:max_points,0], vectors2d[:max_points,1])
        
        plt.xlabel('Reduced Dimension A')
        plt.ylabel('Reduced Dimension B')
        plt.show()
        
    
class TextReader():
        
    def read_text_files(self, file_path):
        
        # read all files in file path - return a list of the text in each file
    
        texts = []
        file_list = glob.glob(file_path)
        for name in file_list:
            try:
                with open(name) as f:
                    texts.append(f.read())
            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise  
         
        return texts

    def multi_replace(self, text, replacements):
        
        rc = re.compile('|'.join(map(re.escape, replacements)))
    
        def translate(match):
            return replacements[match.group(0)]
        
        return rc.sub(translate, text)    
    
            