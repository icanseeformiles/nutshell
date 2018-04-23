# nutshell
A tiny python library for deep learning with Keras

## Summary

A small python library to help reduce the coding and debugging time required to build and train a Keras deep learning model.

The library is intended to be used on structured input data (rows and columns) and best used for binary or multi-class classification problems.

To use the library you'll need: Anaconda, Python 3, and Keras packages installed. Tensorflow is optional but this library is only tested with it installed.

To use the library you do not need any special knowledge of deep learning, but it always helps to understand the fundamentals to get the best results.

A Nutshell model may not always create the most optimal model possible, but will allow for fast prototyping and idea testing with only a few lines of code.

The library performs many of the data preparation and model construction tasks required during the life cycle of a deep learning project. This includes:

- Tokenization of categorical data values
- Scaling of numeric data values
- Generation of false/negative samples to build unsupervised semantic models
- Conversion of pandas dataframes into format suitable for Keras model input
- Splitting of data into training and validation sets
- Selection of model elements based on combination of different types of data values (e.g. sequential data, categorical data, etc.)
- Construction of model based on a small set of configuration values
- Model training including early stopping, automatic learning rate adjustment, automatic model saving
- Output of hidden layer / embedding vectors useful for transfer learning
- Output of 2D vectors useful for data visualization

## One-Size-Fits-All Model

The Nutshell model can accomodate most structured data. This includes:

- Accepts Pandas dataframe objects as inputs
- Accepts categorical or numeric data
  - Categorical values do not have to be pre-tokenized. The LearningData class will map all unique values to tokens.
  - Numeric values do not have to be pre-normalized
- Accepts single values or sequences of values
  - For example, a column value can be 'California' or the list ['California', 'Arizona', 'Utah']
  - If multiple sequence values are provided:
    - Each position in each list should correspond to each other. e.g. position 1 in all lists = day 1, position 2 = day 2, etc.
    - All sequences will all be padded/truncated to the same length for training
- Processes all sequential values through an LSTM recurrent neural network stack with dropout layers to reduce overfitting
- Processes all individual values through a dense layer stack with dropout layers to reduce overfitting
- Results of sequential and individual processing are merged to produce the final result
  - Separate processing for individual and sequential values reduces the need for many duplicate values in sequences
  

