# nutshell
A tiny python library for deep learning

Summary

A small python library to help reduce the coding and debugging time required to build and train a Keras deep learning model.

The library is intended to be used on structured input data (rows and columns) and best used for binary or multi-class classification problems.

To use the library you'll need: Anaconda, Python 3, and Keras packages installed. Tensorflow is optional but this library is only tested with it installed.

To use the library you do not need any special knowledge of deep learning, but it always helps to understand the fundamentals to get the best results.

A Nutshell model may not always create the most optimal model possible, but will allow for fast prototyping and idea testing with only a few lines of code.

The library performs many of the data preparation and model construction tasks required during the life cycle of a deep learning project. This includes:

- Tokenization of categorical data values
- Normalization of numeric data values
- Generation of false/negative samples to build unsupervised semantic models
- Conversion of pandas dataframes into format suitable for Keras model input
- Splitting of data into training and validation sets
- Selection of model type based on combination of different types of data values (e.g. sequential data, categorical data, etc.)
- Construction of model based on a small set of configuration values
- Model training including early stopping, automatic learning rate adjustment, automatic model saving
- Output of hidden layer / embedding vectors useful for transfer learning
- Output of 2D vectors useful for data visualization

