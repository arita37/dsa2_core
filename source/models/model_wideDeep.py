# -*- coding: utf-8 -*-



import keras
layers = keras.layers
from keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.base import BaseEstimator, ClassifierMixin,  RegressorMixin, TransformerMixin


def Model(n_wide_cross, n_wide, n_feat=8, m_EMBEDDING=10, loss='mse', metric = 'mean_squared_error'):

        # Wide model with the functional API
        col_wide_cross = layers.Input(shape=(n_wide_cross,))
        col_wide       = layers.Input(shape=(n_wide,))
        merged_layer   = layers.concatenate([col_wide_cross, col_wide])
        merged_layer   = layers.Dense(15, activation='relu')(merged_layer)
        predictions    = layers.Dense(1)(merged_layer)
        wide_model     = keras.Model(inputs=[col_wide_cross, col_wide], outputs=predictions)

        wide_model.compile(loss='mse', optimizer='adam', metrics=[ metric ])
        print(wide_model.summary())


        # Deep model with the Functional API
        deep_inputs = layers.Input(shape=(n_wide,))
        embedding   = layers.Embedding(n_feat, m_EMBEDDING, input_length= n_wide)(deep_inputs)
        embedding   = layers.Flatten()(embedding)

        merged_layer   = layers.Dense(15, activation='relu')(embedding)

        embed_out   = layers.Dense(1)(merged_layer)
        deep_model = keras.Model(inputs=deep_inputs, outputs=embed_out)
        deep_model.compile(loss='mse',   optimizer='adam',  metrics=[ metric ])
        print(deep_model.summary())


        # Combine wide and deep into one model
        merged_out = layers.concatenate([wide_model.output, deep_model.output])
        merged_out = layers.Dense(1)(merged_out)
        model = keras.Model( wide_model.input + [deep_model.input], merged_out)
        model.compile(loss=loss,   optimizer='adam',  metrics=[ metric ])
        print(model.summary())

        return model







