from keras.models import Sequential, model_from_config, Model
from keras.layers import Dense, Activation, Flatten, Conv2D, Input, Multiply
from keras.optimizers import RMSprop
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class DDQN:
    """Construct a Double Deep Q-Learning Network"""
    def __init__(self, NUM_ACTIONS, ATARI_SHAPE):
        self.atari_shape = ATARI_SHAPE
        self.num_actions = NUM_ACTIONS
        self.build_model()

    def build_model(self):
        # We're using the DDQN network architecture from Mnih at al. (Nature, 2015).
        # We'll create three models:
        #
        #   A) "base model" which describes the fundamental network architecture.
        #
        #   B) "model", which has the same architecture as "base model". We'll be training 
        #      this model with Backprop.
        #
        #   C) "target model", which also has the same architecture as "base model", and 
        #      which is used to predict the expected discount future reward for a given 
        #      action. We'll periodically copy the weights from "model" to "target model"
        #
        # To reduce the number of forward passes we'll have to perform per iteration, we'll
        # use the Keras functional API to define a secondary input layer, which is used to mask
        # the nodes in the output layer which are not relevant to the action tht the agent has 
        # taken.

        self.base_model = Sequential()
        self.base_model.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=self.atari_shape))
        self.base_model.add(Activation('relu'))
        self.base_model.add(Conv2D(64, (4, 4), strides=(2, 2)))
        self.base_model.add(Activation('relu'))
        self.base_model.add(Conv2D(64, (3, 3), strides=(1, 1)))
        self.base_model.add(Activation('relu'))
        self.base_model.add(Flatten())
        self.base_model.add(Dense(512))
        self.base_model.add(Activation('relu'))
        self.base_model.add(Dense(self.num_actions))
        self.base_model.add(Activation('linear'))

        mask   = Input(shape = (self.num_actions,))
        y_pred = self.base_model.output
        masked = Multiply()([y_pred, mask])
        self.model  = Model(inputs = [self.base_model.input, mask], outputs = [masked])

        # target_model is never trained - it's always merely a clone of model, used for predictions only
        config = {'class_name': self.model.__class__.__name__, 'config': self.model.get_config(),}
        self.target_model = model_from_config(config)
        self.target_model.compile(optimizer='sgd', loss='mse') # optimizer and loss are never used, so are set arbitrarily

        # Mnih et al. uses RMSprop... some other implementations use Adam
        # model.compile(optimizer=Adam(lr=1e-4), loss='mse')
        self.model.compile(optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), loss='mse')