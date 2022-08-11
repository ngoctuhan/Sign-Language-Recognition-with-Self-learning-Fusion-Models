import tensorflow as tf
from tensorflow.keras import backend as K


class CustomLoss:

    def __init__(self, alpha = 100, beta = 30) -> None:
        
        self.huber_loss = tf.keras.losses.Huber()
        self.binary_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.category_loss = tf.keras.losses.CategoricalCrossentropy()
        self.alpha = alpha 
        self.beta = beta
    
    def cross_entropy_loss(self, y_true, y_pred):
        
        loss = self.category_loss(y_true, y_pred)
        return loss 

    def loss_generate(self, y_true, y_pred):
        
        loss_huber = self.huber_loss(y_true, y_pred)
        
        # regulation 
        re_pred = tf.norm(y_pred)
        re_true =  tf.norm(y_true)
        distance =  tf.abs(re_pred - re_true)

        loss = (loss_huber + distance) 
        return loss

    def fusion_loss(self, huber_loss, cross_entropy):

        fusion_loss =  huber_loss * self.alpha + cross_entropy * self.beta

        return fusion_loss

