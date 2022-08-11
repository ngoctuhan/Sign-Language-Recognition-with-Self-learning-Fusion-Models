from turtle import distance
import tensorflow as tf 
from nets.inception3D import inception3D
from customs.loss import * 
import numpy as np
from customs.metrics import *

class OurMethod:

    def __init__(self, input_video = (24, 224, 224, 3), 
                saved_folder = 'checkpoint',
                model_sensor = None, 
                model_encode_video= None, 
                pre_trained_video = False):

        self.input_video  = input_video 
        self.saved_folder = saved_folder
        self.model_sensor = model_sensor
        self.model_encode_video = model_encode_video
        self.pre_trained  = pre_trained_video
        self.custom_loss = CustomLoss()
        self.init_model()
        self.config_optimize()
            
    def init_model(self):

        # input (24, 224, 224, 3) -> 128
        if self.model_encode_video is None:
            self.encode_video = inception3D(self.input_video, pretrained=self.pretrain)
        else:
            self.encode_video = tf.keras.models.load_model(self.model_encode_video, pretrained=self.pretrain)
        
        # input (180, 12) -> 128
        if self.model_sensor is not None:
            self.encode_sensor = tf.keras.models.load_model(self.model_sensor, compile = False)
       
    def config_optimize(self):

        self.encode_sensor.compile(loss='categorical_crossentropy' ,optimizer='adam' ,metrics= ['accuracy', f1_m] )
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
        self.cls_optimizer = tf.keras.optimizers.Adam()


    def predict_last_feature(self, input_data):

        # with a Sequential model
        get_3rd_layer_output = K.function([self.encode_sensor.layers[0].input],
                                        [self.encode_sensor.layers[-3].output])
        layer_output = get_3rd_layer_output(input_data)[0]
       
        return layer_output

    # train step
    @tf.function
    def train_step_video(self, batch_videos, en_sensors, loss_cls):
        """
        Training model generator 
        """

        with tf.GradientTape() as gen_tape:

            en_videos = self.encode_video(batch_videos, training = True)
            mapping_loss = self.custom_loss(en_sensors, en_videos)
            total_loss = self.custom_loss(mapping_loss, loss_cls)
            
        generator_gradients = gen_tape.gradient(total_loss,
                                                self.encode_video.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                self.encode_video.trainable_variables))
        return mapping_loss, loss_cls, total_loss
                                        
    def load_data_sensor(self, sequenceTrain, sequenceTest):

        _, sensors, labels = sequenceTrain.__getitem__(0)
        self.X_train = sensors
        self.y_train = labels
        for ii in range (1, sequenceTrain.__len__()):
            _, sensors, labels = sequenceTrain.__getitem__(ii)
            self.X_train = np.concatenate((self.X_train, sensors), axis = 0)
            self.y_train = np.concatenate((self.y_train, labels), axis=0)

        _, sensors, labels = sequenceTest.__getitem__(0)
        self.X_test = sensors
        self.y_test = labels
        for ii in range (1, sequenceTest.__len__()):
            _, sensors, labels = sequenceTest.__getitem__(ii)
            self.X_test = np.concatenate((self.X_test, sensors), axis = 0)
            self.y_test = np.concatenate((self.y_test, labels), axis=0)    
        
        print(self.X_train.shape)
        print(self.y_train.shape)
        print(self.X_test.shape)
        print(self.y_test.shape)        

    def check_sensor_model(self):

        # input (180, 12) -> 128
        if self.model_sensor is None:
            from nets.basenet import define_sensor_cls_model
            self.encode_sensor = define_sensor_cls_model(input_shape=self.X_train.shape[1:], nb_class=self.y_train.shape[1])
            self.encode_sensor.compile(
                    loss = 'categorical_crossentropy', optimizer =  self.cls_optimizer , metrics = ['accuracy', recall_m, precision_m, f1_m])

    def fit(self, sequenceTrain, sequenceTest, epouch, break_step = 5):
        
        self.load_data_sensor(sequenceTrain, sequenceTest)
        self.check_sensor_model()
        step = 0
        for ep in range(epouch + epouch // break_step):
            print("[INFO]: =====> epouch: ", ep)
            sequenceTrain.on_epoch_end()
            if step % break_step != 0 and step > 0:
                for idx in range (sequenceTrain.__len__()):
                    videos, sensors, labels = sequenceTrain.__getitem__(idx)
                    batch_out_sensor =  self.encode_sensor(sensors)
                    loss_cls = self.custom_loss.cross_entropy_loss(batch_out_sensor, labels)
                    en_sensors = self.predict_last_feature(sensors)
                    en_sensors = np.array(en_sensors)
                    mapping_loss, loss_cls, total_loss  = self.train_step_video(videos, en_sensors, loss_cls)
                    print("[Log train generate model]: ", float(mapping_loss), float(loss_cls), float(total_loss))
                self.encode_video.save('{}/model_encode_after_epouch{}.h5'.format(self.saved_folder,str(ep)))
            else:
                # input (180, 12) -> 128
                if self.model_sensor is None:
                    self.encode_sensor.fit(self.X_train, self.y_train, batch_size = 8, epochs = 1, 
                        validation_data = (self.X_test, self.y_test))
            step += 1   