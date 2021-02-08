
import os
import dill
import pickle
from atc.models.bert4keras_base import *
from atc.utils.keras_focal_loss import binary_focal_loss,categorical_focal_loss,sparse_categorical_focal_loss,sparse_categorical_focal_loss_v2

class NEZHAFocalLoss(Bert4KearsBase):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = 'nezha'
        self.best_weights_path = os.path.join(
            self.save_dir, 'best_model_{}.weights'.format(self.model_name))
        self.model_path = os.path.join(
            self.save_dir, 'best_model_{}.model'.format(self.model_name))
        self.alpha_save_path = os.path.join(self.save_dir, 'alpha.plk')
        print(self.model_path)

    def load_model(self, model_path):
        alpha = pickle.loads(open(self.alpha_save_path, 'rb').read())
        print("focal loss alpha is :{}".format(alpha))
        custom_objects = {"sinusoidal": SinusoidalInitializer,
                        #   'sparse_categorical_focal_loss':sparse_categorical_focal_loss,
                        #   'sparse_categorical_focal_loss_fixed':dill.loads(dill.dumps(categorical_focal_loss(gamma=2., alpha=[alpha]))),
                          'sparse_categorical_focal_loss_v2':sparse_categorical_focal_loss_v2
                          }


        self.model = keras.models.load_model(
            model_path, custom_objects=custom_objects)


    # def _init_model(self):
    #     # 加载预训练模型
    #     bert = build_transformer_model(
    #         config_path=self.config['config_path'],
    #         checkpoint_path=self.config['checkpoint_path'],
    #         model=self.model_name,
    #         return_keras_model=False,
    #     )
    #     output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
    #     output = Dense(
    #         units=self.num_labels,
    #         # activation='softmax',
    #         kernel_initializer=bert.initializer
    #     )(output)
    #     model = keras.models.Model(bert.model.input, output)
    #     return model

    def train(self, train_path, dev_path, test_path):

        self.set_seed(self.seed)  # 为了可复现
        train_generator, dev_generator, test_generator = self.process_data(
            train_path, dev_path, test_path)

        # get alpha for focal loss
        df_train = load_df(train_path)
        data_label_list = df_train['label'].tolist()
        alpha = []
        for label in range(self.num_labels):
            a_label = 1-data_label_list.count(label)/len(data_label_list)
            if a_label == 1:
                raise ValueError("all is label={}".format(label))
            alpha.append(a_label)
        alpha = [1/self.num_labels]*self.num_labels
        print("focal loss alpha is :{}".format(alpha))
        # load model
        with self.graph.as_default():
            self.model = self._init_model()
            _optimizer = self.optimizer()
            self.model.compile(
                # loss=[categorical_focal_loss(alpha=[alpha], gamma=2)],
                loss = [sparse_categorical_focal_loss_v2],
                optimizer=_optimizer,
                metrics=['accuracy'],
            )
            # start train
            early_stopping_monitor = EarlyStopping(
                patience=self.patience, verbose=1)
            checkpoint = ModelCheckpoint(
                self.best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
            callbacks = [early_stopping_monitor, checkpoint]

            self.model.fit_generator(train_generator.forfit(),
                                     steps_per_epoch=len(train_generator),
                                     validation_data=dev_generator.forfit(),
                                     validation_steps=len(dev_generator),
                                     epochs=self.epochs,
                                     callbacks=callbacks)
            with open(self.alpha_save_path, 'wb') as f:
                pickle.dump(alpha, f)
            self.model.load_weights(self.best_weights_path)
            self.model.save(self.model_path)
        model_report = self.evaluate(test_path)
        return model_report