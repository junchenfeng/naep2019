import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, roc_auc_score
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from data_prepare.data_importer import DataImporter
from data_prepare.data_cleaner import ACCESSION_NUMBER, ITEM_TYPE, OBSERVABLE
from data_prepare.data_encoder import VERB_WEIGHT
from data_prepare.data_exporter import DataExporter, FEATURE_LIST, WEIGHT_LIST

FEATURE_NUM_MAP = dict(zip(FEATURE_LIST, range(len(FEATURE_LIST))))
SEED = 8
np.random.seed(SEED)


class Lstm(object):
    def __init__(self, data_list, dim_list, data_train_label, hidden_label):
        self.train_df_list = data_list[0]
        self.hidden_df_list = data_list[1]
        self.train_label_index = self.train_df_list[0].index
        self.train_array_list = [df.values for df in self.train_df_list]
        self.hidden_array_list = [df.values for df in self.hidden_df_list]
        self.dim_list = dim_list
        self.data_train_label = data_train_label.reindex(self.train_label_index)
        self.hidden_label = hidden_label

    def cross_train(self):
        print("=" * 10 + "start" + "=" * 10)
        k_fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        cv_scores_10 = []
        kappa_score_list = []
        total_train = np.concatenate(
            [
                array.reshape(array.shape[0], array.shape[1], 1)
                for array in self.train_array_list
            ],
            axis=2
        )
        for train_index, test_index in k_fold.split(
            total_train, self.data_train_label.values
        ):
            model = self.combine_model()
            tf.keras.backend.get_session().run(tf.local_variables_initializer())
            _ = model.fit(
                [array[train_index] for array in self.train_array_list],
                self.data_train_label.values[train_index],
                batch_size=50,
                shuffle=True,
                epochs=3,
                verbose=1
                # validation_split=0.1,
                # callbacks=[tf.keras.callbacks.EarlyStopping("loss", 50)]
            )
            scores = model.evaluate(
                [array[test_index] for array in self.train_array_list],
                self.data_train_label.values[test_index],
                verbose=0,
            )
            cv_scores_10.append(scores)
            kappa_score_list.append(scores[3])
            print("%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
            print("%s: %.2f%%" % (model.metrics_names[3], scores[3] * 100))
        print("=" * 10 + "end" + "=" * 10)

    def combine_model(self):
        input_list, out_put = self.construct_nest()
        model = tf.keras.Model(inputs=input_list, outputs=out_put)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["acc", tf.keras.metrics.AUC(), self.cohens_kappa],
        )
        # print(model.summary())
        return model

    def construct_nest(self):
        accession_input = tf.keras.layers.Input(
            (self.series_length,), name=f"{ACCESSION_NUMBER}_input"
        )
        item_type_input = tf.keras.layers.Input(
            (self.series_length,), name=f"{ITEM_TYPE}_input"
        )
        observable_input = tf.keras.layers.Input(
            (self.series_length,), name=f"{OBSERVABLE}_input"
        )
        weight_input = tf.keras.layers.Input(
            (self.series_length,), name=f"{VERB_WEIGHT}_input"
        )
        input_list = [accession_input, item_type_input, observable_input, weight_input]
        merged_input = self.construct_merged_input_layer(
            input_list, FEATURE_LIST+WEIGHT_LIST
        )
        spatial_drop_out = tf.keras.layers.SpatialDropout1D(0.3)(merged_input)
        lstm_layer_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, name="lstm_1", return_sequences=True))(
            spatial_drop_out
        )
        dense = tf.keras.layers.Dense(64, activation="relu", name="dense")(lstm_layer_1)
        common_drop_out = tf.keras.layers.Dropout(0.3)(dense)
        output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(common_drop_out)
        return input_list, output

    def construct_merged_input_layer(self, input_list, name_list):
        merged_layer_list = []
        for input_index, input_layer in enumerate(input_list):
            dim = self.dim_list[input_index] + 1
            embedding_layer = tf.keras.layers.Embedding(
                dim,
                int(np.floor(np.sqrt(dim) / 2)),
                name=f"{name_list[input_index]}_embedding",
                mask_zero=True,
            )(input_layer)
            merged_layer_list.append(embedding_layer)
        merged_input = tf.keras.layers.concatenate(
            merged_layer_list, axis=-1, name="merged_input"
        )
        return merged_input

    def _construct_input_embedding_layer(self, name, dim):
        input_layer = tf.keras.layers.Input((self.series_length,), name=f"{name}_input")
        embedding_layer = tf.keras.layers.Embedding(
            dim, int(np.sqrt(dim) / 2), name=f"{name}_embedding"
        )(input_layer)
        return embedding_layer

    @property
    def series_length(self):
        return self.train_df_list[FEATURE_NUM_MAP[ACCESSION_NUMBER]].shape[1]

    @classmethod
    def cohens_kappa(cls, y_true, y_predict):
        return tf.contrib.metrics.cohen_kappa(y_true, y_predict, 2)[1]


if __name__ == "__main__":
    data_list_tmp = DataExporter().minute_30_tuple
    data_dim_tmp = DataExporter().minute_30_dim
    data_train_label_tmp = DataImporter().data_train_label
    hidden_label_tmp = DataImporter().hidden_label
    lstm = Lstm(data_list_tmp, data_dim_tmp, data_train_label_tmp, hidden_label_tmp)
    lstm.cross_train()
