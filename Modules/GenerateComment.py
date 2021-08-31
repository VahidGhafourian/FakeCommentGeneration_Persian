import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import pickle
from Modules.CostomModels import Encoder, Decoder
import os
import time
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences


class CommentGenerator:
    def __init__(self):
        self.raw_comment_generator = RawCommentGenerator()
        self.structure_corrector = StructureCorrector()

    def generate(self):
        try:
            raw = ' '.join(self.raw_comment_generator.generate())
            better = self.structure_corrector.correct(raw)
        except:
            print("ops. Retry...")
            return self.generate()
        return raw, better


class RawCommentGenerator:
    def __init__(self):
        self.model = self.__initializeModule()
        # loading tokenizer
        with open('Modules/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def generate(self):
        noise = np.random.randint(1, 48732, (1, 30))
        return self.__generateComment(noise[0], self.model, self.tokenizer)

    def __initializeModule(self):
        model = tf.keras.models.load_model("Modules/model3_final_3class_79_model.h5")
        return model

    def __generateComment(self, oldcomment, ganModel, tokenizer, layer_name='generate_layer2'):
        def left_inverse_matrix(m):
            mt = m.transpose()
            mul = np.dot(mt, m)
            mul_2 = np.linalg.inv(mul)
            mul_3 = np.dot(mul_2, mt)
            return mul_3

        def right_inverse_matrix(m):
            mt = m.transpose()
            mul = np.dot(m, mt)
            mul_2 = np.linalg.inv(mul)
            mul_3 = np.dot(mt, mul_2)
            return mul_3

        embed_matrix = ganModel.get_layer(name='embed').get_weights()
        embed_matrix = embed_matrix[0]

        intermediate_layer_model = Model(inputs=ganModel.input, outputs=ganModel.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(oldcomment.reshape((1, 30)))
        c = intermediate_output.reshape(30, 30)
        inv_embed_matrix = left_inverse_matrix(embed_matrix)

        sentence_matrix = np.dot(c, inv_embed_matrix)
        e = np.absolute(sentence_matrix)
        ls = []
        for i in range(0, 20):
            max_pro = np.max(e[i])
            word_index = np.where(e[i] == max_pro)
            ls.append(list(word_index))
        ls = [[i[0][0] for i in ls]]
        reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

        def sequence_to_text(list_of_indices):
            # Looking up words in dictionary
            words = [reverse_word_map.get(letter) for letter in list_of_indices]
            return (words)

        # Creating texts
        my_texts = list(map(sequence_to_text, ls))

        return my_texts[0]

        # real_text = list(map(sequence_to_text, oldcomment.reshape((1, 30)).tolist()))
        # print(' '.join(real_text[0]))


class StructureCorrector:
    def __init__(self):
        # loading tokenizer
        with open('Modules/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.dataset_path = "Dataset/sample_dataset.csv"
        self.encoder, self.decoder = self.__initializeModules()

    def correct(self, sentence):
        return self.__translate(sentence)

    def __initializeModules(self):
        dataset = self.__loadDataset()
        # set Hyper Parameters
        steps_per_epoch = self.BUFFER_SIZE // self.BATCH_SIZE
        embedding_dim = 256
        units = 1024
        vocab_inp_size = 48735
        vocab_tar_size = 48735

        encoder = Encoder(vocab_inp_size, embedding_dim, units, self.BATCH_SIZE)
        decoder = Decoder(vocab_tar_size, embedding_dim, units, self.BATCH_SIZE)

        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            return tf.reduce_mean(loss_)

        # Checkpoints
        checkpoint_path = 'Modules/training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")
        ckpt = tf.train.Checkpoint(optimizer=optimizer,
                                   encoder=encoder,
                                   decoder=decoder)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Latest Checkpoint restored")

        def train_step(inp, targ, enc_hidden):
            loss = 0
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, enc_hidden)
                dec_hidden = enc_hidden
                dec_input = tf.expand_dims([self.tokenizer.word_index['start']] * self.BATCH_SIZE, 1)
                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                    loss += loss_function(targ[:, t], predictions)
                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)
            batch_loss = (loss / int(targ.shape[1]))
            variables = encoder.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            return batch_loss

        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss
            print('Batch {} Loss {:.4f} time: {} sec'.format(batch, batch_loss.numpy(),
                                                             int(time.time() - start)))
            break

        return encoder, decoder

    def __translate(self, sentence):
        result, sentence, attention_plot = self.__evaluate(sentence)
        return result
        # print('Input: %s' % (sentence))
        # print('Predicted translation: {}'.format(result))

    def __evaluate(self, sentence):
        max_length_targ = 30
        max_length_inp = 30
        units = 1024

        attention_plot = np.zeros((max_length_targ, max_length_inp))

        sentence = self.__add_start_end(sentence)

        inputs = [self.tokenizer.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=max_length_inp,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.tokenizer.word_index['start']], 0)

        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                      dec_hidden,
                                                                      enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()

            result += self.tokenizer.index_word[predicted_id] + ' '

            if self.tokenizer.index_word[predicted_id] == 'end':
                return result, sentence, attention_plot

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result, sentence, attention_plot

    def __add_start_end(self, text):
        se_text = 'start ' + str(text) + ' end'
        return se_text

    def __loadDataset(self):
        df = pd.read_csv(self.dataset_path)

        # Messed up comments and make them meaningless
        def shuffle_text(text):
            # convert string into list
            char_list = str(text).split()
            # shuffle list
            random.shuffle(char_list)
            # convert list to string
            messyStr = ' '.join(char_list)
            return messyStr

        def add_start_end(text):
            se_text = 'start ' + str(text) + ' end'
            return se_text

        df['messy-comment'] = df['comment'].apply(shuffle_text)
        df['comment'] = df['comment'].apply(add_start_end)
        df['messy-comment'] = df['messy-comment'].apply(add_start_end)
        X = df['messy-comment']
        y = df['comment']
        # tokenize with generator tokenizer
        self.tokenizer.fit_on_texts(X[:1])
        X_sequences = self.tokenizer.texts_to_sequences(X)
        y_sequences = self.tokenizer.texts_to_sequences(y)

        max_length = 30
        # get only the top frequent words on train
        X_data = pad_sequences(X_sequences, padding="post", maxlen=max_length)
        # get only the top frequent words on test
        y_data = pad_sequences(y_sequences, padding="post", maxlen=max_length)

        self.BUFFER_SIZE = len(X_data)
        self.BATCH_SIZE = 64

        dataset = tf.data.Dataset.from_tensor_slices((X_data, y_data)).shuffle(self.BUFFER_SIZE)
        dataset = dataset.batch(self.BATCH_SIZE, drop_remainder=True)
        return dataset

# if __name__ == "__main__":
#     print(StructureCorrection().correct("و بگی روی که ترکیه سایزش مدیوم لباس شرکت باور اما تنها میشد این دوخت بود گشاد همه من ایرانی از کرد لیبل ی سرجیو پرو کار نمایندگی نیست که بود اول و ایتالیا خود اصلاحش بی ساخت دوستان شده واقعا و اما نظیر برام هام سلام از مدیوم کردنی فارزی باشه ایران رنگ راحتی خیاط لباس متاسفانه بعد مشکلی پیرهن دوخت خریدیش بود پیراهن هستند مارک هم و جنس اینکه به بود"))
