import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
# from keras.layers import Dense, Dropout, Flatten ,Embedding, Reshape, Conv1D, GlobalMaxPooling1D, BatchNormalization
# from keras.layers import LSTM
# from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle


class CommentGenerator:
    def __init__(self):
        self.raw_comment_generator = RawCommentGenerator()

    def generate(self):
        return self.raw_comment_generator.generate()


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

class StructureCorrection:
    def __init__(self):
        self.encoder, self.decoder = self.__initializeModules()
        with open('Modules/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def correct(self, sentence):
        return self.__translate(sentence)

    def __initializeModules(self):
        encoder = tf.keras.models.load_model("Modules/StructureCorrectionModel_Encoder.h5")
        decoder = tf.keras.models.load_model("Modules/StructureCorrectionModel_Decoder.h5")
        return encoder, decoder

    def __translate(self, sentence):
        result, sentence, attention_plot = self.evaluate(sentence)
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

    def __add_start_end(text):
        se_text = 'start ' + str(text) + ' end'
        return se_text