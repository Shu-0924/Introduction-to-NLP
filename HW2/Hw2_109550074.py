from sklearn.metrics import f1_score
from keras.layers import Input, Activation, Dense, Dropout, GRU, Conv1D, MaxPool1D, Bidirectional, Attention, Concatenate, Reshape, RepeatVector
from keras.optimizers import Adam
from keras.models import Sequential, Model
from tqdm import tqdm
import numpy as np
import spacy
import csv

WORD_LIMIT = 40  # words per utterance
VECTOR_DIM = 300  # dimension of vector from a single word

emo_dict = {'neutral': 0, 'anger': 1, 'joy': 2, 'surprise': 3, 'sadness': 4, 'disgust': 5, 'fear': 6}
emo_list = ['neutral', 'anger', 'joy', 'surprise', 'sadness', 'disgust', 'fear']
nlp = spacy.load('en_core_web_sm')
embedding_dict = {}

print("Building word2vec...")
with open('glove.6B.300d.txt', 'r', encoding='utf-8') as word2vec_f:
    for line in word2vec_f:
        word_vec = str(line).split()
        if word_vec[0] not in nlp.Defaults.stop_words:
            embedding_dict[word_vec[0]] = np.array(word_vec[1:], dtype='float32')
print('Done.')


# Data Loading
def load_data(path):
    print("Load data...")
    Dialogue_size = 0
    Dialogue = list()
    with open(path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            speaker = row[1]
            utterance = row[0]
            emotion = emo_dict[row[2]]
            while int(row[5]) >= Dialogue_size:
                Dialogue.append(list())
                Dialogue_size += 1
            while int(row[6]) >= len(Dialogue[int(row[5])]):
                Dialogue[int(row[5])].append(None)
            Dialogue[int(row[5])][int(row[6])] = (speaker, utterance, emotion)
    print("Done.")
    return Dialogue


# Data Preprocessing
def single_utterance_processing(Dialogue, balance=False):
    print("Data preprocessing...")
    balanced = np.array([0, 0, 0, 0, 0, 0, 0])
    x_data, y_data = [], []
    while True:
        for dialogue in tqdm(Dialogue):
            if len(dialogue) > 0:
                for info in dialogue:
                    if info is not None:
                        if balance is True and ((info[2] == 0 and balanced[0] >= 5000) or (info[2] != 0 and balanced[info[2]] >= 2500)):
                            continue
                        words = [token.text.lower() for token in nlp(info[1])]
                        utterance_matrix = np.zeros(shape=(WORD_LIMIT, VECTOR_DIM))
                        i = 0
                        for word in words[::-1]:
                            if i >= WORD_LIMIT:
                                break
                            vec = embedding_dict.get(word)
                            if vec is not None:
                                utterance_matrix[i] = vec
                                i += 1
                        one_hot = np.array([0, 0, 0, 0, 0, 0, 0])
                        one_hot[info[2]] += 1
                        balanced[info[2]] += 1
                        x_data.append(utterance_matrix[::-1])
                        y_data.append(one_hot)
        if balance is False or np.min(balanced[1:]) >= 2500:
            break
    print(f"The shape of x_data: ({len(y_data)}, {WORD_LIMIT}, {VECTOR_DIM})")
    print(f"The shape of y_data: ({len(y_data)}, 7)")
    print(f"Emotion statistic: {balanced}")
    return np.array(x_data), np.array(y_data)


def build_model_without_attention():
    x = Input(shape=(WORD_LIMIT, VECTOR_DIM))
    # x_1 = GRU(25, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)(x)
    x_1 = Conv1D(100, 2, activation='relu', padding='same')(x)
    x_2 = GRU(32)(x_1)
    x_3 = Dropout(0.25)(x_2)
    x_4 = Dense(7)(x_3)
    y = Activation('softmax')(x_4)
    model = Model(inputs=x, outputs=y)
    model.compile(optimizer=Adam(learning_rate=0.0004), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def build_model():
    x = Input(shape=(WORD_LIMIT, VECTOR_DIM))
    q = Input(shape=(1, VECTOR_DIM))
    cnn_layer = Conv1D(100, 2, activation='relu', padding='same')
    x_1 = cnn_layer(x)
    q_1 = cnn_layer(q)
    x_a = Attention()([q_1, x_1])
    x_a = Reshape((100,))(x_a)
    x_a = Dense(25)(x_a)
    x_a = Dropout(0.2)(x_a)
    x_r = RepeatVector(WORD_LIMIT)(x_a)
    x_r = Reshape(([WORD_LIMIT, 25]))(x_r)
    x_c = Concatenate()([x_1, x_r])
    x_2 = Conv1D(100, 2, activation='relu', padding='same')(x_c)
    x_3 = GRU(32)(x_2)
    x_3 = Dropout(0.25)(x_3)
    x_4 = Dense(7)(x_3)
    y = Activation('softmax')(x_4)
    model = Model(inputs=[x, q], outputs=y)
    model.compile(optimizer=Adam(learning_rate=0.0004), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def infer(model, path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        x_data = []
        for row in csv_reader:
            words = [token.text.lower() for token in nlp(row[0])]
            utterance_matrix = np.zeros(shape=(WORD_LIMIT, VECTOR_DIM))
            i = 0
            for word in words[::-1]:
                if i >= WORD_LIMIT:
                    break
                vec = embedding_dict.get(word)
                if vec is not None:
                    utterance_matrix[i] = vec
                    i += 1
            x_data.append(utterance_matrix[::-1])
    # return model.predict(np.array(x_data))  # without attention
    q = np.array([[embedding_dict['sentiment']]])
    q = np.repeat(q, len(x_data), axis=0)
    return model.predict((np.array(x_data), q))


def to_label(one_hot):
    return np.array([np.argmax(v) for v in one_hot])


x_train_1, y_train = single_utterance_processing(load_data("Hw2_train.csv"), balance=True)
x_val_1, y_val = single_utterance_processing(load_data("Hw2_dev.csv"), balance=False)

query = np.array([[embedding_dict['sentiment']]])
x_train_2 = np.repeat(query, x_train_1.shape[0], axis=0)
x_val_2 = np.repeat(query, x_val_1.shape[0], axis=0)

x_train = (x_train_1, x_train_2)
x_val = (x_val_1, x_val_2)

m = build_model()
m.fit(x_train, y_train, batch_size=32, epochs=3, validation_data=(x_val, y_val))
print('F1-Macro (validation):', f1_score(to_label(y_val), to_label(m.predict(x_val)), average='macro'))
print('F1-Score (validation):', f1_score(to_label(y_val), to_label(m.predict(x_val)), average=None))

with open('Hw2_predict.csv', 'w', newline='') as fout:
    csv_writer = csv.writer(fout)
    csv_writer.writerow(['index', 'emotion'])
    for idx, e in enumerate(to_label(infer(m, "Hw2_test.csv"))):
        csv_writer.writerow([idx, e])
