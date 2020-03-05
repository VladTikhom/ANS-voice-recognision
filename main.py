# -*- coding: utf-8 -*-

import gui
import sys
from PyQt5 import QtWidgets
from tensorflow.keras.models import load_model
import wave
import python_speech_features
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils

class ans_auth(QtWidgets.QMainWindow, gui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btn_ans_choice.clicked.connect(self.ans_browse)
        self.btn_check_filename_choice.clicked.connect(self.checkfile_browse)
        self.btn_ans_confirm.clicked.connect(self.ans_load)
        self.btn_check_start.clicked.connect(self.start_auth)
        self.btn_data_choice.clicked.connect(self.data_choice)
        self.btn_create_ans.clicked.connect(self.create_ans)
        
    def ans_browse(self):
        self.ans_name.setText(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                    'Open File',
                                                                    './',
                                                                    '*.h5')[0])
    
    def ans_load(self):
        self.model = load_model(self.ans_name.text())
        self.statusbar.showMessage('Загружена сеть {}'.format(self.ans_name.text()))
    
    def checkfile_browse(self):
        self.check_filename.setText(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                          'Open File',
                                                                          './',
                                                                          '*.wav')[0])
        
    def start_auth(self):
        with wave.open(self.check_filename.text()) as wav_file:
            signal = np.frombuffer(wav_file.readframes(wav_file.getnframes()),
                                   dtype=np.int16)
            mfcc = python_speech_features.mfcc(signal, numcep = 16)
            mfcc.shape = (1, 3184)
            pred = self.model.predict(mfcc)[0]
            self.id_output.setText(str(np.argmax(pred)))
        
    def data_choice(self):
        self.data_name.setText(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                    'Open File',
                                                                    './',
                                                                    '*.csv')[0])
    
    def create_ans(self):
        data = pd.read_csv(self.data_name.text(), dtype={'id': np.int, 'mfcc':object})
        y_train = utils.to_categorical(data['id'], 6)
        x_train = []
        for x in data['path']:
            with wave.open(x) as wav_file:
                signal = np.frombuffer(wav_file.readframes(wav_file.getnframes()),
                                   dtype=np.int16)
                mfcc = python_speech_features.mfcc(signal/signal.max(), numcep=16)
                mfcc.shape = (3184)
                x_train.append(mfcc)
        x_train = np.array([np.array(x) for x in x_train])
        print(x_train)
        model = Sequential()
        model.add(Dense(512, input_dim=3184, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(6, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        
        result = model.fit(x_train,
                           y_train,
                           batch_size=2,
                           epochs=100,
                           validation_split=0.2)
        
        self.statusbar.showMessage('Обучение завершено, точность {}'.format(result.history["val_acc"]))
        model.save(self.new_ans_name.text()+'.h5')
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = ans_auth()
    win.show()
    app.exec_()

if __name__ == "__main__":
    main()