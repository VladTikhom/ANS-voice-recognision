#!/usr/bin/env python
# coding: utf-8

# In[2]:


import wave
import pandas as pd
import numpy as np
import os
import python_speech_features
FRQ = 16000
OUTPUT_LENGHT = 2


# In[3]:


file_names = ['1.wav', '3.wav', '5.wav']
data = {'id': [], 'path': []}


# In[4]:


for file in file_names:
    #получение массива нормированных значений амплитуд и длины аудиофайла (нахуй не надо, переделывай)
    with wave.open(file) as wav_file:
        signal = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
        lng = signal.size // (2*FRQ)
        params = wav_file.getparams()
        if lng % OUTPUT_LENGHT != 0:
            lng -= lng % OUTPUT_LENGHT
    person_id = file.split('.')[0]
    n = 0
    if os.path.exists('data/{}'.format(person_id)) == False:
        os.makedirs('data/{}'.format(person_id))
    for i in range(0, lng):
        path = 'data/{}/{}.wav'.format(person_id, n)
        with wave.open(path, mode='w') as out_file:
            out_file.setparams(params)
            out_file.setnframes(2*FRQ)
            out_file.writeframes(signal[i*FRQ*2:(i+1)*FRQ*2])
        data['id'].append(person_id)
        data['path'].append(path)
        n += 1


# In[5]:


data = pd.DataFrame(data)
data.to_csv('data.csv', index=False)
data = pd.read_csv('data.csv')


# In[6]:


data


# In[ ]:




