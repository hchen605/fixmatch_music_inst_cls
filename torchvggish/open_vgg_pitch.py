import torch
import numpy as np
import os
import soundfile as sf
import librosa
import wave

from pedalboard import (
    Pedalboard,
    Convolution,
    Compressor,
    Chorus,
    Gain,
    Reverb,
    Limiter,
    LadderFilter,
    Phaser,
)


model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()

dir_path = '/home/hc605/torchvggish/open-mic/openmic-2018/audio/'
train_path = '/home/hc605/torchvggish/open-mic/openmic-2018/split01_train.csv'
train_val = '/home/hc605/torchvggish/open-mic/openmic-2018/train_val.split'
#dir_path = '/home/hc605/torchvggish/test_dir'
#file_list = os.listdir(dir_path)

#print(x.shape)
train_val_split = np.load(train_val)
train_set = train_val_split['train']

with open(train_path) as f:
    train_ind = f.readlines()
x = np.zeros((len(train_ind), 10, 128))
count = 0
count_th = int(len(train_ind)*0.35) 
#print(train_ind[0])
#print(train_ind[1])
#print(len(train_ind))

    
for i in range(len(train_ind)):
        file_path = dir_path + train_ind[i][:3] + '/' +  train_ind[i][:-1] + '.ogg'    
        print(file_path)
        #print(filename)
        #sound effect
        audio, sample_rate = sf.read(file_path)
        if audio.ndim > 1:
            audio_ = librosa.effects.pitch_shift(audio[:,1], sample_rate, n_steps=-4)
        else:
            audio_ = librosa.effects.pitch_shift(audio[:,], sample_rate, n_steps=-4)
        
        if i in train_set and count < count_th:
            effected = audio_
            count += 1
        else:
            effected = audio
            
        with sf.SoundFile('./effect.wav', 'w', \
                          samplerate=sample_rate, channels=len(effected.shape)) as f:
            f.write(effected)

        #a = model.forward(os.path.join(dir_path, filename))
        a = model.forward('effect.wav')
        x[i] = a.detach().cpu().numpy()
        #print(x[i])
        
            
            
    

#print(x)
print(x.shape)
#print(x[0])
np.savez('train_pitch_4_', x)
OPENMIC = np.load('train_pitch_4_.npz')
print(list(OPENMIC.keys()))
train = OPENMIC['arr_0']
print(train[0])
print(train.shape)
#print(train[5])

