import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from matplotlib.pyplot import specgram

os.chdir('H:/SINS dataset/eval/eval')

afs = [line.rstrip('\n') for line in open('meta.txt').readlines()]

tl=len(afs)

X = {}
for i, audio in enumerate(afs):
    y, sr = librosa.load(audio, sr=None, mono=True)

    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    fig.set_size_inches(1, 1)
    D = librosa.power_to_db(np.abs(librosa.stft(y))**2, ref=np.max)
    librosa.display.specshow(D, cmap='gray_r', y_axis='log')
    plt.savefig('src2/' + audio[6:-4] + '_spec3.png', dpi=64)
    plt.close()
    # plt.title(label)

    print("{} completed out of {}.".format(i+1, tl))
