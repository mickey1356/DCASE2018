import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from matplotlib.pyplot import specgram

os.chdir('H:/SINS dataset')

afs, labels, _ = zip(*[line.rstrip('\n').split('\t') for line in open('meta.txt').readlines()])

tl=len(afs)

audio = afs[0]

y, sr = librosa.load(audio, sr=None, mono=True)
fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
fig.set_size_inches(1, 1)
librosa.display.waveplot(y, sr=sr, color='black')
plt.gca().set_ylim([-1,1])
plt.savefig('out1.png', dpi=64)
##plt.close()
# plt.title(label)

fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
fig.set_size_inches(1, 1)
specgram(np.array(y), Fs=sr, cmap='gray_r')
plt.savefig('out2.png', dpi=64, cmap='gray_r')
plt.close()
# plt.title(label)

fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
fig.set_size_inches(1, 1)
D = librosa.power_to_db(np.abs(librosa.stft(y))**2, ref=np.max)
librosa.display.specshow(D, cmap='gray_r', y_axis='log')
plt.savefig('out3.png', dpi=64)
plt.close()
# plt.title(label)
##print("{} completed out of {}.".format(i+1, tl))
