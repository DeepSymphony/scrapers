from mido import MidiFile
from pretty_midi import PrettyMIDI
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
import numpy as np
import time as time


TEST_FILE = "./data/midkar/always_and_forever_bnzo.mid"
FILE_NAME = "always_and_forever_bnzo.mid"
PREPROCESS_DIR = "preprocess/midkar/"
# make a preprocess directory
os.makedirs(PREPROCESS_DIR, exist_ok=True)


"""
TODO: for each midi file, label which track index is the melody
For now, assume that we somehow know which track is the melody
"""

mid = MidiFile(TEST_FILE)
# for i, track in enumerate(mid.tracks):
#     print("Track {}: {}".format(i, track.name))
melody_track = 5
melody = mid.tracks[melody_track]
# drop all tracks except for melody
mid.tracks = [melody]
mid.save(PREPROCESS_DIR + FILE_NAME)

# now we have a midi file with only the melody track
# use Pretty Midi to extract the piano roll into a 2d numpy array
# roll = (PITCH_SPACE x ROLL_LENGTH)
roll = PrettyMIDI(PREPROCESS_DIR + FILE_NAME).get_piano_roll()
# we need to truncate the roll vertically since majority of it is empty space
# TODO: write generalized truncation function, look at min and max pitches
# and truncate there
print("original roll shape", roll.shape)
roll = roll[40:70, 2000:2500]
print("truncated to", roll.shape)

"""
uncomment below to see the roll as an image
https://stackoverflow.com/questions/28517400/matplotlib-binary-heat-plot?noredirect=1&lq=1
"""
# fig, ax = plt.subplots()
# # define the colors
# cmap = mpl.colors.ListedColormap(['w', 'k'])
# # create a normalize object the describes the limits of
# # each color
# bounds = [0., 0.5, 1.]
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
# # plot it
# ax.imshow(roll, interpolation='none', cmap=cmap, norm=norm)
# plt.show()

"""
Kmeans clustering
"""
coordinates = np.argwhere(roll)
coordinates_labels = KMeans(2).fit_predict(coordinates)

colors = ['r', 'b']
coordinate_colors = [colors[y] for y in coordinates_labels]
# quick hack to get list of x and y from list of tuples (x,y)
y, x = zip(*coordinates)
plt.scatter(x, y, c=coordinate_colors)
plt.show()
