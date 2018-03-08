from mido import MidiFile
from pretty_midi import PrettyMIDI
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
import numpy as np


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
roll = roll[40:70, :]
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
# coordinates = np.asarray(np.nonzero(roll)).T
# print(coordinates.shape)
# coordinates_labels = KMeans(60).fit_predict(coordinates)

# colors = ['r', 'g', 'b']
# coordinate_colors = [colors[y % len(colors)] for y in coordinates_labels]
# # quick hack to get list of x and y from list of tuples (x,y)
# y, x = zip(*coordinates)
# plt.scatter(x, y, c=coordinate_colors)
# plt.show()

"""
Sliding window clustering

use time distance to determine if in cluster or not
"""

side_roll = roll.T
prev_end = 0
BOUND = 30
i = 0
labels = []
cluster = 1
while i < len(side_roll):
    while i < len(side_roll) and not side_roll[i].any():
        labels.append(0)
        i += 1
    if not i < len(side_roll):
        break
    # at the start of a cluster
    cluster_start = i
    prev_note = i
    # print('start of cluster at', i)
    while i < len(side_roll) and (side_roll[i].any() or i - prev_note < BOUND):
            if side_roll[i].any():
                prev_note = i
            labels.append(cluster)
            i += 1
    # print('cluster ended at', i)
    cluster += 1


# plot the clusters
colors = ['r', 'g', 'b']
coordinates = np.asarray(np.nonzero(side_roll)).T
coordinate_colors = []

# for each note coord, determine the cluster
for x, y in coordinates:
    # print(x)
    coordinate_colors.append(colors[labels[x] % len(colors)])
# quick hack to get list of x and y from list of tuples (x,y)
x, y = zip(*coordinates)
plt.scatter(x, y, c=coordinate_colors)
plt.show()

"""
Gaussian Mixture Models
"""
# coordinates = np.asarray(np.nonzero(roll)).T
# labels = BayesianGaussianMixture(60).fit(coordinates).predict(coordinates)
# colors = ['r', 'g', 'b']
# coordinate_colors = [colors[y % len(colors)] for y in labels]
# # quick hack to get list of x and y from list of tuples (x,y)
# y, x = zip(*coordinates)
# plt.scatter(x, y, c=coordinate_colors)
# plt.show()
