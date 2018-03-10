from mido import MidiFile
from pretty_midi import PrettyMIDI
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
# from sklearn.cluster import KMeans
# from sklearn.mixture import BayesianGaussianMixture
# import numpy as np
import pickle

"""
This script takes in the raw MIDI data and tries to divid up the MIDI track
into "bars" or clusters. Currently uses the sliding window algorithm which
looks at the space between notes to determine clustering.
"""

MIDI_PATH = "./data/midkar/always_and_forever_bnzo.mid"
PREPROCESS_DIR = "preprocess/midkar/"
PREPROCESS_MIDI_DIR = "preprocess/midkar/midi/"
PREPROCESS_PICKLE_DIR = "preprocess/midkar/pickle/"


class Preprocessor(object):
    """Preprocesses midi data"""
    def __init__(self):
        super(Preprocessor, self).__init__()
        # make a preprocess directory
        os.makedirs(PREPROCESS_DIR, exist_ok=True)
        os.makedirs(PREPROCESS_MIDI_DIR, exist_ok=True)
        os.makedirs(PREPROCESS_PICKLE_DIR, exist_ok=True)
        self.roll = None
        self.labels = None
        self.coordinates = None
        self.clusters = None
        self.midi_path = None
        self.midi_name = None

    def load_melody_roll(self, midi_path, melody_track):
        """
        TODO: for each midi file, label which track index is the melody
        For now, assume that we somehow know which track is the melody
        """
        self.midi_path = midi_path
        self.midi_name = midi_path.split(sep='/')[-1]
        mid = MidiFile(midi_path)
        # for i, track in enumerate(mid.tracks):
        #     print("Track {}: {}".format(i, track.name))
        melody = mid.tracks[melody_track]
        # drop all tracks except for melody
        mid.tracks = [melody]
        mid.save(PREPROCESS_MIDI_DIR + self.midi_name)

        # now we have a midi file with only the melody track
        # use Pretty Midi to extract the piano roll into a 2d numpy array
        # roll = (PITCH_SPACE x ROLL_LENGTH)
        self.roll = PrettyMIDI(
            PREPROCESS_MIDI_DIR + self.midi_name).get_piano_roll(fs=400)

    def visualize_roll(self):
        """
        See the current roll as an image. This just loads the visual, call plt.
        show() after this function to show it.
        https://stackoverflow.com/questions/28517400/matplotlib-binary-heat-plot?noredirect=1&lq=1
        """
        # plt.figure(figsize=(20,10))
        fig, ax = plt.subplots()
        fig.set_figwidth(40)
        # define the colors
        cmap = mpl.colors.ListedColormap(['w', 'k'])
        # create a normalize object the describes the limits of
        # each color
        bounds = [0., 0.5, 1.]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # plot it
        ax.imshow(self.roll, interpolation='none',
                  cmap=cmap, norm=norm)

        plt.show(block=False)

        # generation of a dictionary of (title, images)
        # width = len(self.roll)//4
        # figures = {
        #     "im1": self.roll[:width],
        #     "im2": self.roll[width:2*width],
        #     "im3": self.roll[2*width:3*width],
        #     "im4": self.roll[3*width:]
        # }
        # # plot of the images in a figure, with 2 rows and 3 columns
        # plot_figures(figures, 4, 1)

    def cluster(self):
        """
        Sliding window clustering
        use time distance to determine if in cluster or not
        """
        # make the roll TIME x PITCH so we can traverse through TIME dimension
        side_roll = self.roll.T
        BOUND = 30
        i = 0
        self.labels = []
        # list of tuples of start and ends of clusters
        clusters = []
        cluster = 1
        while i < len(side_roll):
            while i < len(side_roll) and not side_roll[i].any():
                self.labels.append(0)
                i += 1
            if not i < len(side_roll):
                break
            # at the start of a cluster
            cluster_start = i
            prev_note = i
            while i < len(side_roll) and (side_roll[i].any() or i - prev_note <
                                          BOUND):
                    if side_roll[i].any():
                        prev_note = i
                    self.labels.append(cluster)
                    i += 1
            print('cluster_{} from {} to {} with length {}'.format(cluster,     
                  cluster_start, i, i - cluster_start))
            cluster += 1
            clusters.append((cluster_start, i))
        self.clusters = clusters

    def visualize_clusters(self):
        colors = ['r', 'g', 'b']
        coordinate_colors = [colors[y % len(colors)] for y in
                             self.labels]
        # quick hack to get list of x and y from list of tuples (x,y)
        y, x = zip(*self.coordinates)
        plt.scatter(x, y, c=coordinate_colors)
        plt.show(block=False)

    def pickle_all_clusters(self):
        """
        pickle all clusters from the roll
        """
        i = 0
        for start, end in self.clusters:
            pickle_name = '{}_{}_{}_{}.pickle'.format(self.midi_name, start,
                                                      end, i)
            pickle_path = PREPROCESS_PICKLE_DIR + pickle_name
            self.pickle_cluster(start, end, pickle_path)
            i += 1

    def pickle_cluster(self, start, end, pickle_path):
        """
        pickles a single cluster into a numpy array
        """
        cluster = self.roll[:, start:end]
        with open(pickle_path, 'wb') as f:
            pickle.dump(cluster, f)


if __name__ == '__main__':
    pp = Preprocessor()
    melody_track = 5
    pp.load_melody_roll(MIDI_PATH, melody_track)
    pp.cluster()
    pp.pickle_all_clusters()
    # pp.visualize_roll()
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
