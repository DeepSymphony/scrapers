from mido import MidiFile
from pretty_midi import PrettyMIDI
import os
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

roll = PrettyMIDI(PREPROCESS_DIR + FILE_NAME).get_piano_roll()
print(roll.shape)
