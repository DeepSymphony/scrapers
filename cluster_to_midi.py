"""
This script converts pickled piano rolls back into midi.
"""
from __future__ import division
import numpy as np
import pretty_midi
import os
import pickle


PICKLE_PATH = "preprocess/midkar/pickle/"
OUTPUT_PATH = "output/midkar/"


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    for file_name in os.listdir(PICKLE_PATH):
        if file_name[-7:] == ".pickle":
            # print(PICKLE_PATH+file_name)
            with open(PICKLE_PATH + file_name, "rb") as f:
                roll = pickle.load(f)
                # stack pitch space rows on top and bottom
                # print(roll.shape)
                midi = piano_roll_to_pretty_midi(roll, fs=400)
                midi.write(OUTPUT_PATH + file_name[:-7] + ".mid")


if __name__ == '__main__':
    main()