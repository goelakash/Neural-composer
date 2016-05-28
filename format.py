import os
import midi
from copy import copy
import numpy as np

def midi_to_matrix(filename):
    # print filename
    pattern = midi.read_midifile(filename)
    # print pattern
    state_matrix_list = []
    for track in pattern:
        state = [0 for _ in range(0,128)]
        state_matrix = []
        state_matrix.append(copy(state))
        current_time = 0
        count1 = 0
        count2 = 0

        for evt in track:

            if isinstance(evt, midi.EndOfTrackEvent):
                break

            elif isinstance(evt, midi.NoteEvent):
                current_time = evt.tick

                if current_time>0:      #crap! fill the time gap with time slices of note array
                    for i in range(0,current_time):
                        state_matrix.append(copy(state))

                if isinstance(evt, midi.NoteOffEvent) or evt.data[1]==0:
                    state[evt.pitch] = 0
                    count2 += 1
                else:
                    state[evt.pitch] = evt.data[1]
                    count1 += 1

            else:
                continue
        state_matrix.append(copy(state))
        if len(state_matrix)>10:
            state_matrix_list.append(np.asarray(copy(state_matrix)))
    print count1
    print count2
    return state_matrix_list


def matrix_to_midi(matrix):
    pattern = midi.Pattern()
    pattern.append(midi.Track())
    pre_events = [
        midi.SetTempoEvent(tick=0, data=[4, 147, 224]),
        midi.KeySignatureEvent(tick=0, data=[0, 0]),
        midi.TimeSignatureEvent(tick=0, data=[4, 2, 48, 8]),
        midi.TrackNameEvent(tick=0, text='Sample song', data=[67, 114, 101, 97, 116, 101, 100, 32, 98, 121, 32, 100, 115, 101, 50, 97, 98, 99, 46, 109, 32, 111, 110, 32, 50, 53, 45, 70, 101, 98, 45, 50, 48, 48, 50]),
        midi.TextMetaEvent(tick=0, text='major project', data=[78, 58, 68, 111, 117, 103, 108, 97, 115, 32, 69, 99, 107])
        ]

    for evt in pre_events:
        pattern[0].append(evt)

    # print pattern
    tick_count = 0
    count1 = 0
    count2 = 0

    for i in range(1,len(matrix)):

        flag = 0
        for j in range(0,len(matrix[i])):

            if matrix[i][j] != matrix[i-1][j]:
                # rows are correctly charanterised as being interesting or not
                flag = 1
                if matrix[i][j] != 0:
                    count1 += 1
                    pattern[0].append(midi.NoteOnEvent(tick=tick_count, channel=0, data=[j, int(matrix[i][j]) ]))
                elif matrix[i][j] == 0:
                    count2 += 1
                    pattern[0].append(midi.NoteOffEvent(tick=tick_count, channel=0, data=[i, 0]))

            if flag:
                tick_count = 0
                flag = 0

        tick_count += 1

    pattern[0].append(midi.EndOfTrackEvent(tick=1,data=[]))
    print count1
    print count2
    return pattern

# song = midi.read_midifile("bebop.midi")
# # print len(song[0])
# matrix = midi_to_matrix("bebop.midi")
# # print len(matrix)
# pattern = matrix_to_midi(matrix)
# # print len(pattern[0])

# for (e1,e2) in zip(song[0],pattern[0]):
#     if type(e1)==type(e2):
#         print e1
#         print e2
#     else:
#         print e1
#         print e2
#         break
