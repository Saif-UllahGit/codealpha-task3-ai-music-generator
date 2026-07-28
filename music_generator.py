import os
import numpy as np

from music21 import converter, note, chord, stream

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# ======================================================
# LOAD MIDI FILES
# ======================================================

def load_midi_files():

    notes = []

    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "midi_dataset"
    )

    print("\nLoading MIDI files...\n")

    if not os.path.exists(dataset_path):
        print("Dataset folder not found!")
        return notes

    for file in os.listdir(dataset_path):

        if file.lower().endswith(".mid"):

            print("Reading:", file)

            midi = converter.parse(
                os.path.join(dataset_path, file)
            )

            for element in midi.recurse():

                if isinstance(element, note.Note):

                    notes.append(
                        str(element.pitch)
                    )

                elif isinstance(element, chord.Chord):

                    notes.append(
                        ".".join(
                            str(n)
                            for n in element.normalOrder
                        )
                    )

    print("\nTotal Notes Found:", len(notes))

    return notes


# ======================================================
# PREPARE TRAINING DATA
# ======================================================

def prepare_sequences(notes):

    sequence_length = 50

    pitchnames = sorted(set(notes))

    note_to_int = {
        n: i
        for i, n in enumerate(pitchnames)
    }

    network_input = []

    network_output = []

    for i in range(len(notes) - sequence_length):

        sequence_in = notes[i:i + sequence_length]

        sequence_out = notes[i + sequence_length]

        network_input.append(
            [note_to_int[n] for n in sequence_in]
        )

        network_output.append(
            note_to_int[sequence_out]
        )

    network_input = np.reshape(
        network_input,
        (
            len(network_input),
            sequence_length,
            1
        )
    )

    network_input = network_input / float(len(pitchnames))

    network_output = to_categorical(
        network_output
    )

    return network_input, network_output, pitchnames


# ======================================================
# FAST LSTM MODEL
# ======================================================

def create_model(network_input, n_vocab):

    model = Sequential()

    model.add(
        LSTM(
            128,
            input_shape=(
                network_input.shape[1],
                network_input.shape[2]
            ),
            return_sequences=True
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        LSTM(64)
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        Dense(
            64,
            activation="relu"
        )
    )

    model.add(
        Dense(
            n_vocab,
            activation="softmax"
        )
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


# ======================================================
# TRAIN MODEL
# ======================================================

def train_model():

    notes = load_midi_files()

    if len(notes) < 60:

        print("Not enough notes for training.")
        return

    network_input, network_output, pitchnames = prepare_sequences(notes)

    print("\nCreating Model...\n")

    model = create_model(
        network_input,
        len(pitchnames)
    )

    print("Training Started...\n")

    model.fit(
        network_input,
        network_output,
        epochs=10,
        batch_size=128,
        verbose=1
    )

    model.save("music_model.h5")

    np.save(
        "pitchnames.npy",
        np.array(pitchnames)
    )

    print("\nTraining Complete!")
    print("music_model.h5 saved")
    print("pitchnames.npy saved")
# ======================================================
# GENERATE MUSIC
# ======================================================

def generate_music():

    if not os.path.exists("music_model.h5"):

        print("Train the model first!")
        return

    model = load_model("music_model.h5")

    pitchnames = np.load(
        "pitchnames.npy",
        allow_pickle=True
    )

    int_to_note = {
        number: note_name
        for number, note_name in enumerate(pitchnames)
    }

    sequence_length = 50

    pattern = np.random.randint(
        0,
        len(pitchnames),
        sequence_length
    ).tolist()

    prediction_output = []

    print("\nGenerating Music...\n")

    for _ in range(100):

        prediction_input = np.reshape(
            pattern,
            (
                1,
                sequence_length,
                1
            )
        )

        prediction_input = prediction_input / float(len(pitchnames))

        prediction = model.predict(
            prediction_input,
            verbose=0
        )

        index = np.argmax(prediction)

        result = int_to_note[index]

        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:]

    create_midi(prediction_output)

    print("\nMusic saved as generated_music.mid")


# ======================================================
# CREATE MIDI FILE
# ======================================================

def create_midi(prediction_output):

    offset = 0

    output_notes = []

    for pattern in prediction_output:

        if "." in pattern:

            notes_in_chord = pattern.split(".")

            chord_notes = []

            for current_note in notes_in_chord:

                new_note = note.Note(
                    int(current_note)
                )

                new_note.offset = offset

                chord_notes.append(new_note)

            new_chord = chord.Chord(chord_notes)

            new_chord.offset = offset

            output_notes.append(new_chord)

        else:

            new_note = note.Note(pattern)

            new_note.offset = offset

            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write(
        "midi",
        fp="generated_music.mid"
    )


# ======================================================
# MAIN MENU
# ======================================================

def main():

    while True:

        print("\n========== AI MUSIC GENERATOR ==========")
        print("1. Train Model")
        print("2. Generate Music")
        print("3. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":

            train_model()

        elif choice == "2":

            generate_music()

        elif choice == "3":

            print("Goodbye!")
            break

        else:

            print("Invalid Choice!")


if __name__ == "__main__":

    main()