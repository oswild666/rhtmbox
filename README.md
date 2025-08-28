# Granular Markov Sequencer

This project is a sophisticated audio mangling tool that generates new musical sequences from a collection of source audio files. It combines the techniques of granular synthesis with the predictive power of Markov chains to create novel, complex, and often surprising rhythmic and melodic patterns.

The application provides a graphical user interface (GUI) to load audio, analyze it, build a probabilistic model, and generate new audio based on the learned transitions between sonic textures.

## Features

- **Intuitive GUI:** A clear, step-by-step interface built with Tkinter.
- **Multi-file Support:** Load and process multiple audio files (`.wav`, `.mp3`, `.flac`, etc.) at once.
- **Automatic Audio Analysis:**
    - Detects BPM and the first beat of each file.
    - Optionally time-stretches audio to a target BPM to synchronize all sources.
    - Trims leading and trailing silence to process only meaningful audio.
- **Granular Synthesis Engine:**
    - Slices audio into beat-synchronized "grains".
    - Applies fade-in/out to each grain to prevent clicks.
- **Machine Learning Core:**
    - Extracts a rich set of audio features for each grain (spectral properties, MFCCs, RMS, etc.).
    - Uses K-Means clustering to group acoustically similar grains into "states".
    - Builds a 1st or 2nd order Markov Chain to model the transitions between these states.
- **Creative Sequence Generation:**
    - Generates a new sequence of states by traversing the Markov model.
    - "Temperature" control allows you to adjust the randomness of the output, from predictable to chaotic.
- **High-Quality Output:**
    - Concatenates grains with a crossfade for a smooth, continuous audio stream.
    - Allows direct playback of the generated sequence through a selected audio device.
    - Save your creation as a high-quality `.wav` file.

## Installation & Dependencies

This script is written for Python 3. It relies on several external libraries for audio processing and machine learning.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the required packages:**
    You can install all dependencies using pip:
    ```bash
    pip install numpy librosa soundfile sounddevice scikit-learn
    ```

    *Note: The application uses `tkinter` for the GUI, which is typically included with standard Python installations. If it's missing, you may need to install it separately (e.g., `sudo apt-get install python3-tk` on Debian/Ubuntu).*

## How to Use

1.  **Run the application:**
    ```bash
    python granular_sequencer.py
    ```

2.  **Select Output Device:** From the first dropdown menu, choose the audio device you want to use for playback.

3.  **Select Audio Files:** Click the "Select Files..." button to load one or more audio files. Their details will appear in the table.

4.  **Configure Parameters:**
    - **Global & Generation:** Set the `Target BPM` for the output, the number of `K (Clusters)` to group grains into, the total `Output Beats` for the sequence, and other settings.
    - **Markov Model:** Choose the `Order` (1 or 2) and adjust `Smoothing` and `Temperature` to control the generation logic.

5.  **Execute the Workflow:** Press the main buttons in order from left to right.
    - **Analyze Files:** Loads audio, finds BPM, and calculates the required time-stretch ratio for each file.
    - **Extract Grains:** Stretches the audio, slices it into beat-sized grains, and runs feature extraction and clustering.
    - **Generate Sequence:** Builds the Markov model and creates the new audio sequence.

6.  **Audition and Save:**
    - Click **Play** to listen to the generated sequence.
    - Click **Save WAV...** to export the result to a file.

*(Note: It is recommended to add a screenshot of the application GUI here for clarity.)*

## How It Works

The magic of this sequencer comes from two core concepts:

### 1. Granular Synthesis & Feature Clustering

Instead of playing back whole files, the application chops the source audio into tiny, beat-long pieces called **grains**. It then analyzes the acoustic properties of every single grain, creating a mathematical "fingerprint" (a feature vector) for each one.

Using the **K-Means clustering** algorithm, it groups all the grains into a handful of clusters (the `K` value you set). All grains in a single cluster sound acoustically similar to each other. These clusters become the "states" of our musical machine.

### 2. Markov Chains

A **Markov chain** is a model that describes a sequence of events where the probability of each event depends only on the state of the previous event (or previous few events).

This application builds a Markov chain by looking at the original sequence of grains from your source files. It learns the probability of transitioning from one sound-cluster to another. For example, it might learn that a "kick drum" cluster is often followed by a "hi-hat" cluster.

When you click "Generate Sequence," the application performs a "random walk" on this chain: it starts at a random cluster and then rolls a dice to decide which cluster to move to next, based on the learned probabilities. It then picks a random grain from the chosen cluster and adds it to the output, repeating the process to build a full sequence. The **Temperature** parameter skews this dice roll, making the choices more or less random.
