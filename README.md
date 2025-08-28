# Granular Markov Sequencer

This project is a sophisticated audio mangling tool that generates new musical sequences from a collection of source audio files. It combines the techniques of granular synthesis with the predictive power of Markov chains to create novel, complex, and often surprising rhythmic and melodic patterns.

The application provides a graphical user interface (GUI) to load audio, analyze it, build a probabilistic model, and generate new audio based on the learned transitions between sonic textures.

## Features

- **Intuitive GUI:** A clear, step-by-step interface built with Tkinter.
- **Unified Media Support:** Load and process both audio (`.wav`, `.mp3`, etc.) and video (`.mp4`, `.mov`, etc.) files in the same session.
- **Audio-Driven Generation:** The core sequencing logic is driven by the analysis of audio files.
- **Automatic Analysis:**
    - For audio, detects BPM and the first beat.
    - For video, detects FPS and frame count.
    - Optionally time-stretches audio to a target BPM.
- **Synchronized Graining:**
    - Slices audio into beat-synchronized "grains".
    - Slices video into video clips of the same duration, creating a pool of video "grains".
- **Machine Learning Core:**
    - Builds a Markov Chain model based on the acoustic features of the *audio* grains.
- **Creative Sequence Generation:**
    - Generates a new sequence based on the learned audio transitions.
    - For each step in the sequence, a random video grain is selected to create a new visual montage.
    - "Temperature" control allows you to adjust the randomness of the audio sequence.
- **High-Quality Output:**
    - Save the generated audio separately as a `.wav` file.
    - Render the final audio and video montage into a single `.mp4` file using FFmpeg.

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
    pip install numpy librosa soundfile sounddevice scikit-learn opencv-python
    ```

    *Note: The application uses `tkinter` for the GUI, which is typically included with standard Python installations. If it's missing, you may need to install it separately (e.g., `sudo apt-get install python3-tk` on Debian/Ubuntu).*

3.  **Install FFmpeg (Required for Video Export):**
    To save the final combined audio/video file, you must have **FFmpeg** installed on your system and accessible in your PATH.
    - **Windows:** Download a build from the [official FFmpeg website](https://ffmpeg.org/download.html) and add the `bin` directory to your system's PATH environment variable.
    - **macOS (using Homebrew):** `brew install ffmpeg`
    - **Debian/Ubuntu:** `sudo apt-get install ffmpeg`

## How to Use

1.  **Run the application:**
    ```bash
    python granular_sequencer.py
    ```

2.  **Select Output Device:** From the first dropdown menu, choose the audio device you want to use for playback.

3.  **Select Media Files:** Click the "Select Files..." button to load your source audio and video files. Their details will appear in the table. You must include at least one audio file to drive the generation.

4.  **Configure Parameters:**
    - **Global & Generation:** Set the `Target BPM` for the output. This controls the speed of both the audio and video cuts.
    - **Markov Model:** Configure the model that learns from your audio files.

5.  **Execute the Workflow:** Press the main buttons in order from left to right.
    - **Analyze Files:** Loads media, finds BPM for audio and FPS for video.
    - **Extract Grains:** Slices audio and creates video grain references.
    - **Generate Sequence:** Builds the Markov model from the audio and generates a new sequence, picking random video clips to match the audio.

6.  **Audition and Save:**
    - **Play Audio:** Listen to the generated audio track.
    - **Save Audio Only...:** Export just the audio as a `.wav` file.
    - **Save Final Video...:** Combine the generated audio and video into a final `.mp4` file. **(Requires FFmpeg)**.

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
