import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import collections
import random
import threading
import queue
import sys
import os
import subprocess
import tempfile

# Optional dependency: pyrubberband for high-quality time-stretching
# User requested to disable pyrubberband, so we will always use librosa's time_stretch.
HAS_RUBBERBAND = False

# --- Configuration ---
SR = 44100  # Target sample rate
AUDIO_EXTS = ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.m4a']
VIDEO_EXTS = ['.mp4', '.mov', '.avi', '.mkv']


# --- Data Classes ---
class FileMeta:
    """Holds metadata for a single media file."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        _, ext = os.path.splitext(self.filename)
        self.file_type = 'video' if ext.lower() in VIDEO_EXTS else 'audio'

        # Audio-specific
        self.audio = None
        self.bpm = 0
        self.first_beat_offset_ms = 0
        self.stretch = tk.BooleanVar(value=True)
        self.stretch_ratio = 1.0

        # Video-specific
        self.fps = 0
        self.frame_count = 0


class ProjectState:
    """Manages the overall state of the project, including files, grains, and models."""
    def __init__(self):
        self.all_files = []
        self.audio_files = []
        self.video_files = []
        self.grains = []
        self.video_grains = [] # List of (filepath, start_frame, end_frame)
        self.features = []
        self.scaled_features = []
        self.kmeans = None
        self.markov_model = {}
        self.grain_map = {} # Maps cluster label to list of grain indices
        self.output_audio_grains = []
        self.output_video_grains = []

# --- Helper Functions ---
def apply_fade(audio, fade_len, fade_in=True, fade_out=True):
    """Applies a linear fade-in and/or fade-out to an audio segment."""
    fade = np.linspace(0., 1., fade_len)
    if fade_in:
        audio[:fade_len] = audio[:fade_len] * fade
    if fade_out:
        audio[-fade_len:] = audio[-fade_len:] * fade[::-1]
    return audio

def normalize_audio(audio):
    """Normalizes audio to the range [-1, 1]."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio

# --- Main Application Class ---
class GranularGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Granular Markov Sequencer")
        self.geometry("1000x800")

        self.project = ProjectState()
        self.is_playing = False
        self.play_thread = None
        self.stop_event = threading.Event()
        self.ui_queue = queue.Queue()

        # Create a scrollable main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.create_widgets()
        self.check_ui_queue()

    def create_widgets(self):
        """Creates all the GUI widgets."""
        frame = self.scrollable_frame

        # --- Output Device ---
        device_frame = ttk.LabelFrame(frame, text="1. Output Device")
        device_frame.pack(fill=tk.X, padx=10, pady=5)

        self.device_var = tk.StringVar(self, value=str(sd.default.device[1]))
        devices = [dev['name'] for dev in sd.query_devices() if dev['max_output_channels'] > 0]
        device_menu = ttk.OptionMenu(device_frame, self.device_var, self.device_var.get(), *devices)
        device_menu.pack(fill=tk.X, padx=5, pady=5)

        # --- File Selection ---
        file_frame = ttk.LabelFrame(frame, text="2. Media Files")
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(file_frame, text="Select Files...", command=self.select_files).pack(pady=5)

        # File table
        self.tree = ttk.Treeview(file_frame, columns=("type", "filename", "bpm", "offset", "stretch", "ratio"), show="headings")
        self.tree.heading("type", text="Type",)
        self.tree.column("type", width=60, anchor='center')
        self.tree.heading("filename", text="File")
        self.tree.heading("bpm", text="BPM / FPS")
        self.tree.column("bpm", width=80, anchor='center')
        self.tree.heading("offset", text="Offset (ms)")
        self.tree.column("offset", width=80, anchor='center')
        self.tree.heading("stretch", text="Stretch?")
        self.tree.column("stretch", width=60, anchor='center')
        self.tree.heading("ratio", text="Ratio")
        self.tree.column("ratio", width=60, anchor='center')
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree.bind("<Button-1>", self.on_tree_click)

        # --- Parameter Sections (Side-by-side) ---
        param_container = ttk.Frame(frame)
        param_container.pack(fill=tk.X, padx=5, pady=5)

        # --- Global Parameters ---
        params_frame = ttk.LabelFrame(param_container, text="3. Global & Generation Parameters")
        params_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5, expand=True)

        self.target_bpm = self.create_param_entry(params_frame, "Target BPM:", 120.0)
        self.k_states = self.create_param_entry(params_frame, "K (Clusters):", 16)
        self.output_beats = self.create_param_entry(params_frame, "Output Beats:", 64)
        self.crossfade_ms = self.create_param_entry(params_frame, "Crossfade (ms):", 10.0)
        self.trim_db = self.create_param_entry(params_frame, "Silence Trim (dB):", 60.0)

        # --- Markov Parameters ---
        markov_frame = ttk.LabelFrame(param_container, text="4. Markov Model Parameters")
        markov_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5, expand=True)

        self.markov_order = self.create_param_entry(markov_frame, "Order (1 or 2):", 1)
        self.markov_alpha = self.create_param_entry(markov_frame, "Smoothing (α):", 0.1)
        self.markov_temp = self.create_param_entry(markov_frame, "Temperature (T):", 1.0)

        # --- Workflow ---
        workflow_frame = ttk.LabelFrame(frame, text="5. Workflow")
        workflow_frame.pack(fill=tk.X, padx=10, pady=5)

        self.analyze_button = ttk.Button(workflow_frame, text="Analyze Files", command=self.run_analysis)
        self.analyze_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
        self.analysis_progress = ttk.Progressbar(workflow_frame, orient='horizontal', mode='determinate')
        self.analysis_progress.pack(side=tk.LEFT, fill=tk.X, padx=5, expand=True)

        self.extract_button = ttk.Button(workflow_frame, text="Extract Grains", command=self.run_extraction)
        self.extract_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
        self.extraction_progress = ttk.Progressbar(workflow_frame, orient='horizontal', mode='determinate')
        self.extraction_progress.pack(side=tk.LEFT, fill=tk.X, padx=5, expand=True)

        self.generate_button = ttk.Button(workflow_frame, text="Generate Sequence", command=self.run_generation)
        self.generate_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
        self.generation_progress = ttk.Progressbar(workflow_frame, orient='horizontal', mode='determinate')
        self.generation_progress.pack(side=tk.LEFT, fill=tk.X, padx=5, expand=True)

        # --- Playback and Save ---
        output_frame = ttk.LabelFrame(frame, text="6. Output")
        output_frame.pack(fill=tk.X, padx=10, pady=5)

        self.play_button = ttk.Button(output_frame, text="Play Audio", command=self.play_stop)
        self.play_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_audio_button = ttk.Button(output_frame, text="Save Audio Only...", command=self.save_wav)
        self.save_audio_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_video_button = ttk.Button(output_frame, text="Save Final Video...", command=self.save_final_video)
        self.save_video_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.output_audio = None

    def create_param_entry(self, parent, label, default_value):
        """Helper to create a label and entry for a parameter."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(frame, text=label, width=15).pack(side=tk.LEFT)
        var_type = type(default_value)
        if var_type is int:
            var = tk.IntVar(master=self, value=default_value)
        elif var_type is float:
            var = tk.DoubleVar(master=self, value=default_value)
        else:
            var = tk.StringVar(master=self, value=str(default_value))

        entry = ttk.Entry(frame, textvariable=var, width=10)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return var

    def on_tree_click(self, event):
        """Handle clicks on the 'stretch' column to toggle the boolean var."""
        region = self.tree.identify("region", event.x, event.y)
        column = self.tree.identify_column(event.x)
        if region == "cell" and column == "#5": # Column "stretch"
            item_id = self.tree.focus()
            if item_id:
                item_index = self.tree.index(item_id)
                file_meta = self.project.all_files[item_index]
                if file_meta.file_type == 'audio':
                    file_meta.stretch.set(not file_meta.stretch.get())
                    self.update_file_table()

    def update_file_table(self):
        """Refreshes the file table with current data from project state."""
        for i in self.tree.get_children():
            self.tree.delete(i)

        for f in self.project.all_files:
            if f.file_type == 'audio':
                stretch_val = "Yes" if f.stretch.get() else "No"
                bpm_val = f"{f.bpm:.2f}"
                offset_val = f"{f.first_beat_offset_ms:.2f}"
                ratio_val = f"{f.stretch_ratio:.3f}"
            else: # video
                stretch_val = "N/A"
                bpm_val = f"{f.fps:.2f}"
                offset_val = "N/A"
                ratio_val = "N/A"

            self.tree.insert("", "end", values=(
                f.file_type.capitalize(),
                f.filename,
                bpm_val,
                offset_val,
                stretch_val,
                ratio_val
            ))

    def select_files(self):
        """Opens a file dialog to select audio and video files."""
        filetypes = (
            ("Media Files", ' '.join(AUDIO_EXTS) + ' ' + ' '.join(VIDEO_EXTS)),
            ("Audio Files", ' '.join(AUDIO_EXTS)),
            ("Video Files", ' '.join(VIDEO_EXTS)),
            ("All files", "*.*")
        )
        filepaths = filedialog.askopenfilenames(
            title="Select Media Files",
            filetypes=filetypes
        )
        if filepaths:
            self.project.all_files = [FileMeta(fp) for fp in filepaths]
            self.project.audio_files = [f for f in self.project.all_files if f.file_type == 'audio']
            self.project.video_files = [f for f in self.project.all_files if f.file_type == 'video']
            self.update_file_table()
            messagebox.showinfo("Files Selected", f"{len(filepaths)} files selected. Ready to analyze.")

    # --- Threaded Workflow Functions ---

    def run_in_thread(self, target, *args):
        """Runs a function in a separate thread to avoid blocking the GUI."""
        thread = threading.Thread(target=target, args=args)
        thread.daemon = True
        thread.start()

    def run_analysis(self):
        if not self.project.all_files:
            messagebox.showwarning("No Files", "Please select media files first.")
            return
        self.run_in_thread(self._analyze_files_task)

    def run_extraction(self):
        if not self.project.all_files:
            messagebox.showwarning("No Files", "Please select media files first.")
            return
        if not self.project.audio_files and self.project.video_files:
             messagebox.showwarning("Not Analyzed", "Please analyze the files first.")
             return
        self.run_in_thread(self._extract_grains_task)

    def run_generation(self):
        if not self.project.grains:
            messagebox.showwarning("No Grains", "Please extract grains first.")
            return
        self.run_in_thread(self._generate_sequence_task)

    # --- Core Logic Tasks (executed in threads) ---

    def _analyze_files_task(self):
        """Task to load, analyze, and extract metadata from all media files."""
        self.ui_queue.put(("set_progress", self.analysis_progress, 0))
        target_bpm = self.target_bpm.get()
        trim_db = self.trim_db.get()

        for i, f in enumerate(self.project.all_files):
            try:
                if f.file_type == 'audio':
                    # Load, resample to mono 44.1kHz
                    audio, _ = librosa.load(f.filepath, sr=SR, mono=True)

                    # Trim leading/trailing silence
                    audio, _ = librosa.effects.trim(audio, top_db=trim_db)
                    f.audio = audio

                    # Detect BPM and beat positions
                    tempo, beats = librosa.beat.beat_track(y=audio, sr=SR)
                    f.bpm = float(tempo)

                    # Find first non-silent beat
                    beat_times = librosa.frames_to_time(beats, sr=SR)
                    f.first_beat_offset_ms = beat_times[0] * 1000 if len(beat_times) > 0 else 0

                    # Calculate stretch ratio if needed
                    if f.stretch.get():
                        f.stretch_ratio = target_bpm / f.bpm if f.bpm > 0 else 1.0
                    else:
                        f.stretch_ratio = 1.0

                elif f.file_type == 'video':
                    cap = cv2.VideoCapture(f.filepath)
                    if not cap.isOpened():
                        raise Exception("Could not open video file.")
                    f.fps = cap.get(cv2.CAP_PROP_FPS)
                    f.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

            except Exception as e:
                self.ui_queue.put(("error", "Analysis Error", f"Could not process {f.filename}:\n{e}"))

            self.ui_queue.put(("set_progress", self.analysis_progress, (i + 1) * 100 / len(self.project.all_files)))

        self.ui_queue.put(("update_table",))
        self.ui_queue.put(("info", "Analysis Complete", "File analysis finished."))

    def _extract_grains_task(self):
        """Task to process audio and video files into discrete 'grains'."""
        self.ui_queue.put(("set_progress", self.extraction_progress, 0))
        self.project.grains = []
        self.project.features = []
        self.project.video_grains = []
        self.project.scaled_features = []

        target_bpm = self.target_bpm.get()
        if target_bpm <= 0:
            self.ui_queue.put(("error", "Invalid BPM", "Target BPM must be positive."))
            return
        beat_len_sec = 60.0 / target_bpm

        total_files = len(self.project.audio_files) + len(self.project.video_files)
        if total_files == 0:
            self.ui_queue.put(("info", "Extraction Complete", "No files to process."))
            return

        files_processed = 0

        # --- 1. Process Audio Files ---
        if self.project.audio_files:
            beat_len_samples = int(beat_len_sec * SR)
            fade_len = int(self.crossfade_ms.get() / 1000 * SR / 2)

            for f in self.project.audio_files:
                if f.audio is not None:
                    stretched_audio = f.audio
                    if f.stretch.get() and abs(f.stretch_ratio - 1.0) > 0.001:
                        if HAS_RUBBERBAND:
                            stretched_audio = pyrb.time_stretch(f.audio, SR, f.stretch_ratio)
                        else:
                            stretched_audio = librosa.effects.time_stretch(f.audio, rate=f.stretch_ratio)

                    start_sample = int(f.first_beat_offset_ms / 1000 * SR)
                    for start in range(start_sample, len(stretched_audio) - beat_len_samples, beat_len_samples):
                        end = start + beat_len_samples
                        grain = np.copy(stretched_audio[start:end])
                        if len(grain) < beat_len_samples:
                            grain = np.pad(grain, (0, beat_len_samples - len(grain)))

                        self.project.grains.append(apply_fade(grain, fade_len))
                        self.project.features.append(self._extract_features(grain))

                files_processed += 1
                self.ui_queue.put(("set_progress", self.extraction_progress, files_processed * 100 / total_files))

            if not self.project.features:
                self.ui_queue.put(("error", "Extraction Error", "No audio grains extracted. Check audio file analysis."))
                return

            # Standardize audio features
            features_np = np.array(self.project.features)
            self.project.scaled_features = StandardScaler().fit_transform(features_np)

        # --- 2. Process Video Files ---
        if self.project.video_files:
            for f in self.project.video_files:
                if f.fps > 0 and f.frame_count > 0:
                    frames_per_grain = int(beat_len_sec * f.fps)
                    if frames_per_grain > 0:
                        for start_frame in range(0, f.frame_count - frames_per_grain, frames_per_grain):
                            end_frame = start_frame + frames_per_grain
                            self.project.video_grains.append((f.filepath, start_frame, end_frame))

                files_processed += 1
                self.ui_queue.put(("set_progress", self.extraction_progress, files_processed * 100 / total_files))

        # --- 3. Final Report ---
        audio_msg = f"Extracted {len(self.project.grains)} audio grains."
        video_msg = f"Created {len(self.project.video_grains)} video grain references."

        if self.project.grains and self.project.video_grains:
            final_msg = f"{audio_msg} {video_msg}"
        elif self.project.grains:
            final_msg = audio_msg
        elif self.project.video_grains:
            final_msg = video_msg
        else:
            final_msg = "No grains were extracted."

        self.ui_queue.put(("set_progress", self.extraction_progress, 100))
        self.ui_queue.put(("info", "Extraction Complete", final_msg))

    def _extract_features(self, grain):
        """Extracts a feature vector for a single audio grain."""
        n_fft = 2048
        hop_length = 512
        frame_length_rms = 1024 # Recommended to match STFT for RMS

        S, _ = librosa.magphase(librosa.stft(y=grain, n_fft=n_fft, hop_length=hop_length))

        # Spectral features
        centroid = librosa.feature.spectral_centroid(S=S, sr=SR).mean()
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=SR).mean()
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=SR).mean()
        flatness = librosa.feature.spectral_flatness(S=S).mean()

        # RMS
        rms = librosa.feature.rms(y=grain, frame_length=frame_length_rms, hop_length=hop_length).mean()

        # MFCCs
        mfccs = librosa.feature.mfcc(y=grain, sr=SR, n_mfcc=13).mean(axis=1)

        # Chroma
        chroma = librosa.feature.chroma_stft(S=S, sr=SR, n_chroma=12).mean(axis=1)

        # Harmonic-Percussive Ratio
        y_harm, y_perc = librosa.effects.hpss(grain)
        h_rms = np.mean(librosa.feature.rms(y=y_harm, frame_length=frame_length_rms, hop_length=hop_length))
        p_rms = np.mean(librosa.feature.rms(y=y_perc, frame_length=frame_length_rms, hop_length=hop_length))
        hpr = p_rms / (h_rms + 1e-6) # Percussive to Harmonic ratio

        feature_vector = np.concatenate([
            [centroid, bandwidth, rolloff, flatness, rms, hpr],
            mfccs,
            chroma
        ])
        return feature_vector

    def _generate_sequence_task(self):
        """Task to build the Markov model and generate the new audio sequence."""
        self.ui_queue.put(("set_progress", self.generation_progress, 0))

        k = self.k_states.get()
        order = self.markov_order.get()
        alpha = self.markov_alpha.get()
        temp = self.markov_temp.get()
        num_beats = self.output_beats.get()

        # 1. Cluster grains into K states
        try:
            self.project.kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = self.project.kmeans.fit_predict(self.project.scaled_features)
        except Exception as e:
            self.ui_queue.put(("error", "Clustering Error", f"KMeans failed: {e}\nTry a smaller K value."))
            return

        self.ui_queue.put(("set_progress", self.generation_progress, 20))

        # Create a map from cluster label to grain indices
        self.project.grain_map = collections.defaultdict(list)
        for i, label in enumerate(labels):
            self.project.grain_map[label].append(i)

        # 2. Build Markov chain
        self.project.markov_model = {}
        if order == 1:
            transitions = collections.defaultdict(lambda: collections.defaultdict(int))
            for i in range(len(labels) - 1):
                transitions[labels[i]][labels[i+1]] += 1

            # Normalize with add-alpha smoothing
            for state, next_states in transitions.items():
                total = sum(next_states.values()) + alpha * k
                self.project.markov_model[state] = {
                    s: (c + alpha) / total for s, c in next_states.items()
                }
        elif order == 2:
            transitions = collections.defaultdict(lambda: collections.defaultdict(int))
            for i in range(len(labels) - 2):
                key = (labels[i], labels[i+1])
                transitions[key][labels[i+2]] += 1

            # Normalize with add-alpha smoothing
            for state_tuple, next_states in transitions.items():
                total = sum(next_states.values()) + alpha * k
                self.project.markov_model[state_tuple] = {
                    s: (c + alpha) / total for s, c in next_states.items()
                }
        else:
            self.ui_queue.put(("error", "Invalid Order", "Markov order must be 1 or 2."))
            return

        self.ui_queue.put(("set_progress", self.generation_progress, 40))

        # 3. Generate new sequence of states
        if not self.project.markov_model:
            self.ui_queue.put(("error", "Model Empty", "Markov model is empty. Not enough transitions found."))
            return

        current_state = random.choice(list(self.project.markov_model.keys()))
        state_sequence = list(current_state) if order == 2 else [current_state]

        for _ in range(num_beats - order):
            if current_state not in self.project.markov_model:
                # Fallback: choose a random state if we hit a dead end
                current_state = random.choice(list(self.project.markov_model.keys()))
                if order == 2: state_sequence.extend(list(current_state))
                else: state_sequence.append(current_state)

            probs = self.project.markov_model[current_state]

            # Apply temperature sampling
            if temp > 0:
                prob_vals = np.array(list(probs.values()))
                log_probs = np.log(prob_vals + 1e-9) / temp
                exp_probs = np.exp(log_probs)
                softmax_probs = exp_probs / np.sum(exp_probs)
                next_state = np.random.choice(list(probs.keys()), p=softmax_probs)
            else: # Greedy sampling for T=0
                next_state = max(probs, key=probs.get)

            state_sequence.append(next_state)

            if order == 1:
                current_state = next_state
            elif order == 2:
                current_state = (current_state[1], next_state)

        self.ui_queue.put(("set_progress", self.generation_progress, 70))

        # 4. Select Audio and Video Grains for the sequence
        self.project.output_audio_grains = []
        self.project.output_video_grains = []
        for state in state_sequence:
            # Select audio grain from the corresponding cluster
            grain_idx = random.choice(self.project.grain_map[state])
            self.project.output_audio_grains.append(self.project.grains[grain_idx])

            # Select a random video grain if any are available
            if self.project.video_grains:
                video_grain_ref = random.choice(self.project.video_grains)
                self.project.output_video_grains.append(video_grain_ref)

        # 5. Concatenate audio grains to create the final audio output
        output_audio = []
        crossfade_len = int(self.crossfade_ms.get() / 1000 * SR)

        for i, grain in enumerate(self.project.output_audio_grains):
            if i == 0:
                output_audio.append(grain)
            else:
                # Ensure overlap is possible
                if len(output_audio[-1]) > crossfade_len and len(grain) > crossfade_len:
                    overlap = output_audio[-1][-crossfade_len:]
                    grain_start = grain[:crossfade_len]

                    fade_out = np.linspace(1, 0, crossfade_len)
                    fade_in = np.linspace(0, 1, crossfade_len)
                    crossfaded_part = overlap * fade_out + grain_start * fade_in

                    output_audio[-1] = np.concatenate([output_audio[-1][:-crossfade_len], crossfaded_part])
                    output_audio.append(grain[crossfade_len:])
                else: # If grain is too short, just append
                    output_audio.append(grain)

            self.ui_queue.put(("set_progress", self.generation_progress, 70 + (i * 30 / len(state_sequence))))

        self.output_audio = normalize_audio(np.concatenate(output_audio)) if output_audio else np.array([])

        audio_msg = f"Generated {num_beats}-beat audio sequence."
        video_msg = "Video sequence is ready to be rendered." if self.project.output_video_grains else ""
        final_msg = " ".join(filter(None, [audio_msg, video_msg]))

        self.ui_queue.put(("set_progress", self.generation_progress, 100))
        self.ui_queue.put(("info", "Generation Complete", final_msg))

    # --- UI Queue and Playback ---

    def check_ui_queue(self):
        """Checks for messages from worker threads and updates the UI."""
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                cmd, *args = msg

                if cmd == "set_progress":
                    progress_bar, value = args
                    progress_bar['value'] = value
                elif cmd == "update_table":
                    self.update_file_table()
                elif cmd == "info":
                    title, message = args
                    messagebox.showinfo(title, message)
                elif cmd == "error":
                    title, message = args
                    messagebox.showerror(title, message)
                elif cmd == "play_stopped":
                    self.play_button.config(text="Play")
                    self.is_playing = False
                    self.stop_event.clear()

        except queue.Empty:
            pass
        self.after(100, self.check_ui_queue)

    def play_stop(self):
        """Toggles playback of the generated audio."""
        if self.is_playing:
            self.stop_event.set() # Signal the thread to stop
        else:
            if self.output_audio is None:
                messagebox.showwarning("No Audio", "Please generate a sequence first.")
                return

            self.is_playing = True
            self.play_button.config(text="Stop")
            self.stop_event.clear()
            self.play_thread = threading.Thread(target=self._play_audio_task)
            self.play_thread.daemon = True
            self.play_thread.start()

    def _play_audio_task(self):
        """Task to play audio using sounddevice, checking for stop signal."""
        try:
            device_name = self.device_var.get()
            sd.play(self.output_audio, samplerate=SR, device=device_name, blocking=False)

            # Wait for playback to finish or for stop event
            sd.wait()
            # while self.is_playing:
            #     if self.stop_event.is_set():
            #         sd.stop()
            #         break
            #     sd.sleep(50) # check for stop every 50ms

        except Exception as e:
            self.ui_queue.put(("error", "Playback Error", f"Could not play audio:\n{e}"))
        finally:
            sd.stop()
            self.ui_queue.put(("play_stopped",))

    def save_wav(self):
        """Saves the generated audio to a WAV file."""
        if self.output_audio is None:
            messagebox.showwarning("No Audio", "Please generate a sequence first.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=(("WAV files", "*.wav"),),
            title="Save Generated Audio"
        )
        if filepath:
            try:
                sf.write(filepath, self.output_audio, SR)
                messagebox.showinfo("Save Complete", f"Audio saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save file:\n{e}")

    def save_final_video(self):
        """Initiates the process to render and combine audio and video."""
        if self.output_audio is None or len(self.output_audio) == 0:
            messagebox.showwarning("No Audio", "Please generate a sequence first.")
            return
        if not self.project.output_video_grains:
            messagebox.showwarning("No Video", "The generated sequence does not contain video.")
            return

        # Check for ffmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            messagebox.showerror("FFmpeg Error", "FFmpeg not found. Please install ffmpeg and ensure it's in your system's PATH to save the final video.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=(("MP4 Video", "*.mp4"), ("All Files", "*.*")),
            title="Save Final Combined Video"
        )
        if filepath:
            self.run_in_thread(self._render_and_combine_task, filepath)

    def _render_and_combine_task(self, final_filepath):
        """Renders video, saves audio, and merges them using ffmpeg."""
        progress_bar = self.generation_progress
        self.ui_queue.put(("set_progress", progress_bar, 0))

        silent_video_file = None
        temp_audio_file = None

        try:
            # --- 1. Render Silent Video to a temporary file ---
            self.ui_queue.put(("info", "Rendering", "Step 1/3: Rendering silent video..."))
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                silent_video_file = f.name

            # Get video properties
            first_grain_path = self.project.output_video_grains[0][0]
            cap_probe = cv2.VideoCapture(first_grain_path)
            fps = cap_probe.get(cv2.CAP_PROP_FPS)
            width = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_size = (width, height)
            cap_probe.release()
            if fps <= 0: fps = 24

            # Write video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(silent_video_file, fourcc, fps, frame_size)
            total_grains = len(self.project.output_video_grains)
            cap = None
            last_filepath = ""

            for i, (filepath, start_frame, end_frame) in enumerate(self.project.output_video_grains):
                if filepath != last_filepath:
                    if cap: cap.release()
                    cap = cv2.VideoCapture(filepath)
                    if not cap.isOpened(): continue
                    last_filepath = filepath

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for _ in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret: break
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
                    writer.write(frame)
                self.ui_queue.put(("set_progress", progress_bar, (i + 1) * 100 / (total_grains * 3)))

            if cap: cap.release()
            writer.release()
            self.ui_queue.put(("set_progress", progress_bar, 33))

            # --- 2. Save Audio to a temporary file ---
            self.ui_queue.put(("info", "Saving Audio", "Step 2/3: Saving temporary audio file..."))
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_audio_file = f.name
            sf.write(temp_audio_file, self.output_audio, SR)
            self.ui_queue.put(("set_progress", progress_bar, 66))

            # --- 3. Combine with ffmpeg ---
            self.ui_queue.put(("info", "Combining", "Step 3/3: Combining audio and video with FFmpeg..."))
            command = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-i', silent_video_file,
                '-i', temp_audio_file,
                '-c:v', 'copy',      # Copy video stream without re-encoding
                '-c:a', 'aac',       # Encode audio to AAC
                '-shortest',         # Finish encoding on shortest stream
                final_filepath
            ]
            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"FFmpeg failed with error:\n{result.stderr}")

            self.ui_queue.put(("set_progress", progress_bar, 100))
            self.ui_queue.put(("info", "Success", f"Final video saved to {final_filepath}"))

        except Exception as e:
            self.ui_queue.put(("error", "Processing Error", f"Failed to create final video:\n{e}"))
        finally:
            # --- 4. Cleanup ---
            if silent_video_file and os.path.exists(silent_video_file):
                os.remove(silent_video_file)
            if temp_audio_file and os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
            self.ui_queue.put(("set_progress", progress_bar, 0))

    def on_closing(self):
        """Handles window closing event."""
        if self.is_playing:
            self.stop_event.set()
        self.destroy()

if __name__ == "__main__":
    if not HAS_RUBBERBAND:
        print("Warning: pyrubberband not found. Time stretching quality will be lower.", file=sys.stderr)

    app = GranularGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
