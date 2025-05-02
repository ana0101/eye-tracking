import pymovements as pm
import csv
import polars as pl
import os
import glob
import pandas as pd
from word_fixations import *

class ExtractWordFixations:
    """
    Receives the eye tracking data and the experiment data for a session and extracts the word fixations.
    """

    def __init__(self, asc_file_path, aoi_stimuli_path, general_logfile_path, completed_stimuli_path, dataset_root_path, dataset_name, subject_id):
        self.asc_file_path = asc_file_path
        self.aoi_stimuli_path = aoi_stimuli_path
        self.general_logfile_path = general_logfile_path
        self.completed_stimuli_path = completed_stimuli_path
        self.dataset_root_path = dataset_root_path
        self.dataset_name = dataset_name
        self.dataset = None
        self.subject_id = subject_id
        self.gaze_df = None
        self.question_images_version = None
        

    def get_gaze_df_from_asc_file(self):
        """
        Reads the ASCII raw eye tracking data file and returns a GazeDataFrame.
        """
        print(f"Reading {self.asc_file_path}...")
        data = pm.gaze.from_asc(
            self.asc_file_path,
            patterns=[
                r"start_recording_(?P<trial>(?:PRACTICE_)?trial_\d+)_(?P<screen>.+)",
                {
                    "pattern": r"stop_recording_", 
                    "column": "trial", 
                    "value": None
                },
                {
                    "pattern": r"stop_recording_", 
                    "column": "screen", 
                    "value": None
                },
                {
                    "pattern": r"start_recording_(?:PRACTICE_)?trial_\d+_page_\d+",
                    "column": "activity",
                    "value": "reading",
                },
                {
                    "pattern": r"start_recording_(?:PRACTICE_)?trial_\d+_question_\d+",
                    "column": "activity",
                    "value": "question",
                },
                {
                    "pattern": r"start_recording_(?:PRACTICE_)?trial_\d+_(familiarity_rating_screen_\d+|subject_difficulty_screen)",
                    "column": "activity",
                    "value": "rating",
                },
                {
                    "pattern": r"stop_recording_", 
                    "column": "activity", 
                    "value": None
                },
                {
                    "pattern": r"start_recording_PRACTICE_trial_",
                    "column": "practice",
                    "value": True,
                },
                {
                    "pattern": r"start_recording_trial_",
                    "column": "practice",
                    "value": False,
                },
                {
                    "pattern": r"stop_recording_", 
                    "column": "practice", 
                    "value": None
                },
            ],
        )
        print(f"Finished reading {self.asc_file_path}")
        self.gaze_df = data
    

    def set_stimulus_ids(self):
        """
        Sets the stimulus IDs for the gaze data.
        """
        stimulus_ids = {
            "PRACTICE_trial_1": "practice1",
            "PRACTICE_trial_2": "practice2",
        }

        with open(self.completed_stimuli_path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                stimulus_ids[f"trial_{i + 1}"] = f"stimulus{row['stimulus_id']}"

        df = self.gaze_df.frame.with_columns(
            pl.col("trial").replace(stimulus_ids).alias("stimulus_id")
        )
        self.gaze_df.frame = df


    def split_pixels(self):
        """
        Splits the pixel columns into separate columns for x and y coordinates.
        """
        self.gaze_df.frame = self.gaze_df.frame.select(
            [
                pl.all().exclude("pixel"),
                pl.col("pixel").list.get(0).alias("pixel_x"),
                pl.col("pixel").list.get(1).alias("pixel_y"),
            ]
        )


    def create_dataset_directory(self):
        """
        Creates the dataset directory if it doesn't exist.
        """
        dataset_path = os.path.join(self.dataset_root_path, self.dataset_name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            os.makedirs(os.path.join(dataset_path, "raw"))
            os.makedirs(os.path.join(dataset_path, "preprocessed"))
            os.makedirs(os.path.join(dataset_path, "events"))
        print(f"Dataset directory created at {dataset_path}")


    def create_raw_dataset(self):
        """
        Saves the gaze data as CSV files in the raw dataset directory.
        Creates a separate CSV file for each subject, stimulus, and screen.
        """
        raw_dir = os.path.join(self.dataset_root_path, self.dataset_name, "raw")
        for stimulus_id in self.gaze_df.frame["stimulus_id"].unique():
            stimulus_df = self.gaze_df.frame.filter((pl.col("stimulus_id") == stimulus_id))
            for screen in stimulus_df["screen"].unique():
                screen_df = stimulus_df.filter((pl.col("screen") == screen))
                screen_df = screen_df.select([
                    pl.col("time"),
                    pl.col("screen"),
                    pl.col("pixel_x"),
                    pl.col("pixel_y"),
                    pl.col("pupil"),
                ])
                print(f"Saving {self.subject_id} - {stimulus_id} - {screen}")
                screen_df.write_csv(f"{raw_dir}/S{self.subject_id}-{stimulus_id}-{screen}.csv")
        print(f"Raw dataset created at {raw_dir}")


    def create_dataset_object(self):
        """
        Creates a dataset object for the eye tracking data.
        """
        experiment = pm.gaze.Experiment(
            screen_width_px=1920,
            screen_height_px=1080,
            screen_width_cm=53.5,
            screen_height_cm=31.3,
            distance_cm=60,
            origin="upper left",
            sampling_rate=500,
        )

        dataset_definition = pm.DatasetDefinition(
            name=self.dataset_name,
            has_files={"gaze": True, "precomputed_events": False, "precomputed_reading_measures": False},
            experiment=experiment,
            filename_format={"gaze": r"S{subject_id}-{stimulus_id}-{screen}.csv"},
            filename_format_schema_overrides={"gaze": {"subject_id": int, "stimulus_id": str, "screen": str}},
            custom_read_kwargs={"gaze": {"separator": ","}},
            time_column="time",
            time_unit="ms",
            pixel_columns=["pixel_x", "pixel_y"],
            trial_columns=["subject_id", "stimulus_id", "screen"],
        )

        dataset_paths = pm.DatasetPaths(
            root=self.dataset_root_path,
            raw="raw",
            preprocessed="preprocessed",
            events="events",
        )

        self.dataset = pm.Dataset(
            definition=dataset_definition,
            path=dataset_paths,
        )


    def process_dataset(self):
        """
        Processes the dataset.
        """
        print("Processing dataset...")
        self.dataset.load()
        self.dataset.pix2deg()
        self.dataset.pos2vel()
        self.dataset.detect("ivt")
        self.dataset.compute_event_properties(("location", dict(position_column="pixel")))
        self.dataset.save_preprocessed()
        self.dataset.save_events()
        self.dataset.save()
        print("Dataset processed and saved.")


    def get_order_version(self):
        """
        Extracts the order version from the general log file.
        """
        with open(self.general_logfile_path, 'r') as file:
            lines = file.readlines()
            # Get the 7th line (with the order version)
            order_version_line = lines[6].split()[2]
            order_version = order_version_line.split('_')[-1]
            self.question_images_version = order_version


    def map_fixations_to_words(self):
        """
        Maps the fixations to words in the eye tracking data.
        """
        print("Mapping fixations to words...")
        self.dataset.load()
        self.dataset.load_event_files()
        
        stimulus_files = glob.glob(os.path.join(self.aoi_stimuli_path, '*'))
        for event_df in self.dataset.events:
            screen = event_df.frame["screen"].unique()[0]
            for stimulus_file in stimulus_files:
                stimulus_file_name = os.path.basename(stimulus_file).split('_')[1]
                if stimulus_file_name.lower() in screen.lower():
                    stimulus_df = pd.read_csv(stimulus_file)
                    page_question = screen.split('_')[-2]
                    number = screen.split('_')[-1]
                    page = f"{page_question}_{number}"
                    stimulus_df = stimulus_df[stimulus_df['page'] == page]
                    if page_question == 'question':
                        stimulus_df = stimulus_df[stimulus_df['question_image_version'] == f'question_images_version_{self.question_images_version}']
                    stimulus_df = pl.from_pandas(stimulus_df)
                    text_stimulus = pm.stimulus.TextStimulus(aois=stimulus_df, aoi_column='word', start_x_column='top_left_x', start_y_column='top_left_y', width_column='width', height_column='height', page_column='page')
                    print(f"Mapping event DataFrame to TextStimulus for screen: {screen}")
                    event_df.map_to_aois(text_stimulus)
                    # Remove all rows with None values in the 'word' column
                    event_df.frame = event_df.frame.filter(pl.col("word").is_not_null())
                    break
            else:
                print(f"No matching stimulus file found for screen: {screen}")
        print("Fixations mapped to words.")


    def get_words_dictionary(self):
        """
        Returns a dictionary of fixations for each word in the dataset.
        """
        # self.dataset = self.dataset.load(events=True, preprocessed=True)
        words_dict = dict()

        for i, event_df in enumerate(self.dataset.events):
            if "word" not in event_df.frame.columns:
                print(f"Skipping event DataFrame {i}.")
                continue
            print(f"Processing event DataFrame {i}...")
            for row in event_df.frame.iter_rows(named=True):
                screen = row['screen']
                subject_id = row['subject_id']
                word_idx = row['word_idx']
                
                # Initialize nested dictionaries if keys don't exist
                stimulus_key = screen.lower()
                # Remove stimulus_ from the stimulus_key
                stimulus_key = stimulus_key.replace('stimulus_', '')
                if stimulus_key not in words_dict:
                    words_dict[stimulus_key] = dict()
                if subject_id not in words_dict[stimulus_key]:
                    words_dict[stimulus_key][subject_id] = dict()
                if word_idx not in words_dict[stimulus_key][subject_id]:
                    words_dict[stimulus_key][subject_id][word_idx] = dict()
                    words_dict[stimulus_key][subject_id][word_idx]['word'] = row['word']
                    words_dict[stimulus_key][subject_id][word_idx]['fixations'] = WordFixations()
                
                # Append the fixation duration
                words_dict[stimulus_key][subject_id][word_idx]['fixations'].fixations.append(row['duration'])
                
        # Calculate the TRT for each word
        for stimulus_key in words_dict:
            for subject_id in words_dict[stimulus_key]:
                for word_idx in words_dict[stimulus_key][subject_id]:
                    fixations = words_dict[stimulus_key][subject_id][word_idx]['fixations']
                    fixations.TRT = sum(fixations.fixations)

        return words_dict


    def run(self):
        """
        Runs the entire process of extracting word fixations from the eye tracking data.
        Returns the fixations dictionary.
        """
        print("Starting the extraction of word fixations...")
        self.get_gaze_df_from_asc_file()
        self.set_stimulus_ids()
        self.split_pixels()
        self.create_dataset_directory()
        self.create_raw_dataset()
        self.create_dataset_object()
        self.process_dataset()
        self.get_order_version()
        self.map_fixations_to_words()
        words_dict = self.get_words_dictionary()
        print("Extraction of word fixations completed.")
        return words_dict
    