#%%
from ieeg.auth import Session
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from ieeg_metadata import IEEGmetadata
from manualvalidation_data import ManualValidation
from pathlib import Path
from IPython import embed
#%%
class IEEGmetadataValidated(IEEGmetadata, ManualValidation):
    """
    A class that inherits from both IEEGmetadata and ManualValidation.
    """

    def __init__(self):
        """
        Initialize both parent classes
        Set up IEEG session
        """
        super().__init__()
        
    # Main entry point
    def process_subject_data(self, subject_id: str) -> None:
        """
        Process all data for a single subject.
        
        Args:
            subject_id (str): Subject ID in format 'sub-RID0222'
        """
        # Get all required data
        ieeg_data_df = self.get_redcap_data(subjects=[subject_id])
        ieeg_data_df = self.expand_ieeg_days_rows(ieeg_data_df)
        
        start_times_df = self.get_actual_start_times(record_id=[subject_id])
        seizure_times_df = self.get_seizure_times(record_id=[subject_id])
        seizure_times_manual = self.process_seizure_annotations(seizure_times_df)

        # Process each day's data
        for idx, (record_id, data) in enumerate(ieeg_data_df.iterrows(), start=2):
            self._process_single_session(
                record_id=record_id,
                data=data,
                start_times_df=start_times_df,
                seizure_times_manual=seizure_times_manual,
                idx=idx
            )

    def process_seizure_annotations(self, seizure_times: pd.DataFrame) -> pd.DataFrame:
        """
        Convert seizure times from manual validation into a standardized annotation format.
        
        Args:
            seizure_times (pd.DataFrame): DataFrame containing seizure timing information
                Expected columns: ['source', 'start', 'end']
        
        Returns:
            pd.DataFrame: Annotations DataFrame with columns:
                - layer: always 'manual_validation'
                - annotator: source of the seizure annotation
                - description: always 'seizure'
                - type: always 'seizure'
                - start_time_usec: seizure start time in microseconds
                - end_time_usec: seizure end time in microseconds
        """
        # Create annotations DataFrame from seizure times
        annotations = {
            'layer': ['manual_validation'] * len(seizure_times),
            'annotator': seizure_times['source'].tolist(),
            'description': ['seizure'] * len(seizure_times),
            'type': ['seizure'] * len(seizure_times),
            # Convert seconds to microseconds (1e6)
            'start_time_usec': (seizure_times['start'] * 1e6).astype(int).tolist(),
            'end_time_usec': (seizure_times['end'] * 1e6).astype(int).tolist()
        }
        
        # Convert to DataFrame and sort by start time
        annotations_manual = pd.DataFrame(annotations)
        annotations_manual = annotations_manual.sort_values(by='start_time_usec')
        
        return annotations_manual
    
    def _process_single_session(self, record_id: str, data: pd.Series, 
                          start_times_df: pd.DataFrame,
                          seizure_times_manual: pd.DataFrame,
                          idx: int) -> None:
        """
        Process data for a single session of recording.
        
        Args:
            record_id (str): Record ID
            data (pd.Series): Data for this record
            start_times_df (pd.DataFrame): DataFrame with start times
            seizure_times_manual (pd.DataFrame): Processed seizure annotations
            idx (int): Index for accessing start times
        """
        dataset_name = data['ieegportalsubjno']
        
        # Get base metadata
        channels_df, annotations_df, metadata_dict, clips_df = self.save_metadata(
            record_id=record_id, dataset_name=dataset_name)
        
        # Process annotations
        annotations_df_validated = pd.concat(
            [annotations_df, seizure_times_manual], 
            ignore_index=True
        )
        
        # Process clips
        clips_df_validated = self._ieeg_clips(annotations_df_validated, metadata_dict)

        # Add timestamp information if available
        if not start_times_df.empty:
            start_time = start_times_df.iloc[:, idx]
            start_time_value = start_time.values[0] if not pd.isna(start_time.values[0]) else None
            metadata_dict['actual_start_time'] = start_time_value
            clips_df_validated = self.timestamp_clips(clips_df_validated, metadata_dict)
        
        # Save all validated data
        self.save_validated_metadata(
            record_id=record_id,
            dataset_name=dataset_name,
            annotations_df_validated=annotations_df_validated,
            clips_df_validated=clips_df_validated,
            metadata_dict=metadata_dict
        )

    def timestamp_clips(self, clips_df: pd.DataFrame, metadata_dict: Dict) -> pd.DataFrame:
        """
        Index clips DataFrame with timestamps and night/day information at 1-minute intervals.
        
        Args:
            clips_df (pd.DataFrame): DataFrame containing clips data
            metadata_dict (Dict): Dictionary containing metadata

        Returns:
            pd.DataFrame: Clips DataFrame with timestamp index and night/day information
        """
        
        # Convert actual_start_time string to datetime
        actual_start_time = pd.to_datetime(metadata_dict['actual_start_time'])
        
        # Create timestamp index
        timestamps = []
        is_night = []
        for t in clips_df.start_time_usec:
            t = t/1e6 # Convert to seconds
            days_elapsed = int(t // (24 * 3600))
            current_time = actual_start_time + pd.Timedelta(seconds=int(t))
            
            # Format for index
            day_str = f"Day {days_elapsed + 1} {current_time.strftime('%H:%M:%S')}"
            timestamps.append(day_str)
            
            # Check if current time is during night hours (19:00-08:00)
            hour = current_time.hour
            is_night.append(hour >= 19 or hour < 8)
        
        # Reindex the DataFrame to 1-minute intervals
        clips_df.index = pd.Index(timestamps, name='timestamp')
        clips_df['is_night'] = is_night
        
        return clips_df
    
    def save_validated_metadata(self, record_id, dataset_name, 
                      annotations_df_validated=None, 
                      clips_df_validated=None,
                      metadata_dict=None,
                      path_to_save: Path = Path(__file__).parent.parent / 'data'):
        """Save the metadata to a file.
        
        Args:
            record_id: The ID of the record
            dataset_name: Name of the dataset
            path_to_save: Path where metadata will be saved. Defaults to 'data'
        """
        Path(path_to_save / record_id / dataset_name).mkdir(parents=True, exist_ok=True)

        annotations_df_validated.to_csv(Path(path_to_save) / record_id / dataset_name / 'annotations.csv', index=False)
        clips_df_validated.to_csv(Path(path_to_save) / record_id / dataset_name / 'clips.csv')
        with open(Path(path_to_save) / record_id / dataset_name / 'metadata.txt', 'w') as f:
            for key, value in metadata_dict.items():
                f.write(f"{key}: {value}\n")

# %%
if __name__ == '__main__':
    
    subjects_to_find = ["sub-RID0031"]
    
    ieeg = IEEGmetadataValidated()
    
    for subject in subjects_to_find:
        print(f'Processing {subject}')
        try:
            ieeg.process_subject_data(subject)
        except Exception as e:
            print(f'Error processing {subject}: {e}')

# %%

