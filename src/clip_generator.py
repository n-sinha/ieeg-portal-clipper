#%%
import pandas as pd
import numpy as np
from ieeg_metadata_validated import IEEGmetadataValidated
from pathlib import Path
import h5py
from IPython import embed
from loguru import logger
import mne
import edfio

# %%
class ClipGenerator(IEEGmetadataValidated):
    """
    A class that inherits from IEEGmetadataValidated.
    """

    def __init__(self, record_id: str, 
                 data_path = Path(__file__).parent.parent / 'data'):
        """
        Initialize the ClipGenerator.
        """
        super().__init__()
        self.record_id = record_id
        self.data_path = data_path
        
        # Configure loguru logger
        logger.add(
            "clip_generator.log",
            rotation="100 MB",  # Rotate file when it reaches 100MB
            retention="1 week",  # Keep logs for 1 week
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO"
        )

    def find_interictal_clips(self):
        """
        Find the interictal clips.
        """
        dir_path = self.data_path / self.record_id
        for clip_path in dir_path.rglob('*clips.csv'):
            clip = pd.read_csv(clip_path)
            
            # Apply initial interictal conditions
            conditions = ~clip['close_to_event'] & ~clip['is_night']
            is_day_1 = clip['timestamp'].str.contains(r'Day 1\b')
            clips_interictal = clip[conditions & ~is_day_1]
            
            # If no clips found, try processing with annotations
            if clips_interictal.empty:
                clips_interictal = self._remove_redundant_annotations(clip, clip_path)
            
            if not clips_interictal.empty:
                output_path = clip_path.parent / 'clips_interictal.csv'
                clips_interictal.to_csv(output_path, index=False)
            else:
                print(f'No interictal clips found for {self.record_id}')

    def _remove_redundant_annotations(self, clip: pd.DataFrame, clip_path: Path) -> pd.DataFrame:
        """
        Remove redundant annotations from the clips.
        
        Args:
            clip (pd.DataFrame): Original clips dataframe
            clip_path (Path): Path to the clips file
        
        Returns:
            pd.DataFrame: Filtered interictal clips
        """
        annotations_path = clip_path.parent / 'annotations.csv'
        annotations = pd.read_csv(annotations_path)
        annotations_to_remove = r"(?i)(\*?Tech notation: Video/EEG monitoring taking place|\binterictal\b|x)"
        annotations = annotations[~annotations['description'].str.contains(annotations_to_remove, case=False, na=False)]
        
        # Reset clip fields and check overlaps
        clip['has_events'] = False
        clip['events'] = ''
        clip['annotators'] = ''
        clip['layers'] = ''
        clip['close_to_event'] = False
        
        clip_clean = self._check_clip_overlaps(clip, annotations, hours_window=2)

        # Apply conditions again
        conditions = ~clip_clean['close_to_event'] & ~clip_clean['is_night']
        is_day_1 = clip_clean['timestamp'].str.contains(r'Day 1\b')

        clip_clean = clip_clean[conditions & ~is_day_1]
        
        return clip_clean
    
    def mark_interictal_clips(self):
        """
        Get the interictal clips and mark continuous 1-hour segments for extraction.
        
        Returns:
            pd.DataFrame: Interictal clips with marked segments for extraction
        """
        dir_path = self.data_path / self.record_id
        for clip_path in dir_path.rglob('*clips_interictal.csv'):
            # Read the clips
            interictal_clips = pd.read_csv(clip_path)
            
            # Initialize the mark_for_extraction column
            interictal_clips['mark_for_extraction'] = True
           
            # Split the timestamp into day number and time
            interictal_clips[['day_label', 'day_num', 'time']] = interictal_clips['timestamp'].str.split(expand=True)
            
            # Convert time to datetime, keeping track of day number separately
            interictal_clips['time'] = pd.to_datetime(interictal_clips['time'])
            interictal_clips['day_num'] = interictal_clips['day_num'].astype(int)
            
            # Group by day_num instead of day
            for day_num, day_clips in interictal_clips.groupby('day_num'):

                # Sort by time
                day_clips = day_clips.sort_values('time')
                
                # Find continuous segments using time
                time_diff = day_clips['time'].diff()
                new_segment = time_diff > pd.Timedelta(minutes=1)
                segment_id = new_segment.cumsum()
                # find the length of segments
                segment_lengths = day_clips.groupby(segment_id).size()

                # find the longest segment and mark everything except the first 60 minutes as False
                longest_segment = segment_lengths.idxmax()
                
                # Mark all segments that are not the longest segment as False
                not_longest_segment_mask = (segment_id != longest_segment)
                interictal_clips.loc[day_clips[not_longest_segment_mask].index, 'mark_for_extraction'] = False
                
                # For the longest segment, mark everything after 30 minutes as False
                longest_segment_mask = (segment_id == longest_segment)
                segment_indices = day_clips[longest_segment_mask].index
                if len(segment_indices) > 30:
                    interictal_clips.loc[segment_indices[30:], 'mark_for_extraction'] = False
            
            # Remove the final formatting line since 'timestamp' is not a datetime column
            interictal_clips = interictal_clips.drop(columns=['day_label', 'time'])

            self._get_interictal_clips_edf(interictal_clips, clip_path.parent)    

        return interictal_clips
        
    def _get_interictal_clips(self, interictal_clips: pd.DataFrame, clip_path: Path):
        """
        Get the interictal clips and save them to separate H5 files for each day.
        """
        dataset = clip_path.name
        interictal_clips = interictal_clips[interictal_clips['mark_for_extraction']]
        
        # Process each day separately
        for day_num, day_clips in interictal_clips.groupby('day_num'):
            logger.info(f'Processing day {day_num} in {dataset} of {self.record_id}')
            # Create a separate H5 file for each day
            with h5py.File(clip_path / f'interictal_ieeg_day{day_num}.h5', 'w') as f:
                # Use enumerate to get a counter for the clips
                for clip_idx, (index, clip) in enumerate(day_clips.iterrows(), start=1):
                    start_time_usec = clip['start_time_usec']
                    end_time_usec = clip['end_time_usec']
                    
                    ieeg_clip, sampling_rate, channel_labels = self.get_dataset_clips(
                        dataset_name=dataset, 
                        start_time_usec=start_time_usec, 
                        end_time_usec=end_time_usec
                    )
                    
                    clip_num = f'{clip_idx:02d}'
                    # Create dataset directly in the root of the file
                    ieeg_dataset = f.create_dataset(f'clip{clip_num}', data=ieeg_clip)
                    # Add attributes to the dataset
                    ieeg_dataset.attrs['timestamp'] = clip['timestamp']
                    ieeg_dataset.attrs['start_time_usec'] = start_time_usec
                    ieeg_dataset.attrs['end_time_usec'] = end_time_usec
                    ieeg_dataset.attrs['channels_labels'] = channel_labels
                    ieeg_dataset.attrs['sampling_rate'] = sampling_rate

    def _save_as_edf(self, ieeg_clip: pd.DataFrame, sampling_rate: float, 
                     channel_labels: list, output_path: Path, 
                     clip_metadata: dict, day_num: int = None) -> None:
        """
        Convert pandas DataFrame to EDF format and save using edfio.
        
        Args:
            ieeg_clip (pd.DataFrame): EEG data with channels as columns
            sampling_rate (float): Sampling rate in Hz
            channel_labels (list): List of channel names
            output_path (Path): Path where EDF file will be saved
            clip_metadata (dict): Metadata about the clip (timestamp, start_time_usec, etc.)
        """
        # Convert DataFrame to numpy array (channels as columns, samples as rows)
        data = ieeg_clip.values.T  # Transpose to get channels x samples
        
        # Ensure data is in the correct format (float64 for edfio)
        data = data.astype(np.float64)
        
        # Create EdfSignal objects for each channel
        signals = []
        for i, ch_name in enumerate(channel_labels):
            # EDF channel names must be <= 16 characters
            ch_name_short = ch_name[:16] if len(ch_name) > 16 else ch_name
            
            # Calculate physical range for this channel
            physical_min = float(data[i].min())
            physical_max = float(data[i].max())
            
            signals.append(edfio.EdfSignal(
                data=data[i],
                sampling_frequency=sampling_rate,
                label=ch_name_short,
                transducer_type="",
                physical_dimension="uV",  # Microvolts for EEG
                physical_range=(physical_min, physical_max),
                digital_range=(-32768, 32767),  # Standard EDF digital range
                prefiltering=""
            ))
        
        # Create EDF file
        edf_file = edfio.Edf(
            signals=signals,
            patient=edfio.Patient(
                code=f"{self.record_id}",
                sex="X",  # Unknown/not specified
                birthdate=None,
                name="Unknown"
            ),
            recording=edfio.Recording(
                startdate=None,  # Will use current date
                hospital_administration_code="IEEG",
                investigator_technician_code="Unknown",
                equipment_code="IEEG-Portal",
                additional=[f"Day:{day_num}", f"C:{clip_metadata.get('num_clips', 1)}", f"D:{clip_metadata.get('total_duration_minutes', 1):.0f}m"]
            )
        )
        
        # Write the EDF file
        edf_file.write(str(output_path))
        
        logger.info(f'Saved EDF file: {output_path}')

    def _get_interictal_clips_edf(self, interictal_clips: pd.DataFrame, clip_path: Path):
        """
        Get the interictal clips and save them as combined EDF files for each day.
        All clips for a day are combined into one continuous EDF file.
        """
        dataset = clip_path.name
        interictal_clips = interictal_clips[interictal_clips['mark_for_extraction']]
        
        # Process each day separately
        for day_num, day_clips in interictal_clips.groupby('day_num'):
            logger.info(f'Processing day {day_num} in {dataset} of {self.record_id}')
            
            # Create directory for EDF files if it doesn't exist
            edf_dir = clip_path / f'day{day_num}_edf'
            edf_dir.mkdir(exist_ok=True)
            
            # Collect all clips for this day
            all_clips_data = []
            all_channel_labels = None
            sampling_rate = None
            total_duration_usec = 0
            
            # Process each clip and collect data
            for clip_idx, (index, clip) in enumerate(day_clips.iterrows(), start=1):
                start_time_usec = clip['start_time_usec']
                end_time_usec = clip['end_time_usec']
                
                ieeg_clip, clip_sampling_rate, channel_labels = self.get_dataset_clips(
                    dataset_name=dataset, 
                    start_time_usec=start_time_usec, 
                    end_time_usec=end_time_usec
                )
                
                # Store metadata from first clip
                if all_channel_labels is None:
                    all_channel_labels = channel_labels
                    sampling_rate = clip_sampling_rate
                
                # Convert DataFrame to numpy array and transpose (channels x samples)
                clip_data = ieeg_clip.values.T
                all_clips_data.append(clip_data)
                
                # Calculate total duration
                total_duration_usec += (end_time_usec - start_time_usec)
                
                logger.info(f'Collected clip {clip_idx}/{len(day_clips)} for day {day_num}')
            
            # Combine all clips horizontally (concatenate along time axis)
            combined_data = np.concatenate(all_clips_data, axis=1)
            
            # Prepare metadata for the combined file
            first_clip = day_clips.iloc[0]
            last_clip = day_clips.iloc[-1]
            
            combined_metadata = {
                'timestamp': f"Day_{day_num}_combined",
                'start_time_usec': first_clip['start_time_usec'],
                'end_time_usec': last_clip['end_time_usec'],
                'channels_labels': all_channel_labels,
                'sampling_rate': sampling_rate,
                'num_clips': len(day_clips),
                'total_duration_minutes': total_duration_usec / (1e6 * 60)
            }
            
            # Create output filename for combined file
            output_filename = f'day{day_num}_combined.edf'
            output_path = edf_dir / output_filename
            
            # Save combined data as EDF
            self._save_as_edf(
                ieeg_clip=pd.DataFrame(combined_data.T, columns=all_channel_labels),
                sampling_rate=sampling_rate,
                channel_labels=all_channel_labels,
                output_path=output_path,
                clip_metadata=combined_metadata,
                day_num=day_num
            )
            
            logger.info(f'Created combined EDF file: {output_path} with {len(day_clips)} clips ({combined_metadata["total_duration_minutes"]:.1f} minutes)')

# %% 
if __name__ == '__main__':
    
    # subjects_to_find = ['sub-RID0839',
    #         'sub-RID0786',
    #         'sub-RID0646',
    #         'sub-RID0825','sub-RID0596']
    
    subjects_to_find = ['sub-RID0031']
    
    for subject in subjects_to_find:
        try:
            clip_generator = ClipGenerator(record_id=subject)
            logger.info(f"Processing subject: {subject}")
            clip_generator.find_interictal_clips()
            interictal_clips = clip_generator.mark_interictal_clips()
        except Exception as e:
            logger.error(f"Error processing {subject}: {str(e)}")

# %%
