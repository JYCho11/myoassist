"""
H5 to NPZ Converter for Reference Motion Data
==============================================
This script converts OpenSim IK data from H5 format to NPZ format
compatible with the myoassist framework for imitation learning.

Usage:
    python h5_npz_convert.py --input <h5_file> --output <npz_file> --level <level_name>
    
Example:
    python h5_npz_convert.py --input "C:/workspace/opensim data/S004.h5" --output "output.npz" --level level_08mps

Expected H5 structure:
    - root
      - level_08mps (or other level)
        - ik_data
          - pelvis_tx (time series data)
          - pelvis_tz
          - pelvis_ty
          - ... (other joint angles without 'q_' prefix)

Output NPZ structure (matching short_reference_gait.npz):
    - metadata: dict with trial information
    - series_data: dict with 'q_' prefixed joint angles and 'dq_' velocities
"""

import h5py
import numpy as np
import argparse
import os
from pathlib import Path


class H5toNPZConverter:
    """Convert H5 OpenSim IK data to NPZ reference motion format"""
    
    def __init__(self, h5_path, subject_name="S004", level_name="level_08mps", trial_name="trial_01"):
        """
        Initialize converter
        
        Args:
            h5_path: Path to H5 file
            subject_name: Subject name in H5 file (e.g., 'S004')
            level_name: Name of the level group in H5 file (e.g., 'level_08mps')
            trial_name: Name of the trial (e.g., 'trial_01')
        """
        self.h5_path = h5_path
        self.subject_name = subject_name
        self.level_name = level_name
        self.trial_name = trial_name
        
        # Lower body joints - available in H5 file
        self.lower_body_joints = [
            'pelvis_tx', 'pelvis_tz', 'pelvis_ty',
            'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
            'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
            'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
            'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l',
            'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
        ]
        
        # Upper body joints - NOT in H5 file, will be filled with zeros
        self.upper_body_joints = [
            'arm_flex_r', 'arm_add_r', 'arm_rot_r',
            'elbow_flex_r', 'pro_sup_r', 'wrist_flex_r', 'wrist_dev_r',
            'arm_flex_l', 'arm_add_l', 'arm_rot_l',
            'elbow_flex_l', 'pro_sup_l', 'wrist_flex_l', 'wrist_dev_l'
        ]
        
        # All joint names (for output)
        self.all_joint_names = self.lower_body_joints + self.upper_body_joints
    
    def load_h5_data(self):
        """
        Load data from H5 file
        Structure: subject_name/level_name/trial_name/MoCap/ik_data/
        
        Returns:
            dict: Dictionary with joint time series data (without 'q_' prefix)
                  Includes both loaded data and zeros for missing upper body joints
        """
        print(f"Loading H5 file: {self.h5_path}")
        print(f"  Subject: {self.subject_name}")
        print(f"  Level: {self.level_name}")
        print(f"  Trial: {self.trial_name}")
        
        try:
            with h5py.File(self.h5_path, 'r') as f:
                # Navigate to the IK data: subject/level/trial/MoCap/ik_data
                if self.subject_name not in f:
                    available_subjects = list(f.keys())
                    raise ValueError(f"Subject '{self.subject_name}' not found. Available subjects: {available_subjects}")
                
                subject_group = f[self.subject_name]
                
                if self.level_name not in subject_group:
                    available_levels = list(subject_group.keys())
                    raise ValueError(f"Level '{self.level_name}' not found. Available levels: {available_levels}")
                
                level_group = subject_group[self.level_name]
                
                if self.trial_name not in level_group:
                    available_trials = list(level_group.keys())
                    raise ValueError(f"Trial '{self.trial_name}' not found. Available trials: {available_trials}")
                
                trial_group = level_group[self.trial_name]
                
                if 'MoCap' not in trial_group:
                    available_groups = list(trial_group.keys())
                    raise ValueError(f"MoCap not found in trial. Available groups: {available_groups}")
                
                mocap_group = trial_group['MoCap']
                
                if 'ik_data' not in mocap_group:
                    available_groups = list(mocap_group.keys())
                    raise ValueError(f"ik_data not found in MoCap. Available groups: {available_groups}")
                
                ik_data = mocap_group['ik_data']
                
                # Extract lower body joint data from H5
                data_dict = {}
                available_joints = list(ik_data.keys())
                print(f"\nAvailable joints in H5: {available_joints}")
                print(f"\nLoading lower body joints...")
                
                # Define which joints need degree to radian conversion
                # Position coordinates (pelvis_tx, ty, tz) stay in meters, angles convert to radians
                angular_joints = [
                    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                    'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
                    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
                    'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l',
                    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
                ]
                
                num_frames = None
                for joint_name in self.lower_body_joints:
                    if joint_name in ik_data:
                        data = np.array(ik_data[joint_name])
                        
                        # Convert angles from degrees to radians
                        if joint_name in angular_joints:
                            data = np.deg2rad(data)
                            print(f"  ✓ Loaded {joint_name}: shape {data.shape} (converted deg→rad)")
                        else:
                            print(f"  ✓ Loaded {joint_name}: shape {data.shape} (meters, no conversion)")
                        
                        # FIX: Invert sign for left hip adduction (H5 data has opposite sign convention)
                        if joint_name == 'hip_adduction_l':
                            data = -data
                            print(f"  ⚠ Fixed sign for {joint_name} (H5 convention correction)")
                        
                        data_dict[joint_name] = data
                        if num_frames is None:
                            num_frames = len(data_dict[joint_name])
                    else:
                        print(f"  ✗ Warning: Joint '{joint_name}' not found in H5 file")
                
                if not data_dict:
                    raise ValueError("No matching joint data found in H5 file")
                
                # ========== MODIFIED: Copy upper body data from short_reference_gait.npz ==========
                # Load upper body data from the reference file
                print(f"\nLoading upper body joint data from short_reference_gait.npz...")
                try:
                    ref_file = np.load('rl_train/reference_data/short_reference_gait.npz', allow_pickle=True)
                    ref_series = ref_file['series_data'].item()
                    
                    # Interpolate reference data to match num_frames
                    ref_num_frames = len(ref_series['q_pelvis_tx'])
                    
                    for joint_name in self.upper_body_joints:
                        q_key = f"q_{joint_name}"
                        if q_key in ref_series:
                            # Interpolate from reference data to match current data length
                            ref_data = np.array(ref_series[q_key])
                            x_ref = np.linspace(0, 1, len(ref_data))
                            x_new = np.linspace(0, 1, num_frames)
                            data_dict[joint_name] = np.interp(x_new, x_ref, ref_data)
                            print(f"  ✓ Copied {joint_name}: shape {data_dict[joint_name].shape} (from reference)")
                        else:
                            # Fallback: use zeros if not found
                            data_dict[joint_name] = np.zeros(num_frames)
                            print(f"  ⚠ Created {joint_name}: shape {data_dict[joint_name].shape} (zeros, not found in reference)")
                    
                    ref_file.close()
                    print(f"  Successfully copied upper body data from reference file")
                    
                except Exception as e:
                    print(f"  ⚠ Warning: Could not load reference file: {e}")
                    print(f"  Falling back to zeros for upper body joints...")
                    for joint_name in self.upper_body_joints:
                        data_dict[joint_name] = np.zeros(num_frames)
                        print(f"  ✓ Created {joint_name}: shape {data_dict[joint_name].shape} (zeros)")
                
                # ========== ORIGINAL CODE (주석처리) ==========
                # # Fill upper body joints with zeros
                # print(f"\nFilling upper body joints with zeros (num_frames={num_frames})...")
                # for joint_name in self.upper_body_joints:
                #     data_dict[joint_name] = np.zeros(num_frames)
                #     print(f"  ✓ Created {joint_name}: shape {data_dict[joint_name].shape} (zeros)")
                # ========== END ORIGINAL CODE ==========
                
                return data_dict
                
        except Exception as e:
            print(f"Error loading H5 file: {e}")
            raise
    
    def find_symmetric_frame(self, data_dict):
        """
        Find the most symmetric frame in gait cycle for treadmill data
        
        Args:
            data_dict: Dictionary with joint time series
            
        Returns:
            int: Index of most symmetric frame
        """
        print("\n" + "=" * 80)
        print("FINDING SYMMETRIC FRAME FOR TREADMILL DATA")
        print("=" * 80)
        
        # Extract key joints
        hip_flex_r = data_dict['hip_flexion_r']
        hip_flex_l = data_dict['hip_flexion_l']
        knee_r = data_dict['knee_angle_r']
        knee_l = data_dict['knee_angle_l']
        ankle_r = data_dict['ankle_angle_r']
        ankle_l = data_dict['ankle_angle_l']
        
        # Compute symmetry score (lower = more symmetric)
        symmetry_score = (np.abs(hip_flex_r - hip_flex_l) + 
                         np.abs(knee_r - knee_l) + 
                         np.abs(ankle_r - ankle_l))
        
        # Find minimum (most symmetric)
        most_symmetric_idx = np.argmin(symmetry_score)
        
        print(f"Total frames: {len(hip_flex_r)}")
        print(f"Most symmetric frame: {most_symmetric_idx}")
        print(f"  Symmetry score: {symmetry_score[most_symmetric_idx]:.4f}")
        print(f"  Hip flex R/L: {hip_flex_r[most_symmetric_idx]:.4f} / {hip_flex_l[most_symmetric_idx]:.4f}")
        print(f"  Knee R/L: {knee_r[most_symmetric_idx]:.4f} / {knee_l[most_symmetric_idx]:.4f}")
        print(f"  Ankle R/L: {ankle_r[most_symmetric_idx]:.4f} / {ankle_l[most_symmetric_idx]:.4f}")
        
        return most_symmetric_idx
    
    def reorder_from_symmetric_frame(self, data_dict, start_idx):
        """
        Reorder all time series to start from symmetric frame
        
        Args:
            data_dict: Dictionary with joint time series
            start_idx: Index to use as new start
            
        Returns:
            dict: Reordered data dictionary
        """
        print(f"\nReordering data to start from frame {start_idx}...")
        
        reordered = {}
        for key, data in data_dict.items():
            # Roll array so start_idx becomes index 0
            reordered[key] = np.roll(data, -start_idx)
            
        print(f"  ✓ Reordered {len(reordered)} joints")
        return reordered
    
    def convert_pelvis_tx_to_cumulative(self, data_dict, dt=0.01):
        """
        Convert pelvis_tx from treadmill position to cumulative forward distance
        
        For treadmill data, pelvis_tx oscillates around a fixed position.
        We convert it to cumulative forward distance by integrating velocity.
        
        Args:
            data_dict: Dictionary with joint time series
            dt: time step (0.01s = 100Hz)
            
        Returns:
            dict: Modified data dictionary with cumulative pelvis_tx
        """
        print("\n" + "=" * 80)
        print("CONVERTING PELVIS_TX FROM TREADMILL TO CUMULATIVE DISTANCE")
        print("=" * 80)
        
        pelvis_tx = data_dict['pelvis_tx'].copy()
        
        print(f"Original pelvis_tx range: [{np.min(pelvis_tx):.4f}, {np.max(pelvis_tx):.4f}] m")
        print(f"Original std dev: {np.std(pelvis_tx):.6f} m")
        
        # Compute velocity from position changes
        velocity = np.gradient(pelvis_tx) / dt
        
        # Use mean treadmill speed as reference
        mean_speed = 0.8  # 0.8 m/s from level_08mps
        print(f"Reference treadmill speed: {mean_speed} m/s")
        
        # Create cumulative distance: start at 0, accumulate with mean speed
        cumulative_distance = np.arange(len(pelvis_tx)) * mean_speed * dt
        
        # Add oscillation from original data (deviation from mean position)
        mean_position = np.mean(pelvis_tx)
        oscillation = pelvis_tx - mean_position
        cumulative_distance += oscillation
        
        # Ensure starting from 0
        cumulative_distance -= cumulative_distance[0]
        
        data_dict['pelvis_tx'] = cumulative_distance
        
        print(f"Converted pelvis_tx range: [{np.min(cumulative_distance):.4f}, {np.max(cumulative_distance):.4f}] m")
        print(f"Total distance traveled: {cumulative_distance[-1]:.2f} m over {len(pelvis_tx)} frames ({len(pelvis_tx)*dt:.1f}s)")
        print(f"  ✓ Pelvis_tx converted to cumulative forward distance")
        
        return data_dict
    
    def compute_velocities(self, positions, dt=0.01):
        """
        Compute velocities from position data using finite differences
        
        Args:
            positions: numpy array of position time series
            dt: time step (default: 0.01s = 100Hz)
            
        Returns:
            numpy array of velocity time series
        """
        # Use central differences for interior points
        velocities = np.zeros_like(positions)
        velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
        
        # Use forward/backward differences for endpoints
        velocities[0] = (positions[1] - positions[0]) / dt
        velocities[-1] = (positions[-1] - positions[-2]) / dt
        
        return velocities
    
    def convert_to_npz_format(self, h5_data, dt=0.01, trial_name="converted_trial", save_as_list=True):
        """
        Convert H5 data to NPZ format
        
        Args:
            h5_data: Dictionary with joint time series (without 'q_' prefix)
            dt: time step for velocity computation
            trial_name: name for metadata
            save_as_list: If True, save as Python list (matching original format)
                         If False, save as numpy array
            
        Returns:
            dict: NPZ-compatible data structure
        """
        print("\nConverting to NPZ format...")
        print(f"  Data format: {'list' if save_as_list else 'numpy array'}")
        
        series_data = {}
        
        # Add position data with 'q_' prefix (in the correct order)
        print("\nAdding position data (q_)...")
        for joint_name in self.all_joint_names:
            q_key = f"q_{joint_name}"
            # Convert to list if needed (matching original format)
            if save_as_list:
                series_data[q_key] = h5_data[joint_name].tolist()
            else:
                series_data[q_key] = h5_data[joint_name]
            print(f"  Added {q_key}: shape {h5_data[joint_name].shape}")
        
        # Compute and add velocity data with 'dq_' prefix (by differentiating q_)
        print("\nComputing velocities (dq_ by differentiation)...")
        for joint_name in self.all_joint_names:
            dq_key = f"dq_{joint_name}"
            velocities = self.compute_velocities(h5_data[joint_name], dt=dt)
            # Convert to list if needed (matching original format)
            if save_as_list:
                series_data[dq_key] = velocities.tolist()
            else:
                series_data[dq_key] = velocities
            print(f"  Computed {dq_key}: shape {velocities.shape}")
        
        # Calculate sampling rate
        sampling_rate = int(1.0 / dt)
        num_frames = len(list(h5_data.values())[0]) if h5_data else 0
        
        # Create metadata (matching original format)
        metadata = {
            'sample_rate': sampling_rate,  # Hz
            'data_length': num_frames,
            # Additional info (commented out to match original format more closely)
            # 'trial_name': trial_name,
            # 'source_file': os.path.basename(self.h5_path),
            # 'subject_name': self.subject_name,
            # 'level_name': self.level_name,
            # 'trial': self.trial_name,
            # 'dt': dt,
            # 'lower_body_joints': self.lower_body_joints,
            # 'upper_body_joints': self.upper_body_joints,
            # 'note': 'Upper body joints are filled with zeros'
        }
        
        return {
            'metadata': metadata,
            'series_data': series_data
        }
    
    def save_npz(self, output_path, npz_data):
        """
        Save data to NPZ file
        
        Args:
            output_path: Path to output NPZ file
            npz_data: Dictionary with 'metadata' and 'series_data'
        """
        print(f"\nSaving to NPZ file: {output_path}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save with allow_pickle=True for metadata dict
        np.savez(
            output_path,
            metadata=npz_data['metadata'],
            series_data=npz_data['series_data']
        )
        
        print(f"Successfully saved NPZ file: {output_path}")
        
        # Verify the saved file
        self.verify_npz(output_path)
    
    def verify_npz(self, npz_path):
        """
        Verify the saved NPZ file
        
        Args:
            npz_path: Path to NPZ file to verify
        """
        print("\nVerifying saved NPZ file...")
        
        try:
            data = np.load(npz_path, allow_pickle=True)
            print(f"  Keys: {list(data.keys())}")
            
            if 'metadata' in data:
                metadata = data['metadata'].item()
                print(f"  Metadata: {metadata}")
            
            if 'series_data' in data:
                series_data = data['series_data'].item()
                print(f"  Series data keys (first 10): {list(series_data.keys())[:10]}")
                
                # Check a sample joint
                sample_key = list(series_data.keys())[0]
                print(f"  Sample data ({sample_key}): shape={series_data[sample_key].shape}, "
                      f"dtype={series_data[sample_key].dtype}")
            
            print("  Verification successful!")
            
        except Exception as e:
            print(f"  Verification failed: {e}")
    
    def convert(self, output_path, dt=0.01, trial_name=None, save_as_list=True, 
                reorder_symmetric=True, convert_treadmill=True):
        """
        Full conversion pipeline
        
        Args:
            output_path: Path to output NPZ file
            dt: time step for velocity computation
            trial_name: name for metadata (default: derived from h5 filename)
            save_as_list: If True, save as Python list (matching original format)
            reorder_symmetric: If True, reorder to start from most symmetric frame
            convert_treadmill: If True, convert pelvis_tx from treadmill to cumulative distance
        """
        if trial_name is None:
            trial_name = Path(self.h5_path).stem
        
        # Load H5 data
        h5_data = self.load_h5_data()
        
        # Process treadmill data
        if reorder_symmetric:
            symmetric_idx = self.find_symmetric_frame(h5_data)
            h5_data = self.reorder_from_symmetric_frame(h5_data, symmetric_idx)
        
        if convert_treadmill:
            h5_data = self.convert_pelvis_tx_to_cumulative(h5_data, dt=dt)
        
        # Convert to NPZ format
        npz_data = self.convert_to_npz_format(h5_data, dt=dt, trial_name=trial_name, save_as_list=save_as_list)
        
        # Save NPZ file
        self.save_npz(output_path, npz_data)
        
        print("\n" + "="*60)
        print("Conversion complete!")
        print("="*60)


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Convert H5 OpenSim IK data to NPZ reference motion format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python h5_npz_convert.py --input "C:/workspace/opensim data/LD/S004.h5" --output reference.npz
  
  # Specify different level name
  python h5_npz_convert.py --input S004.h5 --output reference.npz --level level_10mps
  
  # Specify different subject and trial
  python h5_npz_convert.py --input S004.h5 --output reference.npz --subject S004 --trial trial_01
  
  # Specify different sampling rate
  python h5_npz_convert.py --input S004.h5 --output reference.npz --dt 0.02
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input H5 file path'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output NPZ file path'
    )
    
    parser.add_argument(
        '--subject', '-s',
        default='S004',
        help='Subject name in H5 file (default: S004)'
    )
    
    parser.add_argument(
        '--level', '-l',
        default='level_08mps',
        help='Level name in H5 file (default: level_08mps)'
    )
    
    parser.add_argument(
        '--trial', '-t',
        default='trial_01',
        help='Trial name in H5 file (default: trial_01)'
    )
    
    parser.add_argument(
        '--dt',
        type=float,
        default=0.01,
        help='Time step for velocity computation in seconds (default: 0.01)'
    )
    
    parser.add_argument(
        '--trial-name',
        default=None,
        help='Trial name for metadata (default: derived from input filename)'
    )
    
    parser.add_argument(
        '--format',
        choices=['list', 'array'],
        default='list',
        help='Output format: list (Python list, default) or array (numpy array)'
    )
    
    args = parser.parse_args()
    
    # Create converter and run
    converter = H5toNPZConverter(
        h5_path=args.input,
        subject_name=args.subject,
        level_name=args.level,
        trial_name=args.trial
    )
    
    converter.convert(
        output_path=args.output,
        dt=args.dt,
        trial_name=args.trial_name,
        save_as_list=(args.format == 'list')
    )


if __name__ == "__main__":
    main()
