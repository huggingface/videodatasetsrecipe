import os
import json
import shutil
from pathlib import Path
import subprocess
import argparse
import tempfile
from typing import Optional, Dict, List, Tuple
import sys


def read_and_validate_data(metadata_dir: str, video_dir: str) -> List[Tuple[str, dict, str]]:
    """
    Read metadata and video files, matching them by filename.
    Returns list of tuples: (filename_base, metadata_content, video_path)
    """
    paired_data = []
    missing_pairs = []
    
    # Get all json files
    json_files = list(Path(metadata_dir).glob('*.json'))
    
    print(f"Found {len(json_files)} JSON files in metadata directory")
    
    for json_path in json_files:
        base_name = json_path.stem
        video_path = Path(video_dir) / f"{base_name}.mp4"
        
        # Check if corresponding video exists
        if not video_path.exists():
            missing_pairs.append(base_name)
            continue
            
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error reading metadata file {json_path}: {str(e)}")
            continue
            
        paired_data.append((base_name, metadata, str(video_path)))
    
    # Report statistics
    print(f"\nDataset Statistics:")
    print(f"Total paired files: {len(paired_data)}")
    if missing_pairs:
        print(f"Missing video files for: {', '.join(missing_pairs)}")
    
    return paired_data

def create_video_dataset(
    source_video_dir: str,
    source_metadata_dir: str,
    output_base_dir: str,
    examples_per_folder: int = 9500,
    max_total_examples: Optional[int] = None,
) -> int:
    """
    Create a video dataset organized in folders with accompanying metadata.
    
    Args:
        source_video_dir: Directory containing source video files
        source_metadata_dir: Directory containing metadata JSON files
        output_base_dir: Base directory for the organized dataset
        examples_per_folder: Maximum number of examples per folder
        max_total_examples: Maximum total examples to process (None for all)
    
    Returns:
        int: Total processed examples
    """
    
    # Create base output directory
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    
    # Read and validate data
    paired_data = read_and_validate_data(source_metadata_dir, source_video_dir)
    if not paired_data:
        raise ValueError("No valid paired files found")
    
    current_folder = 0
    processed_examples = 0
    current_folder_examples = 0

    # Create the first folder
    current_folder_path = os.path.join(output_base_dir, f"{current_folder:04d}")
    Path(current_folder_path).mkdir(exist_ok=True)
    
    # Initialize metadata file for the current folder
    metadata_file = open(os.path.join(current_folder_path, "metadata.jsonl"), "w")
    
    for base_name, metadata, video_path in paired_data:
        # Break if we've reached the maximum examples
        if max_total_examples and processed_examples >= max_total_examples:
            break
            
        # If we've reached the examples per folder limit, create a new folder
        if current_folder_examples >= examples_per_folder:
            metadata_file.close()
            current_folder += 1
            current_folder_examples = 0
            current_folder_path = os.path.join(output_base_dir, f"{current_folder:04d}")
            Path(current_folder_path).mkdir(exist_ok=True)
            metadata_file = open(os.path.join(current_folder_path, "metadata.jsonl"), "w")
        
        # Copy video to new location
        video_filename = f"{base_name}.mp4"
        destination_path = os.path.join(current_folder_path, video_filename)
        shutil.copy2(video_path, destination_path)
        
        # Create metadata entry (including original filename and all metadata)
        metadata_entry = {
            "file_name": video_filename,
            **metadata  # Include all fields from original metadata
        }
        
        # Write metadata entry
        metadata_file.write(json.dumps(metadata_entry) + "\n")
        
        current_folder_examples += 1
        processed_examples += 1
        
        if processed_examples % 100 == 0:  # Progress update every 100 examples
            print(f"Processed {processed_examples} examples")
    
    # Close the last metadata file
    metadata_file.close()
    
    print(f"Dataset creation complete. Processed {processed_examples} examples across {current_folder + 1} folders")
    return processed_examples

def upload_to_huggingface(dataset_path: str, hf_dataset_name: str) -> None:
    """Upload the dataset to Hugging Face"""
    try:
        cmd = [
            "huggingface-cli", 
            "upload-large-folder", 
            hf_dataset_name, 
            dataset_path,
            "--repo-type=dataset"
        ]
        subprocess.run(cmd, check=True)
        print(f"Successfully uploaded dataset to {hf_dataset_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error uploading to Hugging Face: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Create and upload a video dataset to Hugging Face")
    parser.add_argument("--video-dir", required=True, help="Directory containing source video files")
    parser.add_argument("--metadata-dir", required=True, help="Directory containing metadata JSON files")
    parser.add_argument("--hf-dataset-name", required=True, help="Hugging Face dataset name (e.g., 'username/dataset-name')")
    parser.add_argument("--examples-per-folder", type=int, default=9500, 
                       help="Maximum examples per folder (max 10000)")
    parser.add_argument("--max-examples", type=int, help="Maximum total examples to process")
    parser.add_argument("--temp-dir", help="Temporary directory for dataset creation (default: system temp directory)")
    
    args = parser.parse_args()
    
    # Validate examples_per_folder
    if args.examples_per_folder > 10000:
        print("Error: examples-per-folder cannot exceed 10000")
        sys.exit(1)
    elif args.examples_per_folder <= 0:
        print("Error: examples-per-folder must be greater than 0")
        sys.exit(1)
    
    # Use provided temp directory or create one
    temp_base_dir = args.temp_dir or tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_base_dir}")
    
    try:
        # Create dataset
        processed_examples = create_video_dataset(
            source_video_dir=args.video_dir,
            source_metadata_dir=args.metadata_dir,
            output_base_dir=temp_base_dir,
            examples_per_folder=args.examples_per_folder,
            max_total_examples=args.max_examples
        )
        
        if processed_examples > 0:
            # Upload to Hugging Face
            upload_to_huggingface(temp_base_dir, args.hf_dataset_name)
        else:
            print("No examples were processed. Aborting upload.")
            sys.exit(1)
            
    finally:
        if not args.temp_dir:  # Only remove if we created the temp directory
            shutil.rmtree(temp_base_dir, ignore_errors=True)

if __name__ == "__main__":
    main()