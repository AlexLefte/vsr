import os
import subprocess
import argparse

# Function to extract frames from video files
def extract_frames(input_folder, output_folder, fps):
    # Make sure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all .mp4 files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            # Get the clip ID from the filename (e.g., id_1280x720_24fps.mp4 -> id1)
            clip_id = filename.split('_')[0]

            # Define the output folder for the current clip
            clip_folder = os.path.join(output_folder, f'clip_{clip_id}')
            os.makedirs(clip_folder, exist_ok=True)

            # Define the output file path for frames
            output_path = os.path.join(clip_folder, 'images_%04d.png')

            # Use FFmpeg to extract frames (image sequence) from the video
            command = [
                'ffmpeg',
                '-i', os.path.join(input_folder, filename),
                '-vf', f'fps={fps}',  # Extract frames at the specified fps
                '-q:v', '1',  # Highest quality for the PNG output
                output_path
            ]

            # Run the FFmpeg command
            subprocess.run(command, check=True)

            print(f"Frames extracted for {filename} into {clip_folder}")

    print("Extraction complete.")


def main():
    parser = argparse.ArgumentParser(description='Extract frames from .mp4 videos and save them as PNG images.')
    parser.add_argument('-i', '--input', type=str, help='The path to the folder containing .mp4 files')
    parser.add_argument('-o', '--output', type=str, help='The base output folder to save extracted frames')
    parser.add_argument('-f', '--fps', type=int, default=24, help='Frames per second to extract (default: 24)')
    
    # Parse the arguments
    args = parser.parse_args()

    # Call the function to extract frames
    extract_frames(args.input, args.output, args.fps)

# Run the script
if __name__ == '__main__':
    main()
