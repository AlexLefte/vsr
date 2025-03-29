import os
import subprocess
import argparse

# Function to extract frames from video files
def extract_frames(input_folder, output_folder, val):
    # Make sure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all .mp4 files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):           
            # Process metadata differently
            if val:
                # Get the clip ID from the filename (e.g., id_320x180_24fps_qp17.mp4)
                clip_id, res, fps, qp = filename.split('_')
                fps = int(fps[:2])
                qp = int(qp[2:4])
            else:
                # Get the clip ID from the filename (e.g., id_1280x720_24fps.mp4)
                clip_id, res, fps = filename.split('_')
                fps = int(fps[:2])

            # Define the output folder for the current clip
            if val:
                # Get qp path
                qp_folder = os.path.join(output_folder, f'qp_{qp}')
                os.makedirs(qp_folder, exist_ok=True)

                # Get the clip path
                clip_folder = os.path.join(qp_folder, f'clip_{clip_id}')
            else:
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
    parser.add_argument('-v', '--val', action='store_true', help="Whether to process validation sets (different QP)")
    
    # Parse the arguments
    args = parser.parse_args()

    # Call the function to extract frames
    extract_frames(args.input, args.output, args.val)

# Run the script
if __name__ == '__main__':
    main()
