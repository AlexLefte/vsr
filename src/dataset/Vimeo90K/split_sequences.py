import os
import shutil
import os.path as osp
import argparse

# Move files function
def move_files(txt_file, destination_folder):
    with open(txt_file, 'r') as f:
        for line in f:
            sequence = line.strip()  # Read the sequence (e.g. 00001/0001)
            sequence_folder = os.path.join(sequence_path, sequence)
            
            if os.path.isdir(sequence_folder):  # Check seq folder
                # Move to destination
                shutil.move(sequence_folder, os.path.join(destination_folder, sequence))
                print(f"Moved {sequence_folder} to {destination_folder}")
            else:
                print(f"Sequence {sequence_folder} doesn't exist!")
                

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to sequences directory.')
    args = parser.parse_args()

    # Paths
    source_folder = args.path
    sequence_path = osp.join(source_folder, 'sequences')
    train_folder = osp.join(source_folder, 'train')
    val_folder = osp.join(source_folder, 'test')

    # Create destination folders
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Train/test splits 
    train_txt = osp.join(source_folder, 'sep_trainlist.txt')
    test_txt = osp.join(source_folder, 'sep_testlist.txt')

    # Move files
    move_files(train_txt, train_folder)
    move_files(test_txt, val_folder)