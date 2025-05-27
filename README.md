# 📽️ Video Super-Resolution Framework

This project is **based on [EGVSR](https://github.com/Thmen/EGVSR)** and extends it with modular components for alignment, feature fusion, reconstruction, and adversarial training. It aims to provide a flexible and extensible framework for experimenting with and benchmarking Video Super-Resolution (VSR) models.

---

## 📌 Features

- Modular pipeline with plug-and-play alignment and reconstruction blocks
- Support for multiple alignment strategies:
  - FNet - optical flow-based alignment
  - Cascading Deformable convolution alignment (DCNAlignNet)
  - Pyramidal, Cascading & Deformable (PCD) + Temporal-Spatial Attention (TSA) from EDVR
- Optional adversarial training with spatio-temporal discriminator (TecoGAN-style)
- Benchmark support: Vimeo90K, Vid4, ICME.

---

## 📁 Project Structure
    project_root/
    ├── src/
    │   ├── codes/
    │   │   ├── data/      # custom datasets 
    │   │   ├── metrics/   # metric calculator 
    │   │   ├── models/    # models and network architectures
    │   │   ├── utils/     # Data manipulation   
    │   ├── data_utils/    # Useful scripts to download and preprocess datasets
    │   ├── config/    # Training and validation configuration files
    │   ├── scripts/   # Scripts for plotting and metrics computaion on PNGs
    ├── weights/       # containes our pretrained models + the official 
    └── README.md
---

## 📷 Visual Results

![Visual Results on Vimeo90K](assets/v90k_collage.png)

![Visual Results on Vid4](assets/vid4_collage.png)

![Visual Results on ICME (different levels of QP)](assets/icme_collage.png)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/vsr-project.git
cd vsr-project
```

### 2. Set up the envirnorment
```bash
conda create -n vsr python=3.12
conda activate vsr
pip install -r requirements.txt
```

### 3. Dataset preparation

#### 3.1 Vimeo90K
1. Download the Dataset using the provided script: `src/codes/dataset_utils/Vimeo90K/download_dataset.py`. Make sure to setup the destination file path accordingly.
2. Split into Train/Test Sets
Use the following command to reorganize sequences into `train/` and `test/` directories:
    ```bash
    python scripts/split_vimeo.py -p destination_path
    ```
    This uses the `sep_trainlist.txt` and `sep_testlist.txt` files provided with the dataset to move sequences accordingly.
3. Rename sequences and sub-sequences using:
    ```bash
    python scripts/rename_dataset_folder.py
    ```
    This script flattens the folder structure by merging the first two directory levels into a single folder (e.g., 00001/0001/ → 00001_0001/). It then moves the .png frames into the new folder and removes empty directories. Make sure to set the source folder accordingly!
4. After completion your folder structure should look like this:
    ```bash
    destination_path/_
    ├── train/
    │   ├── 00001_0001/
    │   ├── ...
    ├── test/
    │   ├── 00002_0003/
    │   ├── sequence_subsequence/
    ```

#### 3.2 ICME 
1. Download the `.mp4` videos using the `src/codes/datase_utils/downloader.py`:
    ```bash
    python downloader.py --list-of-files path/to/list.txt --local-path path/to/download/
    ```
    * list.txt: plain text file with one URL per line.

    * Downloads videos into path/to/download/ while preserving the folder structure.
2. Extract frames using `src/codes/datase_utils/extract_frames.py`
    ```bash
    python extract_frames.py -i path/to/download/ -o path/to/output_frames/
    ```
    Add `-v` if you're processing the validation clips with QP variations:
    ```bash
    python extract_frames.py -i path/to/download/ -o path/to/output_frames/ -v
    ```
3. Optional: create a mixed-QP dataset

    Use `create_mixed_dataset.py` to randomly select one QP per clip and consolidate:
    ```bash
    python create_mixed_dataset.py \
    --input path/to/output_frames/ \
    --output path/to/LR_QP_MIXED/ \
    --lists path/to/save_qp_lists/ \
    --seed 42
    ```
    * This creates a flat mixed-QP dataset with one version per clip.

    * It also saves .txt files listing which clips came from which QP level.
4. After completion your folder structure should look like this:
    ```bash
    destination_path/_
    ├── train_gt/
    │   ├── 00001_0001/
    │   ├── ...
    ├── test_lr/
    │   ├── qp_17
    │        ├── 00001_0001
    │        ├── ...
    │   ├── qp_22
    │        ├── 00001_0001
    │        ├── ...
    ```
#### 3.3 Organizing data for training
You can train a model with this framework in two ways:

* Directly using raw PNG image files, or

* Using a structured LMDB database, which offers faster data loading and is therefore recommended for efficient training.

---

1. Training on raw PNG files

    The raw PNG data should be organized in a hierarchical folder structure as follows:
    ```bash
    destination_path/
    ├── GT/               # High-Resolution (Ground Truth) sequences
    │   ├── 00001/
    │   ├── 00002/
    │   └── ...
    ├── LR/               # Low-Resolution sequences (inputs)
    │   ├── 00001/
    │   ├── 00002/
    │   └── ...
    ```
    * Each numbered folder (e.g., 00001) contains sequential frames of one video sequence stored as .png files.

2. Training using LMDB databases
LMDB databases store all frames in a compact format, improving data loading speed during training, which is especially beneficial for large datasets.

* For each HR/LR dataset, you should create a corresponding LMDB database file pair.

* Use the provided script src/scripts/create_lmdb.py to generate the LMDB files.
    ```bash
    python src/scripts/create_lmdb.py \
    --dataset Vimeo90K \
    --raw_dir /path/to/raw/GT/ \
    --lmdb_dir /path/to/output/GT.lmdb \
    --split_ratio 2_2 \
    --split \
    --downsample
    ```
    Parameters explained:

    * --dataset: Name of your dataset (used for metadata).

    * --raw_dir: Path to the folder containing raw .png sequences (HR or LR).

    * --lmdb_dir: Destination path to save the LMDB database.


    * --split_ratio (optional): Split each frame spatially into H x W subregions (e.g., 2_2 splits each frame into 4 quarters). Requires --split to be enabled.

    * --split (flag): Enable spatial splitting of frames into multiple smaller clips.

    * --downsample (flag): Downsample images by a factor of 4 before saving (useful if your hardware requires it).

    Notes:

    * Run this command separately for HR and LR datasets to create corresponding LMDB databases.

    * The script outputs metadata files that the training framework will use to index and load frames efficiently.


### 4. Train a model
Use the `train mode` to start training your model.

Example command:
```bash
python main_script.py --exp_dir experiments/exp1 --mode train --opt config/train.yml --gpu_id 0
```
* `--exp_dir` : Directory where checkpoints, logs, tensorboard data, and results will be saved.

* `--mode` train : Starts the training procedure.

* `--opt` : Path to the YAML config file containing all training and dataset options. You may find several configuration files in `src/config`.

* `--gpu_id` : GPU to use (set to -1 for CPU).

### 5. Evaluate a model
Use the test mode to evaluate a pretrained model checkpoint.

Example command:
```bash
python main_script.py --exp_dir experiments/exp1 --mode test --opt config/test.yml --gpu_id 0
```
The chosen config YAML should specify:
* `model.generator.load_path`: pretrained model checkpoint to evaluate.

* Dataset info with prefix `test*` for testing datasets.

* Flags like save_res and save_json for saving output images and metrics.

**Note:**  you can find several pretrained weights undr the `weights` directory.

### 6. Profile a model (Size and FPS)
Use the profile mode to get model parameters, FLOPs, and optionally FPS speed.

Example command:
```bash
python main_script.py --exp_dir experiments/exp1 --mode profile --opt config/profile.yml --gpu_id 0 --lr_size 3x256x256 --test_speed
```
* `--lr_size` : Input frame size in Channels x Height x Width format (e.g., 3x256x256).

* `--test_speed` : Optional flag to run speed (FPS) tests.

### 7. Simple Inference 
The `src/codes/simple_inference.py` script performs video super-resolution inference on a set of images.

```bash
python simple_inference.py -input <cale_către_imaginile_DVD> -output <cale_către_imaginile_UHD_rezultate> -model <cale_către_modelul_antrenat> [-scale <factor_scalare>]
```
Arguments:
* `-input` : Path to the root folder containing subfolders with input low-resolution image sequences.

* `-output` : Path where the output high-resolution images will be saved, preserving the input folder structure.

* `-model` : Path to the trained super-resolution model file (PyTorch .pth file).

* `-scale` : Upscaling factor (default is 4).

Expected Input Data Structure:
* The input directory contains one or more subdirectories (clips), each holding sequential images to be processed, for example:
```css
input/
├── clip1/
│   ├── image001.png
│   ├── image002.png
│   └── ...
├── clip2/
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── ...
```
* The output directory will mirror the input folder structure, with the processed high-resolution images saved accordingly.

### Additional Notes
* Ensure the YAML config file has all necessary entries such as:

    * Dataset paths: dataset.train, dataset.test*

    * Model parameters under model.generator

    * Training options: train.total_iter, train.start_iter, train.gt_bit_depth

    * Testing options: test.test_freq, test.save_res, test.save_json, test.res_dir, test.json_dir

    * Logger options: logger.log_freq, logger.ckpt_freq, logger.tb_freq

* Important Note on **BasicVSR Inference**:
    * To run video super-resolution inference using BasicVSR you need to set up [BasicSR](https://github.com/XPixelGroup/BasicSR) framework by following the instructions provided within their reposiory.
    * Use `inference/inference_basicvsr.py` to perform prediction:
        ```bash
        python inference_basicvsr.py --model_path <path_to_pretrained_model> --input_path <input_images_or_video> --save_path <output_folder> [--interval <frames_per_batch>]
        ```
    * Arguments:
        * `--model_path`: Path to the pretrained BasicVSR model.
        * `--input_path`: Folder with input images or a video file.
        * `--save_path`: Folder to save super-resolved output images.
        * `--interval`: Number of frames processed per batch (default 15).
    * You can use the pretrained model: `weights/BasicVSR_Vimeo90K_BIx4.pth`, which is the original model provided by BasicSR.