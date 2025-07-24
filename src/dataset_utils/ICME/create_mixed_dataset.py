import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict


def main(args):
    random.seed(args.seed)

    lr_root = Path(args.input)
    output_root = Path(args.output)
    lists_root = Path(args.lists)

    # Step 1: Find all QP folders and map each clip name to the QPs it's available in
    qp_dirs = list(lr_root.glob("qp_*"))
    clip_to_qps = {}  # clip_name -> list of QPs where it exists

    for qp_dir in qp_dirs:
        qp_value = qp_dir.name  # e.g., QP_22
        for clip_dir in qp_dir.iterdir():
            if clip_dir.is_dir():
                clip_name = clip_dir.name
                clip_to_qps.setdefault(clip_name, []).append(qp_value)

    all_clip_names = list(clip_to_qps.keys())
    print(f"Found {len(all_clip_names)} unique clips across {len(qp_dirs)} QP levels.")

    # Step 2: Randomly select one QP for each clip
    selected_qp_per_clip = {}  # clip_name -> chosen QP
    for clip_name, qps in clip_to_qps.items():
        selected_qp = random.choice(qps)
        selected_qp_per_clip[clip_name] = selected_qp

    # 3. Load or create per-QP clip selection
    qp_to_selected_clips = defaultdict(list)
    # Check if .txt lists exist in the lists folder
    existing_txts = {p.stem: p for p in lists_root.glob("QP_*.txt")}
    for clip_name, qps in clip_to_qps.items():
        chosen_qp = None

        # Check if this clip exists in any of the .txt files
        for qp in qps:
            if qp in existing_txts:
                with open(existing_txts[qp], "r") as f:
                    listed_clips = set(line.strip() for line in f)
                    if clip_name in listed_clips:
                        chosen_qp = qp
                        break

        # If not listed, choose randomly and append to list
        if not chosen_qp:
            chosen_qp = random.choice(qps)
            txt_path = lists_root / f"{chosen_qp}.txt"
            with open(txt_path, "a") as f:
                f.write(clip_name + "\n")

        selected_qp_per_clip[clip_name] = chosen_qp
        qp_to_selected_clips[chosen_qp].append(clip_name)

    # Step 4: Create output folders if they don't exist
    output_root.mkdir(parents=True, exist_ok=True)
    lists_root.mkdir(parents=True, exist_ok=True)

    # Step 5: Copy selected clips to output folder
    for clip_name, qp in selected_qp_per_clip.items():
        src = lr_root / qp / clip_name
        dst = output_root / clip_name
        shutil.copytree(src, dst)
        print(f"Copied: {clip_name} from {qp}")

    # Step 6: Save a .txt file listing selected clips for each QP
    for qp, clip_list in qp_to_selected_clips.items():
        txt_path = lists_root / f"{qp}.txt"
        with open(txt_path, "w") as f:
            for clip in sorted(clip_list):
                f.write(clip + "\n")
        print(f"Saved: {txt_path} ({len(clip_list)} clips)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly select one QP per clip and copy clips into a QP-mixed dataset folder."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the LR folder containing QP_* subfolders")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output folder where selected clips will be copied (e.g., LR_QP_MIXT)")
    parser.add_argument("--lists", type=str, required=True,
                        help="Path to folder where per-QP .txt lists will be saved")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    main(args)
