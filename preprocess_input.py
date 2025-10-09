from cosmos_transfer1.diffusion.inference.preprocessors import Preprocessors
from check_videos import check_mp4_valid
import os
import json
import argparse



def test_video(input_video, output_folder, prompt="a person working", control_inputs=None, cfg_path=None):
    assert check_mp4_valid(input_video)
    os.makedirs(output_folder, exist_ok=True)

    if control_inputs is None:
        control_inputs = {
            "edge": {"canny_threshold": "medium", "control_weight": 0.25},
            "depth": {"control_weight": 0.25},
            "vis": {"blur_strength": "medium", "control_weight": 0.25},
        }

    for hint_key in list(control_inputs.keys()):
        expected_file = os.path.join(output_folder, f"{hint_key}_input_control_0.mp4")
        if os.path.exists(expected_file):
            print(f"[INFO] Found existing control video for '{hint_key}', skipping generation.")
            control_inputs.pop(hint_key)

    blur_strength   = control_inputs.get("vis", {}).get("blur_strength", "medium")
    canny_threshold = control_inputs.get("edge", {}).get("canny_threshold", "medium")

    preprocessor = Preprocessors()
    preprocessor(
        input_video=input_video,
        input_prompt=prompt,
        control_inputs=control_inputs,
        output_folder=output_folder,
        blur_strength=blur_strength,
        canny_threshold=canny_threshold,
    )

    from pathlib import Path
    
    if cfg_path:
        update_video_dir = Path(output_folder)  # <--- same folder as generated outputs

        os.makedirs(update_video_dir, exist_ok=True)
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        for hint_key in control_inputs.keys():
            expected_file = os.path.join(update_video_dir, f"{hint_key}_input_control_0.mp4")
            if os.path.exists(expected_file):
                # update field input_hint_path
                if hint_key not in cfg:
                    cfg[hint_key] = {}
                cfg[hint_key]["input_hint_path"] = expected_file
                cfg[hint_key]["control_weight"] = cfg[hint_key].get("control_weight", 0.25)
        # save new JSON next to control videos
        tmp_path, file = cfg_path.split("/")
        new_path = tmp_path + "/update" 
        os.makedirs(new_path,exist_ok=True)
        with open(new_path + "/" + file, "w") as f:
            json.dump(cfg, f, indent=4)
        print(f"[INFO] Saved updated JSON with hint paths: {new_path+"/"+file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess control videos for Cosmos-Transfer1")
    parser.add_argument(
        "--video_save_folder",
        type=str,
        required=True,
        help="Folder to save generated control videos."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to JSON config or .mp4 file path."
    )

    args = parser.parse_args()

    # --- Parse JSON or direct mp4 ---
    input_video = None
    control_inputs = {}
    prompt = "a person working"

    cfg_path = ""
    if args.input_json.endswith(".json"):
        
        cfg_path = args.input_json
        with open(args.input_json, "r") as f:
            cfg = json.load(f)

        prompt = cfg.get("prompt", prompt)

        input_video = cfg.get("input_video_path")
        if not input_video:
            raise ValueError(" Missing 'input_video_path' in JSON config!")

        for k, v in cfg.items():
            if k in ["prompt", "input_video_path"]:
                continue
            control_inputs[k] = v

        print(f"[INFO] Loaded prompt: {prompt}")
        print(f"[INFO] Loaded control_inputs: {list(control_inputs.keys())}")

    else:
        input_video = args.input_json
        print("[INFO] Using direct .mp4 input, default control_inputs loaded.")
        control_inputs = None 

    test_video(
        input_video=input_video,
        output_folder=args.video_save_folder,
        prompt=prompt,
        control_inputs=control_inputs,
        cfg_path=cfg_path
    )
