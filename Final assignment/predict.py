"""
This script provides and example implementation of a prediction pipeline 
for a PyTorch U-Net model. It loads a pre-trained model, processes input 
images, and saves the predicted segmentation masks. 

You can use this file for submissions to the Challenge server. Customize 
the `preprocess` and `postprocess` functions to fit your model's input 
and output requirements.
"""
from pathlib import Path
import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose, 
    ToImage, 
    Resize, 
    ToDtype, 
    Normalize,
    InterpolationMode,
)
from config import MODEL_TYPE, MODEL_PATH
from tqdm import tqdm

BATCH_SIZE = 4

# Fixed paths inside participant container
# Do NOT chnage the paths, these are fixed locations where the server will 
# provide input data and expect output data.
# Only for local testing, you can change these paths to point to your local data and output folders.
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
#MODEL_PATH = "/app/model.pt"

local_benchmark_dir = Path("/cityscape-adverse")
print(f"Found /cityscape-adverse: {local_benchmark_dir.exists()}")

#### Information about the dataset ####
num_classes = 19
ignore_index = 255
weather_folders = ["autumn", "dawn", "foggy", "night", "rainy", "snow", "spring", "sunny", "original"] # skip "original"
class_names = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", 
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", 
    "car", "truck", "bus", "train", "motorcycle", "bicycle"
]
city_names = ["frankfurt", "lindau", "munster"]

labelpixel_to_id = {7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 31:16, 32:17, 33:18}

def build_model(): 
    if MODEL_TYPE == "segformer":
        from Segformer import Model
        print("SEGFORMER LOADINGGG")
        return Model()
    elif MODEL_TYPE == "unet":
        from UNet import Model
        print("UNET LOADINGGG")
        return Model()
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")


def preprocess(img: Image.Image) -> torch.Tensor:
    # Implement your preprocessing steps here
    # For example, resizing, normalization, etc.
    # Return a tensor suitable for model input
    transform = Compose([
        ToImage(),
        Resize(size=(512, 512), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    # Implement your postprocessing steps here
    # For example, resizing back to original shape, converting to color mask, etc.
    # Return a numpy array suitable for saving as an image
    pred_soft = nn.Softmax(dim=1)(pred)
    pred_max = torch.argmax(pred_soft, dim=1, keepdim=True)  # Get the class with the highest probability
    prediction = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_max)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()  # Remove batch and channel dimensions if necessary

    return prediction_numpy

def match_label_to_id(gt):
    gt_mapped = np.full_like(gt, fill_value=ignore_index)  # Initialize with ignore_index
    for label_pixel, id in labelpixel_to_id.items():
        gt_mapped[gt == label_pixel] = id
    return gt_mapped

def compute_per_image_class_iou(prediction, gt):
    validity_mask = (gt != ignore_index)
    prediction_valid = prediction[validity_mask]
    gt_valid = gt[validity_mask]
    
    ious_per_class = {}
    
    for class_id in range(19):  # For each class
        pred_class = (prediction_valid == class_id)
        gt_class = (gt_valid == class_id)

        intersection = np.sum(pred_class & gt_class)
        union = np.sum(pred_class) + np.sum(gt_class) - intersection

        if union == 0:
            ious_per_class[class_id] = np.nan
        else:
            ious_per_class[class_id] = intersection / union
        
    valid_ious = [iou for iou in ious_per_class.values() if not np.isnan(iou)]
    miou = np.mean(valid_ious) if len(valid_ious) > 0 else np.nan
        
    return ious_per_class, miou

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = build_model()
    
    state_dict = torch.load(
        MODEL_PATH, 
        map_location=device,
        weights_only=True,
    )
        
    model.load_state_dict(
        state_dict, 
        strict=True,  # Ensure the state dict matches the model architecture
    )
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png"))  # DO NOT CHANGE, IMAGES WILL BE PROVIDED IN THIS FORMAT
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            original_shape = np.array(img).shape[:2]

            # Preprocess
            img_tensor = preprocess(img).to(device)

            # Forward pass
            pred = model(img_tensor)

            # Postprocess to segmentation mask
            seg_pred = postprocess(pred, original_shape)

            # Create mirrored output folder
            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Save predicted mask
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)
            
        print("Checking whether local benchmark dir exists")
        if local_benchmark_dir.exists():
            print("found local benchmark dir, running mIoU evaluation")
            weather_mious = {w: [] for w in weather_folders}

            total_img = 0
            with torch.no_grad():
                for weather in tqdm(weather_folders, desc="Weather", ncols=80):
                    weather_dir = local_benchmark_dir / "val" / weather
                    print(f"going into weather condition: {weather}")
                    if not weather_dir.exists():
                        print(f"bomboclaat, weather {weather_dir} aint there bruv")
                        continue

                    for city in city_names:
                        city_dir = weather_dir / city
                        print(f"Processing city: {city}")
                        if not city_dir.exists():
                            print("bomboclaat, city dir aint there bruv:", city_dir)
                            continue

                        img_paths = list(city_dir.glob("*.png"))
                        if not img_paths:
                            print("bomboclaat, no images found for city dir:", city_dir)
                            continue

                        # sample 20 random images per city
                        if len(img_paths) > 20:
                            img_paths = random.sample(img_paths, 20)

                        total_img += len(img_paths)
                        
                        # batch inference
                        for i in range(0, len(img_paths), BATCH_SIZE):
                            chunk = img_paths[i:i+BATCH_SIZE]
                            batch = []
                            img_names = []

                            for img_path in chunk:
                                print(f"Processing image: {img_path.name}")
                                img = Image.open(img_path).convert("RGB")
                                original_shape = (1024, 2048)

                                # correct stem extraction
                                img_names.append(img_path.stem.replace("_leftImg8bit", ""))

                                batch.append(preprocess(img).squeeze(0))

                            batch_tensor = torch.stack(batch).to(device)
                            predictions = model(batch_tensor)
                            print("Batch done")

                            for prediction, img_name in zip(predictions, img_names):
                                print(f"Postprocessing image: {img_name}")
                                prediction_mask = postprocess(prediction.unsqueeze(0), original_shape)

                                gt_path = local_benchmark_dir / "val_label" / city / f"{img_name}_gtFine_labelIds.png"
                                if not gt_path.exists():
                                    print("couldnt find gt for image:", img_name)
                                    continue

                                gt = np.array(Image.open(gt_path))
                                gt = match_label_to_id(gt)

                                ious_per_class, miou = compute_per_image_class_iou(prediction_mask, gt)
                                weather_mious[weather].append(ious_per_class)
            print("Finished mIoU evaluation, saving results...")
            
            ######## added ########
            print("Per‑weather per‑class mIoU")

            for weather in weather_folders: # goes through each weather condition
                class_iou_lists = weather_mious.get(weather, []) # these are the iou values per image for the given weather condition, per class

                # accumulate per-class IoUs
                per_class_values = {cid: [] for cid in range(num_classes)} # creates dictionary. To collect all iou values of class 0, 1, 2,3,...18 across all images of the given weather condition. So we will have 19 lists, each containing the iou values of that class across all images of the given weather condition.

                for img_iou_dict in class_iou_lists: # per image iou dict for some weather condition
                    for cid, val in img_iou_dict.items(): # per class iou value for the given image
                        if not np.isnan(val): # only consider valid iou values (ignore nans) 
                            per_class_values[cid].append(val) # add the iou value to the list of that class.

                # once we have all iou values for each class across all images of the given weather condition, we can compute the mean iou for each class. This will give us a single mIoU value per class for that weather condition.
                mean_per_class = {
                    cid: (np.mean(vals) if len(vals) > 0 else np.nan)
                    for cid, vals in per_class_values.items()
                }

                print(f"{weather}")
                for cid, miou in mean_per_class.items():
                    cname = class_names[cid]
                    if np.isnan(miou):
                        print(f"{cname:15s}:  n/a")
                    else:
                        print(f"{cname:15s}:  {miou:.4f}")
            ######## added ########
            
        else:
            print("cityscape-adverse not found — skipping local evaluation.")
            

            
if __name__ == "__main__":
    main()
