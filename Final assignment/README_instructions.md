## Starting point and required files

This repository is based on the provided 5LSM0 final assignment template. The original course instructions are still included in the repository. This README mainly documents the implementation-specific files that were added or modified for this project.

### Files needed to start this implementation

The main implementation files are located in the `Final assignment/` directory.

The most important files are:

| File | Purpose |
|---|---|
| `config.py` | Selects which model to use, either `segformer` or `unet`.|
| `UNet.py` | Attention ResUNet model. |
| `Segformer.py` | SegFormer model. |
| `unified_train.py` | Main training script selects the model based on `config.py`. |
| `main.sh` | Shell script that launches training with the chosen model and hyperparameters.|
| `jobscript_slurm.sh` | SLURM job script used to run `main.sh` on Snellius.|
| `predict.py` | Inference script used inside the Docker submission container.|
| `Dockerfile` | Builds the Docker image for challenge-server submission. General Docker submission steps are explained in `README-Submission.md`. |

### Files/directories not included by default

Some files and folders are generated, downloaded, or added separately and are therefore not expected to be present immediately after cloning:

| File/folder | Location | How it is obtained |
|---|---|---|
| `data/` | Snellius | Downloaded using the course-provided `download_docker_and_data.sh` script. This contains the original Cityscapes data used for training and validation. See `README-Slurm.md`. |
| `container.sif` | Snellius | Downloaded using `download_docker_and_data.sh|
| Adverse Cityscapes dataset | Local machine | Added separately. This dataset is used only for the local weather-condition evaluation. It can be downloaded from Hugging Face: `https://huggingface.co/datasets/naufalso/cityscape-adverse/viewer`. Only the validation split is needed. |
| Matching original Cityscapes validation labels | Local machine | Added separately. Download `gtFine_trainvaltest.zip` from the official Cityscapes download page and use the `gtFine/val/` folder. |
| `mit-b1/` | Local machine / Docker build context | Added separately when using SegFormer with local pretrained weights. Download the model files from `https://huggingface.co/nvidia/mit-b1` and place the complete folder inside `Final assignment/` as `Final assignment/mit-b1/`. This location is required because the submission `Dockerfile` expects the folder to be available in the same build context as `predict.py`, `config.py`, `Segformer.py`, and the selected model checkpoint. |


For the local adverse-weather evaluation, the adverse dataset should be placed in the repository with the following structure:

```text
NNCV-main/
└── cityscape-adverse/
    ├── val/
    │   ├── autumn/
    │   │   ├── frankfurt/
    │   │   ├── lindau/
    │   │   └── munster/
    │   ├── dawn/
    │   │   ├── frankfurt/
    │   │   ├── lindau/
            .
            .
            .
            .
            .
    │       └── munster/
    └── val_label/
        ├── frankfurt/
        ├── lindau/
        └── munster/
```

## Starting training
The user is assumed to already have the required data, container, `.env` file, and implementation files in place.

### 1. Go to the project folder on Snellius

### 2. Select the model to train

Open `config.py` and set the model:

```python
MODEL_TYPE = "segformer"
```

or:

```python
MODEL_TYPE = "unet"
```
The training command is defined in `main.sh`.

For SegFormer, the script runs:

```bash
python3 unified_train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 1e-4 \
    --num-workers 12 \
    --seed 42 \
    --experiment-id "segformer"
```

For UNet, the script runs:

```bash
python3 unified_train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --num-workers 12 \
    --seed 42 \
    --experiment-id "unet"
```

### 3. Submit the training job

### 4. Extract weights 
The trained weights used for the reported experiments are also provided in the `Final assignment/` folder:

| File | Description |
|---|---|
| `model_segformer.pt` | SegFormer checkpoint without data augmentation. |
| `model_segformer_w_aug.pt` | SegFormer checkpoint trained with data augmentation. |
| `model_unet.pt` | Attention ResUNet checkpoint without data augmentation. |
| `model_unet_w_aug.pt` | Attention ResUNet checkpoint trained with data augmentation. |

These files can be used directly for inference or Docker submission without retraining the models. To use one of them, set the corresponding model type and checkpoint path in `config.py`.

## Running inference

Inference is handled by `predict.py`. The script loads the selected model checkpoint, processes input images, and writes predicted segmentation masks to an output folder.

### 1. Select the model checkpoint

Open `config.py` and set the model type and checkpoint path.

Example for SegFormer with augmentation:

```python
MODEL_TYPE = "segformer"
MODEL_PATH = "/app/model_segformer_w_aug.pt"
```

Example for SegFormer without augmentation:

```python
MODEL_TYPE = "segformer"
MODEL_PATH = "/app/model_segformer.pt"
```

Example for Attention ResUNet with augmentation:

```python
MODEL_TYPE = "unet"
MODEL_PATH = "/app/model_unet_w_aug.pt"
```

Example for Attention ResUNet without augmentation:

```python
MODEL_TYPE = "unet"
MODEL_PATH = "/app/model_unet.pt"
```

The `MODEL_TYPE` must match the checkpoint. For example, do not load a UNet checkpoint while `MODEL_TYPE` is set to `"segformer"`.

### 2. Build the Docker image

From the repository root, run:

```bash
docker build -t nncv-submission:latest -f "Final assignment/Dockerfile" "Final assignment"
```

The Dockerfile copies the required inference files into the container, including:

```text
predict.py
config.py
UNet.py
Segformer.py
model_segformer.pt
model_segformer_w_aug.pt
model_unet.pt
model_unet_w_aug.pt
mit-b1/
```

### 3. Prepare local input and output folders

For local testing, place input images in:

```text
local_data/
```

Create an output folder:

```text
local_output/
```

### 4. Run inference locally

On Windows PowerShell:

```powershell
docker run --rm 
  -v "${PWD}\local_data:/data" 
  -v "${PWD}\local_output:/output" 
  -v "${PWD}\cityscape-adverse:/cityscape-adverse" 
  nncv-submission:latest
```
The predicted masks are saved to `local_output/` with the same filenames as the input images.

### 5. Run local adverse-weather evaluation

If the `cityscape-adverse/` folder is available locally, it can also be mounted into the Docker container.

If `/cityscape-adverse` exists inside the container, `predict.py` also computes per-weather, per-class IoU scores for the local adverse-weather validation set.

If `/cityscape-adverse` is not mounted, the script skips this local evaluation and only performs standard inference on `/data`.

### 6. Export the Docker image for challenge submission

### 7. Inference summary

The inference workflow is:

```text
local_data/*.png
        ↓
Docker container
        ↓
predict.py
        ↓
selected model from config.py
        ↓
local_output/*.png
```

