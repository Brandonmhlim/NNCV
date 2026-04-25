## Starting point and required files

This repository is based on the provided 5LSM0 final assignment template. The original course instructions are still included in the repository. This README mainly documents the implementation-specific files that were added or modified for this project.

### Files needed to start this implementation

The main implementation files are located in the `Final assignment/` directory.

The most important files are:

| File | Purpose |
|---|---|
| `config.py` | Selects which model to use, either `segformer` or `unet`, and defines the model checkpoint path used during inference. |
| `UNet.py` | Contains the Attention ResUNet model implementation. |
| `Segformer.py` | Contains the SegFormer model implementation. |
| `unified_train.py` | Main training script used for both UNet and SegFormer. It selects the model based on `config.py`. |
| `main.sh` | Shell script that launches training with the chosen model and hyperparameters. This is executed inside the cluster container. |
| `jobscript_slurm.sh` | SLURM job script used to run `main.sh` on Snellius. General usage is explained in `README-Slurm.md`. |
| `predict.py` | Inference script used inside the Docker submission container. |
| `Dockerfile` | Builds the Docker image for challenge-server submission. General Docker submission steps are explained in `README-Submission.md`. |

### Files/directories not included by default

Some files and folders are generated, downloaded, or added separately and are therefore not expected to be present immediately after cloning:

| File/folder | Location | How it is obtained |
|---|---|---|
| `data/` | Snellius | Downloaded using the course-provided `download_docker_and_data.sh` script. This contains the original Cityscapes data used for training and validation. See `README-Slurm.md`. |
| `container.sif` | Snellius | Downloaded using `download_docker_and_data.sh`. This is the Singularity container used to run the project on Snellius. See `README-Slurm.md`. |
| Adverse Cityscapes dataset | Local machine | Added separately. This dataset is used only for the local weather-condition evaluation and is not part of the default course template. It can be downloaded from Hugging Face: `https://huggingface.co/datasets/naufalso/cityscape-adverse/viewer`. Only the validation split is needed. |
| Matching original Cityscapes validation labels | Local machine | Added separately. The adverse-weather images need to be evaluated against the matching original Cityscapes validation ground-truth labels. Download `gtFine_trainvaltest.zip` from the official Cityscapes download page and use the `gtFine/val/` folder. |
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
    │   │   └── munster/
    │   ├── foggy/
    │   │   ├── frankfurt/
    │   │   ├── lindau/
    │   │   └── munster/
    │   ├── night/
    │   │   ├── frankfurt/
    │   │   ├── lindau/
    │   │   └── munster/
    │   ├── original/
    │   │   ├── frankfurt/
    │   │   ├── lindau/
    │   │   └── munster/
    │   ├── rainy/
    │   │   ├── frankfurt/
    │   │   ├── lindau/
    │   │   └── munster/
    │   ├── snow/
    │   │   ├── frankfurt/
    │   │   ├── lindau/
    │   │   └── munster/
    │   ├── spring/
    │   │   ├── frankfurt/
    │   │   ├── lindau/
    │   │   └── munster/
    │   └── sunny/
    │       ├── frankfurt/
    │       ├── lindau/
    │       └── munster/
    └── val_label/
        ├── frankfurt/
        ├── lindau/
        └── munster/
