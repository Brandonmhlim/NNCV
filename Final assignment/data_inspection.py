import wandb
import csv

ENTITY = "m-h-lim-eindhoven-university-of-technology"
PROJECT = "5lsm0-cityscapes-segmentation"

RUNS = {
    "unet_aug": "1gyz7dg8",
    "segformer_aug": "zvj1iy32",
    "unet_no_aug": "h3jjev91",
    "segformer_no_aug": "b9acjaf8",
}

api = wandb.Api()

for name, run_id in RUNS.items():
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    history = run.history(samples=100000)

    train_losses_by_epoch = {}
    valid_losses_by_epoch = {}

    for _, row in history.iterrows():
        epoch = row.get("epoch")

        if epoch is None:
            continue

        epoch = int(epoch)

        train_loss = row.get("train_loss")
        if train_loss == train_loss:  # not NaN
            if epoch not in train_losses_by_epoch:
                train_losses_by_epoch[epoch] = []
            train_losses_by_epoch[epoch].append(float(train_loss))

        valid_loss = row.get("valid_loss")
        if valid_loss == valid_loss:  # not NaN
            valid_losses_by_epoch[epoch] = float(valid_loss)

    all_epochs = sorted(set(train_losses_by_epoch.keys()) | set(valid_losses_by_epoch.keys()))

    output_file = f"{name}_losses.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "valid_loss"])

        for epoch in all_epochs:
            if epoch in train_losses_by_epoch:
                mean_train_loss = sum(train_losses_by_epoch[epoch]) / len(train_losses_by_epoch[epoch])
            else:
                mean_train_loss = ""

            valid_loss = valid_losses_by_epoch.get(epoch, "")
            writer.writerow([epoch, mean_train_loss, valid_loss])

    print(f"saved {output_file}")