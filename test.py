import torch
from dataset import MapDataset
from generator_model import Generator
from torchvision.utils import save_image
import config

def evaluate(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for idx, (x, _) in enumerate(data_loader):
            x = x.to(device)

            # generate output
            output = model(x)

            # save generated image
            save_image(output, f"test_sample_generated/generated_{idx}.png")

def main():
    # load test dataset
    test_dataset = MapDataset(root_dir=config.VAL_DIR)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )

    # load generator model
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    checkpoint = torch.load(config.CHECKPOINT_GEN, map_location=config.DEVICE)
    gen.load_state_dict(checkpoint["state_dict"])

    # generate and save images
    evaluate(gen, test_loader, config.DEVICE)

if __name__ == "__main__":
    main()
