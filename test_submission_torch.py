import sys
import pathlib
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms


def main():
    # Load the classes
    data_dir = pathlib.Path('./tiny-imagenet-200/train/')
    CLASSES = sorted([item.name for item in data_dir.glob('*')])
    im_height, im_width = 64, 64
    num_classes = 200
    input_size = 299

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    models = []
    for i in range(4):
        models.append(torch.hub.load('pytorch/vision:v0.9.0', "inception_v3", pretrained=True))
        inception_aux_in_ftrs = models[i].AuxLogits.fc.in_features
        models[i].AuxLogits.fc = nn.Linear(inception_aux_in_ftrs, num_classes)
        inception_in_ftrs = models[i].fc.in_features
        models[i].fc = nn.Linear(inception_in_ftrs, num_classes)

    models[0].load_state_dict(torch.load(
        "./final_models/inception-fine-tuned-aug-10.pt", map_location=device))
    models[1].load_state_dict(torch.load(
        "./final_models/inception-fine-tuned-noaug-10.pt", map_location=device))
    models[2].load_state_dict(torch.load(
        "./final_models/inception-fine-tuned-noaug-with-stability-10.pt", map_location=device))
    models[3].load_state_dict(torch.load(
        "./final_models/inception-merged-10.pt", map_location=device))

    for model in models:
        model.eval()

    weights = [0.264, 0.264, 0.264, 0.208]

    data_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Loop through the CSV file and make a prediction for each line
    with open('eval_classified.csv', 'w') as eval_output_file:  # Open the evaluation CSV file for writing
        for line in pathlib.Path(sys.argv[1]).open():  # Open the input CSV file for reading
            image_id, image_path, image_height, image_width, image_channels = line.strip().split(
                ',')  # Extract CSV info

            print(image_id, image_path, image_height, image_width, image_channels)
            with open(image_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img = data_transforms(img)[None, :]
            ensemble_outputs = []
            for model in models:
              ensemble_outputs.append(model(img))
            outputs = weights[0] * ensemble_outputs[0]
            for i in range(1, len(weights)):
              outputs += weights[i] * ensemble_outputs[i]
            _, predicted = outputs.max(1)

            # Write the prediction to the output file
            eval_output_file.write('{},{}\n'.format(image_id, CLASSES[predicted]))


if __name__ == '__main__':
    main()
