import torch
import config
import utils
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import Dataset


class YoloPascalVocDataset(Dataset):
    def __init__(self, set_type, augment=False):
        assert set_type in {'train', 'test'}
        self.dataset = VOCDetection(
            root=config.DATA_PATH,
            year='2007',
            image_set=('train' if set_type == 'train' else 'val'),
            download=True,
            transform=T.Compose([
                T.ToTensor(),
                T.Resize(config.IMAGE_SIZE)
            ])
        )
        self.augment = augment
        self.classes = utils.load_class_dict()

        # Generate class index if needed
        index = 0
        if len(self.classes) == 0:
            for i, data_pair in enumerate(tqdm(self.dataset, desc=f'Generating class dict')):
                data, label = data_pair
                for j, bbox_pair in enumerate(utils.get_bounding_boxes(label)):
                    name, coords = bbox_pair
                    if name not in self.classes:
                        self.classes[name] = index
                        index += 1
            utils.save_class_dict(self.classes)

    def __getitem__(self, i):
        data, label = self.dataset[i]
        x_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[0])
        y_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[1])
        scale = 1 + 0.2 * random.random()

        # Augment images
        if self.augment:
            data = TF.normalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            data = TF.affine(data, angle=0.0, scale=scale, translate=(x_shift, y_shift), shear=0.0)

        grid_size_x = data.size(dim=2) / config.S  # Images in PyTorch have size (channels, height, width)
        grid_size_y = data.size(dim=1) / config.S

        # Process bounding boxes into the SxSx(5*B+C) ground truth tensor
        boxes = {}
        depth = 5 * config.B + config.C  # 5 numbers per bbox, then one-hot encoding of label
        ground_truth = torch.zeros((config.S, config.S, depth))
        for j, bbox_pair in enumerate(utils.get_bounding_boxes(label)):
            name, coords = bbox_pair
            assert name in self.classes, f"Unrecognized class '{name}'"
            class_index = self.classes[name]
            x_min, x_max, y_min, y_max = coords

            # Augment labels
            if self.augment:
                half_width = config.IMAGE_SIZE[0] / 2
                half_height = config.IMAGE_SIZE[1] / 2
                x_min = utils.scale_bbox_coord(x_min, half_width, scale) + x_shift
                x_max = utils.scale_bbox_coord(x_max, half_width, scale) + x_shift
                y_min = utils.scale_bbox_coord(y_min, half_height, scale) + y_shift
                y_max = utils.scale_bbox_coord(y_max, half_height, scale) + y_shift

            # Calculate the position of center of bounding box
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            col = int(mid_x // grid_size_x)
            row = int(mid_y // grid_size_y)

            # Insert bounding box into ground truth tensor
            if 0 <= col < config.S and 0 <= row < config.S:
                key = (row, col)
                bbox_index = boxes.get(key, 0)
                if bbox_index < config.B:
                    bbox_truth = (
                        (mid_x - col * grid_size_x) / grid_size_x,      # X coordinate relative to grid square
                        (mid_y - row * grid_size_y) / grid_size_y,      # Y coordinate relative to grid square
                        (x_max - x_min) / config.IMAGE_SIZE[0],         # Width
                        (y_max - y_min) / config.IMAGE_SIZE[1],         # Height
                        1.0  # Confidence
                    )
                    bbox_start = 5 * bbox_index
                    bbox_end = 5 * (bbox_index + 1)
                    ground_truth[row, col, bbox_start:bbox_end] = torch.tensor(bbox_truth)
                    boxes[key] = bbox_index + 1

                # Insert class one-hot encoding into ground truth
                one_hot = torch.zeros(config.C)
                one_hot[class_index] = 1.0
                ground_truth[row, col, -config.C:] = one_hot
        return data, ground_truth

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    # Display data
    obj_classes = utils.load_class_array()
    train_set = YoloPascalVocDataset('train', augment=True)

    negative_labels = 0
    smallest = 0
    largest = 0
    for data, label in train_set:
        negative_labels += torch.sum(label < 0).item()
        smallest = min(smallest, torch.min(data).item())
        largest = max(largest, torch.max(data).item())
        utils.plot_boxes(data, label, obj_classes)
    print('num_negatives', negative_labels)
    print('dist', smallest, largest)
