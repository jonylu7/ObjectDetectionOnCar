import torch
import json
import os
import config
import matplotlib.patches as patches
from matplotlib import pyplot as plt


def get_iou(p, a):
    p_tl, p_br = bbox_to_coords(p)          # (batch, S, S, B, 2)
    # print(p_tl)
    # print(p_br)
    a_tl, a_br = bbox_to_coords(a)

    # Largest top-left corner and smallest bottom-right corner give the intersection
    coords_join_size = (-1, -1, -1, config.B, config.B, 2)
    tl = torch.max(
        p_tl.unsqueeze(4).expand(coords_join_size),         # (batch, S, S, B, 1, 2) -> (batch, S, S, B, B, 2)
        a_tl.unsqueeze(3).expand(coords_join_size)          # (batch, S, S, 1, B, 2) -> (batch, S, S, B, B, 2)
    )
    br = torch.min(
        p_br.unsqueeze(4).expand(coords_join_size),
        a_br.unsqueeze(3).expand(coords_join_size)
    )
    # print(tl)
    # print(br)

    intersection_sides = torch.clamp(br - tl, min=0.0)
    # print(intersection_sides)
    intersection = intersection_sides[..., 0] \
                   * intersection_sides[..., 1]       # (batch, S, S, B, B)

    # p_sides = p_br - p_tl
    # p_area = p_sides[:, :, :, :, 0] * p_sides[:, :, :, :, 1]
    p_area = bbox_attr(p, 2) * bbox_attr(p, 3)                  # (batch, S, S, B)
    p_area = p_area.unsqueeze(4).expand_as(intersection)        # (batch, S, S, B, 1) -> (batch, S, S, B, B)

    # a_sides = a_br - a_tl
    # a_area = a_sides[:, :, :, :, 0] * a_sides[:, :, :, :, 1]
    a_area = bbox_attr(a, 2) * bbox_attr(a, 3)                  # (batch, S, S, B)
    a_area = a_area.unsqueeze(3).expand_as(intersection)        # (batch, S, S, 1, B) -> (batch, S, S, B, B)

    union = p_area + a_area - intersection

    # Catch division-by-zero
    zero_unions = (union == 0.0)
    union[zero_unions] = config.EPSILON
    intersection[zero_unions] = 0.0

    return intersection / union


def bbox_to_coords(t):
    """Changes format of bounding boxes from [x, y, width, height] to ([x1, y1], [x2, y2])."""

    width = bbox_attr(t, 2)
    x = bbox_attr(t, 0)
    x1 = x - width / 2.0
    x2 = x + width / 2.0

    height = bbox_attr(t, 3)
    y = bbox_attr(t, 1)
    y1 = y - height / 2.0
    y2 = y + height / 2.0

    return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)


def scheduler_lambda(epoch):
    if epoch < config.WARMUP_EPOCHS + 75:
        return 1
    elif epoch < config.WARMUP_EPOCHS + 105:
        return 0.1
    else:
        return 0.01


def load_class_dict():
    if os.path.exists(config.CLASSES_PATH):
        with open(config.CLASSES_PATH, 'r') as file:
            return json.load(file)
    new_dict = {}
    save_class_dict(new_dict)
    return new_dict


def load_class_array():
    classes = load_class_dict()
    result = [None for _ in range(len(classes))]
    for c, i in classes.items():
        result[i] = c
    return result


def save_class_dict(obj):
    folder = os.path.dirname(config.CLASSES_PATH)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(config.CLASSES_PATH, 'w') as file:
        json.dump(obj, file, indent=2)


def get_dimensions(label):
    size = label['annotation']['size']
    return int(size['width']), int(size['height'])


def get_bounding_boxes(label):
    width, height = get_dimensions(label)
    x_scale = config.IMAGE_SIZE[0] / width
    y_scale = config.IMAGE_SIZE[1] / height
    boxes = []
    objects = label['annotation']['object']
    for obj in objects:
        box = obj['bndbox']
        coords = (
            int(int(box['xmin']) * x_scale),
            int(int(box['xmax']) * x_scale),
            int(int(box['ymin']) * y_scale),
            int(int(box['ymax']) * y_scale)
        )
        name = obj['name']
        boxes.append((name, coords))
    return boxes


def bbox_attr(data, i):
    """Returns the Ith attribute of each bounding box in data."""

    attr_start = config.C + i
    return data[..., attr_start::5]


def scale_bbox_coord(coord, center, scale):
    return ((coord - center) * scale) + center


def plot_boxes(data, labels, classes, threshold=0.5):
    """Plots bounding boxes on the given image."""

    grid_size_x = data.size(dim=2) / config.S
    grid_size_y = data.size(dim=1) / config.S

    fig, ax = plt.subplots()
    plt.imshow(data.permute(1, 2, 0))
    for i in range(labels.size(dim=0)):
        for j in range(labels.size(dim=1)):
            for k in range(config.B):
                bbox_start = 5 * k + config.C
                bbox_end = 5 * (k + 1) + config.C
                bbox = labels[i, j, bbox_start:bbox_end]
                confidence = bbox[4].item()
                if confidence > threshold:
                    class_index = torch.argmax(labels[i, j, :config.C]).item()
                    width = bbox[2] * config.IMAGE_SIZE[0]
                    height = bbox[3] * config.IMAGE_SIZE[1]
                    bbox_tl = (
                        (bbox[0] + j) * grid_size_x - width / 2,
                        (bbox[1] + i) * grid_size_y - height / 2
                    )
                    rect = patches.Rectangle(
                        bbox_tl,
                        width,
                        height,
                        facecolor='none',
                        linewidth=1,
                        edgecolor='orange'
                    )
                    ax.add_patch(rect)
                    ax.text(
                        bbox_tl[0],
                        bbox_tl[1],
                        f'{classes[class_index]} {round(confidence * 100, 1)}%',
                        bbox=dict(facecolor='orange', edgecolor='none'),
                        fontsize=6
                    )
    plt.show()
