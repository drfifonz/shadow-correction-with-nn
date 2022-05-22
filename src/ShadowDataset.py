import os
import numpy as np
import torch
from PIL import Image


SHADOW_DATASET_PATH = "set_A"
MASK_DATASET_PATH = "set_B"
CLEAR_DATASET_PATH = "set_C"
# TEST_SHADOW_DATASET_PATH = "test/test_A"
# TEST_MASK_DATASET_PATH = "test/test_B"
# TEST_CLEAR_DATASET_PATH = "test/test_C"


class ShadowDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms) -> None:
        # dtype = torch.float32
        self.root = root
        self.transform = transforms

        # loading images
        self.shadow_images = list(
            sorted(os.listdir(os.path.join(root, SHADOW_DATASET_PATH)))
        )
        self.shadow_mask = list(
            sorted(os.listdir(os.path.join(root, MASK_DATASET_PATH)))
        )

    def __getitem__(self, index):
        # loading images and masks
        shadow_image_path = os.path.join(
            self.root, SHADOW_DATASET_PATH, self.shadow_images[index]
        )
        shadow_mask_path = os.path.join(
            self.root, MASK_DATASET_PATH, self.shadow_mask[index]
        )
        # converting images to RGB
        img = Image.open(shadow_image_path).convert("RGB")

        # converting masks to np.array
        mask = Image.open(shadow_mask_path)
        mask = np.array(mask)

        shadow_ids = np.unique(mask)
        # 1st id is background, so it is removed
        shadow_ids = shadow_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == shadow_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_shadows = len(shadow_ids)
        boxes = []
        for i in range(num_shadows):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # converting everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_shadows,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_shadows,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.shadow_images)
