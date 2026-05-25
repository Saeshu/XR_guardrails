import albumentations as A
import random

# GOOD transformations
good_transforms = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    A.GaussianBlur(blur_limit=(3,5), p=0.3),
    A.Rotate(limit=5, p=0.5),
])

# BAD transformations (stronger)
bad_transforms = [
    ("low_contrast", A.RandomBrightnessContrast(brightness_limit=-0.5, contrast_limit=-0.5, p=1.0)),
    ("high_contrast", A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1.0)),
    ("rotation", A.Rotate(limit=25, p=1.0)),
    ("blur", A.GaussianBlur(blur_limit=(9,15), p=1.0)),
    ("noise", A.GaussNoise(var_limit=(50,100), p=1.0)),
    ("downsample", A.Downscale(scale_min=0.3, scale_max=0.5, p=1.0)),
]

def apply_augmentation(image):
    if random.random() < 0.5:
        # GOOD sample
        augmented = good_transforms(image=image)["image"]
        metadata = {
            "label": 0,
            "issues": []
        }
    else:
        # BAD sample
        issue, transform = random.choice(bad_transforms)
        augmented = transform(image=image)["image"]
        metadata = {
            "label": 1,
            "issues": [issue]
        }

    return augmented, metadata
  
