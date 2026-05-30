import albumentations as A
import random
import cv2

# -------------------------
# GOOD transformations
# -------------------------
good_transforms = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    A.GaussianBlur(blur_limit=(3,5), p=0.3),
    A.Rotate(limit=5, p=0.5),
])

# -------------------------
# BAD transformations pool
# -------------------------
bad_transform_pool = [
    ("low_contrast", A.RandomBrightnessContrast(brightness_limit=-0.5, contrast_limit=-0.5, p=1.0)),
    ("high_contrast", A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1.0)),
    ("rotation", A.Rotate(limit=25, p=1.0)),
    ("blur", A.GaussianBlur(blur_limit=(9,15), p=1.0)),
    ("noise", A.GaussNoise(var_limit=(50,100), p=1.0)),
    ("downsample", A.Downscale(scale_min=0.3, scale_max=0.5, p=1.0)),
    ("shift_scale_rotate", A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=25,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=1.0
    )),
    ("occlusion", A.CoarseDropout(
        max_holes=3,
        max_height=40,
        max_width=40,
        fill_value=0,
        p=1.0
    ))
]

# -------------------------
# AUGMENTATION FUNCTION
# -------------------------
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
        num_issues = random.choice([1, 2])  # allow multiple issues
        selected = random.sample(bad_transform_pool, num_issues)

        issues = []
        augmented = image.copy()

        for issue_name, transform in selected:
            augmented = transform(image=augmented)["image"]
            issues.append(issue_name)

        metadata = {
            "label": 1,
            "issues": issues
        }

    return augmented, metadata
