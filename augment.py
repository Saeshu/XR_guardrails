import albumentations as A
import random
import cv2

# -------------------------
# GOOD transformations
# -------------------------
good_transforms = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.8),
    A.GaussianBlur(blur_limit=(3,5), p=0.3),
    A.Rotate(limit=5, p=0.5),
])

# -------------------------
# EASY BAD (obvious)
# -------------------------
easy_bad = [
    ("extreme_blur", A.GaussianBlur(blur_limit=(15,25), p=1.0)),
    ("extreme_noise", A.GaussNoise(var_limit=(100,200), p=1.0)),
    ("extreme_dark", A.RandomBrightnessContrast(brightness_limit=-0.7, contrast_limit=-0.7, p=1.0)),
]

# -------------------------
# MEDIUM BAD
# -------------------------
medium_bad = [
    ("blur", A.GaussianBlur(blur_limit=(9,15), p=1.0)),
    ("noise", A.GaussNoise(var_limit=(50,100), p=1.0)),
    ("rotation", A.Rotate(limit=25, p=1.0)),
    ("downsample", A.Downscale(scale_min=0.3, scale_max=0.5, p=1.0)),
]

# -------------------------
# HARD BAD (subtle + realistic)
# -------------------------
hard_bad = [
    ("low_contrast", A.RandomBrightnessContrast(brightness_limit=-0.3, contrast_limit=-0.3, p=1.0)),
    ("shift_scale_rotate", A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=20,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=1.0
    )),
    ("occlusion", A.CoarseDropout(
        max_holes=2,
        max_height=30,
        max_width=30,
        fill_value=0,
        p=1.0
    )),
]

def apply_augmentation(image):

    r = random.random()

    if r < 0.5:
        augmented = good_transforms(image=image)["image"]
        return augmented, {"label": 0, "issues": []}

    elif r < 0.7:
        pool = easy_bad
    elif r < 0.9:
        pool = medium_bad
    else:
        pool = hard_bad

    num_issues = random.choice([1, 2])
    selected = random.sample(pool, num_issues)

    issues = []
    augmented = image.copy()

    for name, transform in selected:
        augmented = transform(image=augmented)["image"]
        issues.append(name)

    return augmented, {"label": 1, "issues": issues}
