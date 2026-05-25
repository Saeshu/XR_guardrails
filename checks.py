import numpy as np

def check_contrast(image):
    return np.std(image)

def check_brightness(image):
    return np.mean(image)

def diagnose(image):
    issues = []

    if check_contrast(image) < 0.05:
        issues.append("low_contrast")

    if check_brightness(image) < 0.2:
        issues.append("underexposed")

    return issues
