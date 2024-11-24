import cv2
import numpy as np
import random
import math

def random_perspective(img, targets=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0):
    """
    Apply a perspective transformation with random parameters to an image and its bounding boxes.
    """
    height, width = img.shape[:2]

    # Create a center adjustment matrix
    C = np.eye(3)
    C[0, 2] = -width / 2
    C[1, 2] = -height / 2

    # Add perspective distortion
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    # Rotation and scaling
    R = np.eye(3)
    angle = random.uniform(-degrees, degrees)
    scale_factor = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D((0, 0), angle, scale_factor)

    # Apply shear transformation
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # Add random translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    # Combine all transformations into one matrix
    M = T @ S @ R @ P @ C
    if perspective:
        img = cv2.warpPerspective(img, M, (width, height), borderValue=(114, 114, 114))
    else:
        img = cv2.warpAffine(img, M[:2], (width, height), borderValue=(114, 114, 114))

    # Adjust bounding box coordinates if provided
    if len(targets):
        n = len(targets)
        coords = np.ones((n, 3))
        coords[:, :2] = targets
        coords = coords @ M.T
        if perspective:
            coords = (coords[:, :2] / coords[:, 2:3]).reshape(-1, 2)
        else:
            coords = coords[:, :2].reshape(-1, 2)
        targets = coords

    return img, targets


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """
    Randomly modify the hue, saturation, and brightness of an image.
    """
    random_factors = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    dtype = img.dtype
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_img[..., 0] = (hsv_img[..., 0] * random_factors[0]) % 180
    hsv_img[..., 1] = np.clip(hsv_img[..., 1] * random_factors[1], 0, 255)
    hsv_img[..., 2] = np.clip(hsv_img[..., 2] * random_factors[2], 0, 255)
    return cv2.cvtColor(hsv_img.astype(dtype), cv2.COLOR_HSV2BGR)


def cutout(image, targets, fraction=0.5):
    """
    Randomly mask out regions of the image to simulate occlusions.
    """
    height, width = image.shape[:2]
    scales = [0.5, 0.25, 0.125, 0.0625]

    for scale in scales:
        mask_h = random.randint(1, int(height * scale))
        mask_w = random.randint(1, int(width * scale))

        x_min = max(0, random.randint(0, width) - mask_w // 2)
        y_min = max(0, random.randint(0, height) - mask_h // 2)
        x_max = min(width, x_min + mask_w)
        y_max = min(height, y_min + mask_h)

        image[y_min:y_max, x_min:x_max] = [random.randint(0, 255) for _ in range(3)]

    return image, targets


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize an image to maintain its aspect ratio with padding.
    """
    shape = img.shape[:2]
    scaling_ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    unpadded_shape = (int(round(shape[1] * scaling_ratio)), int(round(shape[0] * scaling_ratio)))
    dw, dh = new_shape[1] - unpadded_shape[0], new_shape[0] - unpadded_shape[1]
    dw, dh = np.mod(dw, 32) // 2, np.mod(dh, 32) // 2
    img = cv2.resize(img, unpadded_shape, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img


def load_mosaic(images, targets, img_size):
    """
    Combine four images into a mosaic layout.
    """
    s = img_size
    yc, xc = [random.randint(s // 2, s * 3 // 2) for _ in range(2)]
    mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
    mosaic_targets = []

    for img, target in zip(images, targets):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = xc, yc, min(xc + w, s * 2), min(yc + h, s * 2)
        mosaic_img[y1:y2, x1:x2] = img[:y2 - y1, :x2 - x1]
        mosaic_targets.append(target)

    mosaic_targets = np.concatenate(mosaic_targets, axis=0)
    return mosaic_img, mosaic_targets


def mixup(img1, targets1, img2, targets2, alpha=0.5):
    """
    Blend two images together and combine their associated labels.
    """
    blended_img = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    blended_targets = np.concatenate((targets1, targets2), axis=0)
    return blended_img, blended_targets
