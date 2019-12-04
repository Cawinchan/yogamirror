import sys
import cv2
import numpy as np

sys.path.append('/usr/local/python')

from openpose import pyopenpose as op

def scale_transform(coords):
    """
    Parameters:
    coords (25x3 ndarray): array of (x,y,c) coordinates

    Returns:
    ndarray: coords scaled to 1x1 with center at (0,0)
    ndarray: confidence scores of each joint
    """
    coords, scores = coords[:, :, :-1], coords[:, :, -1]
    diff = coords.max(axis=1) - coords.min(axis=1)
    diff_max = np.max(diff, axis=0)
    mean = coords.mean(axis=1).reshape(
                coords.shape[0],
                1,
                coords.shape[-1]
    )
    out = (coords - mean) / diff_max

    return out, scores

def video_cv(video):
    """Converts video into a BGR format for display

    This is abstracted out to allow for experimentation

    Args:
        video: A numpy array with 1 byte per pixel, 3 channels RGB

    Returns:
        A numpy array with with 1 byte per pixel, 3 channels BGR
    """
    return video[:, :, :: -1]  # RGB -> BGR
#    return video[:,:,:]

def make_vector(poseKeypoints):
    """
    Parameters:
    poseKeypoints (ndarray): Single person output from OpenPose

    Returns:
    ndarray: scaled, transformed, normalized row vector
    """
    
    N, D, C = poseKeypoints.shape
    coords, pose_scores = scale_transform(poseKeypoints)
    pose_scores = pose_scores.reshape((N, D, 1))
    coords_vec = np.concatenate([coords, pose_scores], axis=2)
    coords_vec /= np.linalg.norm(coords_vec, axis=2)[:, :, np.newaxis]

    return coords_vec


def get_ordinal_score(score):
    """
    Parameters:
    score (float): similarity score between two poses
                   between 0 and 1

    Returns:
    string: string text of the results
    float: transparency value
    tuple: color of overlay
    """
    alpha = 0.2 #0.2
    overlay_color = (0, 0, 250)

    if score > 0.852: #0.712
        out = "Genius!"
        overlay_color = (0, 250, 0)
    elif score > 0.452: #0.412
        out = "Almost there!"
        overlay_color = (250, 145, 0)
    elif score > 0.298: #0.298
        out = "Nice try!"
    else:
        out = "Try harder!"

    return out, alpha, overlay_color


def crop_image(full_image, w, h):
    full_image = cv2.resize(full_image,
                            (w, h))
    #w_min = w // 2 - (w // 4)
    #w_max = w // 2 + (w // 4)
    #out = full_image[:h, w_min:w_max]
    out = full_image
    return out

def transform_image(img,w,h):
    transformed_image = cv2.flip(
            crop_image(
                img,
                w, h
            ), 1)
    return transformed_image


def get_webcam(w, h):
    stream = cv2.VideoCapture(0)
    if (stream.isOpened() is False):
        print("Error opening video stream or file")
        raise SystemExit(1)
    stream.set(3, w)
    stream.set(4, h)
    return stream


def get_image(stream, w, h):
    ret, img_original = stream.read()

    # Reset video if reached end
    if not img_original.any():
        stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, img_original = stream.read()

    img = cv2.flip(
            crop_image(
                img_original,
                w, h
            ), 1)

    return img


def label_img(opWrapper, img):
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.waitAndEmplace([datum])
    opWrapper.waitAndPop([datum])
    return datum
