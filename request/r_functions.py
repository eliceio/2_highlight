from PIL import Image
import cv2
import numpy as np


def cv_format_check_color(img):
    mask_cv_img = np.squeeze(np.array(img))
    if len(mask_cv_img.shape) == 2:
        mask_cv_img = cv2.cvtColor(mask_cv_img, cv2.COLOR_GRAY2BGR)
    elif len(mask_cv_img.shape) == 3:
        mask_cv_img = cv2.cvtColor(mask_cv_img, cv2.COLOR_RGB2BGR)
    else:
        print("[ERROR] mask_cv_img should have 2 or 3 dimensions")
    return mask_cv_img


def cv_format_check_gray(img):
    mask_cv_img = np.squeeze(np.array(img))

    if len(mask_cv_img.shape) == 2:
        pass
    elif len(mask_cv_img.shape) == 3:
        mask_cv_img = cv2.cvtColor(mask_cv_img, cv2.COLOR_RGB2GRAY)
    else:
        print("[ERROR] mask_cv_img should have 2 or 3 dimensions")
    return mask_cv_img

def url_file_open_to_np_g(url):
    print("new url_file_open_to_np_g = ", url)
    with Image.open(url) as image_open:
        img_cv = np.array(image_open)

    # img = Image.open(url)
    # img_cv = np.array(img)
    # img.close()
    img_cv = cv_format_check_gray(img_cv)
    return img_cv


def url_file_open_to_np_c(url):
    print("new url_file_open_to_np_c = ", url)
    with Image.open(url) as image_open:
        img_cv = np.array(image_open)

    # img = Image.open(url)
    # img_cv = np.array(img)
    # img.close()
    img_cv = cv_format_check_color(img_cv)
    return img_cv