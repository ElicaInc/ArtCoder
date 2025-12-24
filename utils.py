import torch
from PIL import Image
import os
import numpy as np
import math
import pandas as pd
from torchvision import transforms
import shutil

unloader = transforms.ToPILImage()
load = transforms.ToTensor()


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    
    # Convert to RGB if needed (handles grayscale, palette, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.LANCZOS)
    return img


def add_pattern(target_PIL, code_PIL, module_number=37, module_size=16):
    target_img = np.asarray(target_PIL)
    code_img = np.array(code_PIL)
    
    # Ensure code_img has the same dimensions as target_img
    if len(code_img.shape) == 2 and len(target_img.shape) == 3:
        # Convert grayscale to RGB by repeating the same channel
        code_img = np.stack([code_img] * 3, axis=-1)
    elif len(code_img.shape) == 3 and code_img.shape[2] == 4 and len(target_img.shape) == 3:
        # Convert RGBA to RGB
        code_img = code_img[:, :, :3]
    
    output = target_img.copy()
    output = np.require(output, dtype='uint8', requirements=['O', 'W'])
    ms = module_size  # module size
    mn = module_number  # module_number
    
    # Only add essential QR patterns (finders and alignment) - preserve content everywhere else
    # Add finder patterns only (3 corners) - 7x7 each with 1-module separators
    output[0 * ms:(8 * ms), 0 * ms:(8 * ms), :] = code_img[0 * ms:(8 * ms), 0 * ms:(8 * ms), :]  # Top-left finder
    output[((mn - 8) * ms):(mn * ms), 0 * ms:(8 * ms), :] = code_img[((mn - 8) * ms):(mn * ms), 0 * ms:(8 * ms), :]  # Bottom-left finder  
    output[0 * ms:(8 * ms), ((mn - 8) * ms):(mn * ms), :] = code_img[0 * ms:(8 * ms), ((mn - 8) * ms):(mn * ms), :]  # Top-right finder
    
    # Add alignment pattern (center at position 30,30 for version 5) - 5x5 pattern
    align_center = 30
    align_size = 2  # 5x5 pattern, so ±2 from center  
    align_start_row = (align_center - align_size) * ms
    align_end_row = (align_center + align_size + 1) * ms
    align_start_col = (align_center - align_size) * ms  
    align_end_col = (align_center + align_size + 1) * ms
    output[align_start_row:align_end_row, align_start_col:align_end_col, :] = code_img[align_start_row:align_end_row, align_start_col:align_end_col, :]

    output = Image.fromarray(output.astype('uint8'))
    print('Added finder and alignment patterns.')
    return output


def del_file(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def get_action_matrix(img_target, img_code, module_size=16, IMG_SIZE=592, Dis_b=50, Dis_w=200):
    img_code = np.require(np.asarray(img_code.convert('L')), dtype='uint8', requirements=['O', 'W'])
    img_target = np.require(np.array(img_target.convert('L')), dtype='uint8', requirements=['O', 'W'])

    ideal_result = get_binary_result(img_code, module_size)
    center_mat = get_center_pixel(img_target, module_size)
    error_module = get_error_module(center_mat, code_result=ideal_result,
                                    threshold_b=Dis_b,
                                    threshold_w=Dis_w)
    return error_module, ideal_result


def get_binary_result(img_code, module_size, module_number=37):
    binary_result = np.zeros((module_number, module_number))
    for j in range(module_number):
        for i in range(module_number):
            module = img_code[i * module_size:(i + 1) * module_size, j * module_size:(j + 1) * module_size]
            module_color = np.around(np.mean(module), decimals=2)
            if module_color < 128:
                binary_result[i, j] = 0
            else:
                binary_result[i, j] = 1
    return binary_result


def get_center_pixel(img_target, module_size):
    center_mat = np.zeros((37, 37))
    for j in range(37):
        for i in range(37):
            module = img_target[i * module_size:(i + 1) * module_size, j * module_size:(j + 1) * module_size]
            module_color = np.mean(module[5:12, 5:12])
            center_mat[i, j] = module_color
    return center_mat


def get_error_module(center_mat, code_result, threshold_b, threshold_w):
    error_module = np.ones((37, 37))  # 0 means correct,1 means error
    for j in range(37):
        for i in range(37):
            center_pixel = center_mat[i, j]
            right_result = code_result[i, j]
            if right_result == 0 and center_pixel < threshold_b:
                error_module[i, j] = 0
            elif right_result == 1 and center_pixel > threshold_w:
                error_module[i, j] = 0
            else:
                error_module[i, j] = 1
    return error_module


def get_target(binary_result, b_robust, w_robust, module_num=37, module_size=16):
    img_size = module_size * module_num
    target = np.require(np.ones((img_size, img_size)), dtype='uint8', requirements=['O', 'W'])

    for i in range(module_num):
        for j in range(module_num):
            # print(str(i) + ' == ' + str(j))
            one_binary_result = binary_result[i, j]
            if one_binary_result == 0:
                target[i * module_size:(i + 1) * module_size, j * module_size:(j + 1) * module_size] = b_robust
            else:
                target[i * module_size:(i + 1) * module_size, j * module_size:(j + 1) * module_size] = w_robust

    target = load(Image.fromarray(target.astype('uint8')).convert('RGB')).unsqueeze(0)
    return target


def save_image_epoch(tensor, path, name, code_pil, addpattern=True):
    """Save a single image."""
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    if addpattern == True:
        image = add_pattern(image, code_pil, module_number=37, module_size=16)
    image.save(os.path.join(path, "epoch_" + str(name)))


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def get_3DGauss(module_size=16):
    # Dynamic Gaussian kernel generation based on module_size
    s, e = 0, module_size - 1
    sigma = module_size * 0.09375  # 1.5/16 ratio for dynamic scaling
    mu = (module_size - 1) / 2  # Center position
    
    x, y = np.mgrid[s:e:complex(module_size), s:e:complex(module_size)]
    z = (1 / (2 * math.pi * sigma ** 2)) * np.exp(-((x - mu) ** 2 + (y - mu) ** 2) / (2 * sigma ** 2))
    z = torch.from_numpy(MaxMinNormalization(z.astype(np.float32)))
    
    # Apply threshold filtering
    threshold = 0.1
    z[z < threshold] = 0
    
    return z


def MaxMinNormalization(loss_img):
    maxvalue = np.max(loss_img)
    minvalue = np.min(loss_img)
    img = (loss_img - minvalue) / (maxvalue - minvalue)
    img = np.around(img, decimals=2)
    return img


def print_options(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
