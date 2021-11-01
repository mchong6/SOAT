from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import os
import cv2
import dlib
from PIL import Image
import numpy as np
import math
import scipy
import scipy.ndimage


# Number of style channels per StyleGAN layer
style2list_len = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 
                  512, 512, 512, 512, 512, 256, 256, 256, 128, 128]

# Layer indices of ToRGB modules
rgb_layer_idx = [1,4,7,10,13,16,19,22,25]

google_drive_paths = {
    "stylegan2-church-config-f.pt": "https://drive.google.com/uc?id=1ORsZHZEeFNEX9HtqRutt1jMgrf5Gpcat",
    "model_ir_se50.pt": "https://drive.google.com/uc?id=1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn",
    "dlibshape_predictor_68_face_landmarks.dat": "https://drive.google.com/uc?id=11BDmNKS1zxSZxkgsEvQoKgFd8J264jKp",
    "e4e_ffhq_encode.pt": "https://drive.google.com/uc?id=1cUv_reLE6k3604or78EranS7XzuVMWeO"
}


def ensure_checkpoint_exists(model_weights_filename):
    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename in google_drive_paths
    ):
        gdrive_url = google_drive_paths[model_weights_filename]
        try:
            from gdown import download as drive_download

            drive_download(gdrive_url, model_weights_filename, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or, manually download the checkpoint file:",
                gdrive_url
            )

    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename not in google_drive_paths
    ):
        print(
            model_weights_filename,
            " not found, you may need to manually download the model weights."
        )

# given a list of filenames, load the inverted style code
@torch.no_grad()
def load_source(files, generator, device='cuda'):
    sources = []
    
    for file in files:
        source = torch.load(f'./inversion_codes/{file}.pt')['latent'].to(device)

        if source.size(0) != 1:
            source = source.unsqueeze(0)

        if source.ndim == 3:
            source = generator.get_latent(source, truncation=1, is_latent=True)
            source = list2style(source)
            
        sources.append(source)
        
    sources = torch.cat(sources, 0)
    if type(sources) is not list:
        sources = style2list(sources)
        
    return sources
# convert a style vector [B, 9088] into a suitable format (list) for our generator's input
def style2list(s):
    output = []
    count = 0 
    for size in style2list_len:
        output.append(s[:, count:count+size])
        count += size
    return output

# convert the list back to a style vector
def list2style(s):
    return torch.cat(s, 1)

# flatten spatial activations to vectors
def flatten_act(x):
    b,c,h,w = x.size()
    x = x.pow(2).permute(0,2,3,1).contiguous().view(-1, c) # [b,c]
    return x.cpu().numpy()

def show(imgs, title=None):

    plt.figure(figsize=(5 * len(imgs), 5))
    if title is not None:
        plt.suptitle(title + '\n', fontsize=24).set_y(1.05)

    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i + 1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0.02)

def part_grid(target_image, refernce_images, part_images):
    def proc(img):
        return (img * 255).permute(1, 2, 0).squeeze().cpu().numpy().astype('uint8')

    rows, cols = len(part_images) + 1, len(refernce_images) + 1
    fig = plt.figure(figsize=(cols*4, rows*4))
    sz = target_image.shape[-1]

    i = 1
    plt.subplot(rows, cols, i)
    plt.imshow(proc(target_image[0]))
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.title('Source', fontdict={'size': 26})

    for img in refernce_images:
        i += 1
        plt.subplot(rows, cols, i)
        plt.imshow(proc(img))
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.title('Reference', fontdict={'size': 26})

    for j, label in enumerate(part_images.keys()):
        i += 1
        plt.subplot(rows, cols, i)
        plt.imshow(proc(target_image[0]) * 0 + 255)
        plt.text(sz // 2, sz // 2, label.capitalize(), fontdict={'size': 30})
        plt.axis('off')
        plt.gca().set_axis_off()

        for img in part_images[label]:
            i += 1
            plt.subplot(rows, cols, i)
            plt.imshow(proc(img))
            plt.axis('off')
            plt.gca().set_axis_off()

        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.subplots_adjust(wspace=0, hspace=0)
    return fig


def display_image(image, size=None, mode='nearest', unnorm=False, title=''):
    # image is [3,h,w] or [1,3,h,w] tensor [0,1]
    if image.is_cuda:
        image = image.cpu()
    if size is not None and image.size(-1) != size:
        image = F.interpolate(image, size=(size,size), mode=mode)
    if image.dim() == 4:
        image = image[0]
    image = ((image.clamp(-1,1)+1)/2).permute(1, 2, 0).detach().numpy()
    plt.figure()
    plt.title(title)
    plt.axis('off')
    plt.imshow(image)

def get_parsing_labels():
    color = torch.FloatTensor([[0, 0, 0],
                      [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
                      [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0],
                      [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192,128,128],
                      [0, 64, 0], [0, 0, 64], [128, 0, 192], [0, 192, 128], [64,128,192], [64,64,64]])
    return (color/255 * 2)-1

def decode_segmap(seg):
    seg = seg.float()
    label_colors = get_parsing_labels()
    r = seg.clone()
    g = seg.clone()
    b = seg.clone()

    for l in range(label_colors.size(0)):
        r[seg == l] = label_colors[l, 0]
        g[seg == l] = label_colors[l, 1]
        b[seg == l] = label_colors[l, 2]

    output = torch.stack([r,g,b], 1)
    return output

def remove_idx(act, i):
    # act [N, 128]
    return torch.cat([act[:i], act[i+1:]], 0)

def interpolate_style(s, t, q):
    if isinstance(s, list):
        s = list2style(s)
    if isinstance(t, list):
        t = list2style(t)
    if s.ndim == 1:
        s = s.unsqueeze(0)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if len(s) != len(t):
        s = s.expand(t.size(0), -1)
    q = q.float()
        
    return (1 - q) * s + q * t
    
def index_layers(w, i):
    return [w[j][[i]] for j in range(len(w))]


def normalize_im(x):
    return (x.clamp(-1,1)+1)/2

def l2(a, b):
    return (a-b).pow(2).sum(1)

def cos_dist(a,b):
    return -F.cosine_similarity(a, b, 1)

def downsample(x):
    return F.interpolate(x, size=(256,256), mode='bilinear')

def get_landmark(filepath, predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    for k, d in enumerate(dets):
        shape = predictor(img, d)

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm


def align_face(filepath, output_size=512):
    """
    :param filepath: str
    :return: PIL Image
    """
    ensure_checkpoint_exists("dlibshape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor("dlibshape_predictor_68_face_landmarks.dat")
    lm = get_landmark(filepath, predictor)

    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = Image.open(filepath)

    transform_size = output_size
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    # Return aligned image.
    return img

