from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import os
import cv2
from PIL import Image
import numpy as np
import math
import scipy
import scipy.ndimage
import torchvision

# Number of style channels per StyleGAN layer
style2list_len = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 
                  512, 512, 512, 512, 512, 256, 256, 256, 128, 128]
# for 1024 x 1024
#style2list_len = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 
#                  512, 512, 512, 512, 512, 256, 256, 256, 128, 128, 
#                  128, 64, 64, 64, 32, 32]

# Layer indices of ToRGB modules
rgb_layer_idx = [1,4,7,10,13,16,19,22,25]

google_drive_paths = {
    "church.pt": "https://drive.google.com/uc?id=1ORsZHZEeFNEX9HtqRutt1jMgrf5Gpcat",
    "face.pt": "https://drive.google.com/uc?id=1dOBo4xWUwM7-BwHWZgp-kV1upaD6tHAh",
    "landscape.pt": "https://drive.google.com/uc?id=1rN5EhwiY95BBNPvOezhX4SZ_tEOR0qe2",
    "disney.pt": "https://drive.google.com/uc?id=1n2uQ5s2XdUBGIcZA9Uabz1mkjVvKWFeG",
}

@torch.no_grad()
def load_model(generator, model_file_path):
    ensure_checkpoint_exists(model_file_path)
    ckpt = torch.load(model_file_path, map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"], strict=False)
    return generator.mean_latent(50000)

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

def normalize(x):
    return (x+1)/2

def tensor2bbox_im(x):
    return np.array(torchvision.transforms.functional.to_pil_image(normalize(x[0])))

def prepare_bbox(boxes):
    output = []
    for i in range(len(boxes)):
        y1,x1,y2,x2 = boxes[i][0]
        output.append((256*np.array([x1,y1, x2-x1, y2-y1])).astype(np.uint8))
    return output
