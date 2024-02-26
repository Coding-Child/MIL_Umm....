import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from util.mask import tissue_mask

import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
os.environ["OMP_NUM_THREADS"] = "1"


def dropout_white_bg(image):
    """
    Dropout the white background from an image. A pixel is considered as background
    if the value of all RGB channels are greater than the threshold.

    Args:
    - image (PIL Image or Torch Tensor): The input image.
    - threshold (int): Threshold value to consider a pixel as white background.

    Returns:
    - Torch Tensor: Image tensor with white background dropped out.
    """
    img = np.array(image)
    mask = tissue_mask(img)

    img[~mask] = 0
    image = Image.fromarray(img)

    if not TF._is_pil_image(image) and not isinstance(image, torch.Tensor):
        raise TypeError('Image should be PIL Image or torch.Tensor. Got {}'.format(type(image)))

    if TF._is_pil_image(image):
        image = TF.to_tensor(image)  # Convert PIL Image to torch Tensor

    return image


def get_embedding(model, patches):
    embeddings = list()

    model.eval()
    with torch.no_grad():
        for patch in patches:
            patch = dropout_white_bg(patch)
            patch = patch.unsqueeze(0).cuda()
            embedding = model(patch)

            embeddings.append(embedding.cpu().numpy())

    embeddings = np.vstack(embeddings)

    return embeddings


def get_pseudo_label(model, patches, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    pca = PCA(n_components=0.95, random_state=42)
    scaler = StandardScaler()

    embeddings = get_embedding(model, patches)
    embeddings = scaler.fit_transform(embeddings)
    embeddings = pca.fit_transform(embeddings)

    labels = kmeans.fit_predict(embeddings)

    return labels
