import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib.patches import Patch
from torchvision.transforms import transforms
from tqdm import tqdm

import umap
from models.PreResNet import ResNet18
from models.resnet import SupCEResNet

# setup
SMALL_SIZE = 10
MEDIUM_SIZE = 18
BIG_SIZE = 20
BIGGER_SIZE = 25

plt.rc('font', size=BIG_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIG_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title
plt.rc('text', usetex=True)
plt.rcParams['axes.axisbelow'] = True


def _matplotlib_points(
        points,
        ax=None,
        labels=None,
        label_names=None,
        values=None,
        cmap="Blues",
        color_key=[None],
        color_key_cmap="Spectral",
        background="white",
        width=800,
        height=800,
        show_legend=True,
):
    """Use matplotlib to plot points"""
    point_size = 100.0 / np.sqrt(points.shape[0])
    # print(np.max(points[:, 0]), np.min(points[:, 0]), np.max(points[:, 1]), np.min(points[:, 1]))
    legend_elements = None

    if ax is None:
        dpi = plt.rcParams["figure.dpi"]
        fig = plt.figure(figsize=(width / dpi, height / dpi))
        ax = fig.add_subplot(111)

    ax.set_facecolor(background)

    # Color by labels
    if labels is not None:
        if labels.shape[0] != points.shape[0]:
            raise ValueError(
                "Labels must have a label for "
                "each sample (size mismatch: {} {})".format(
                    labels.shape[0], points.shape[0]
                )
            )
        print(color_key)
        if color_key is None:
            unique_labels = np.unique(labels)
            num_labels = unique_labels.shape[0]
            color_key = plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
            legend_elements = [
                Patch(facecolor=color_key[i], label=label_names[unique_labels[i]])
                for i, k in enumerate(unique_labels)
            ]

        if isinstance(color_key, dict):
            colors = pd.Series(labels).map(color_key)
            unique_labels = np.unique(labels)
            legend_elements = [
                Patch(facecolor=color_key[k], label=label_names[k]) for k in unique_labels
            ]
        else:
            unique_labels = np.unique(labels)
            if len(color_key) < unique_labels.shape[0]:
                raise ValueError(
                    "Color key must have enough colors for the number of labels"
                )

            new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
            legend_elements = [
                Patch(facecolor=color_key[i], label=label_names[k])
                for i, k in enumerate(unique_labels)
            ]
            colors = pd.Series(labels).map(new_color_key)
        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=colors)

    # Color by values
    elif values is not None:
        if values.shape[0] != points.shape[0]:
            raise ValueError(
                "Values must have a value for "
                "each sample (size mismatch: {} {})".format(
                    values.shape[0], points.shape[0]
                )
            )
        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=values, cmap=cmap)

    # No color (just pick the midpoint of the cmap)
    else:

        color = plt.get_cmap(cmap)(0.5)
        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=color)

    if show_legend and legend_elements is not None:
        ax.legend(handles=legend_elements)

    return ax, legend_elements


def points(
        umap_object,
        labels=None,
        label_names=None,
        theme=None,
        cmap="Blues",
        color_key=None,
        color_key_cmap="Spectral",
        background="white",
        width=800,
        height=800,
        show_legend=True,
        rot_x=False,
        rot_y=False
):
    if not hasattr(umap_object, "embedding_"):
        raise ValueError(
            "UMAP object must perform fit on data before it can be visualized"
        )

    if theme is not None:
        cmap = umap._themes[theme]["cmap"]
        color_key_cmap = umap._themes[theme]["color_key_cmap"]
        background = umap._themes[theme]["background"]

    points = umap_object.embedding_
    if rot_x:
        points[:, 0] *= -1
    if rot_y:
        points[:, 1] *= -1
    if points.shape[1] != 2:
        raise ValueError("Plotting is currently only implemented for 2D embeddings")

    dpi = plt.rcParams["figure.dpi"]
    fig = plt.figure(figsize=(width / dpi, height / dpi))
    ax = fig.add_subplot(111)

    ax, legend_elements = _matplotlib_points(
        points,
        ax,
        labels,
        label_names,
        None,
        cmap,
        color_key,
        color_key_cmap,
        background,
        width,
        height,
        show_legend,
    )

    ax.set(xticks=[], yticks=[])

    return ax, legend_elements


def export_legend(legend_elements, filename="legend.pdf"):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(handles=legend_elements)
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def create_model_selfsup(net='resnet18', dataset='cifar10'):
    checkpoint = torch.load('pretrained/ckpt_{}_{}.pth'.format(dataset, net))
    sd = {}
    for ke in checkpoint['model']:
        nk = ke.replace('module.', '')
        sd[nk] = checkpoint['model'][ke]
    model = SupCEResNet(net, num_classes=10)
    model.load_state_dict(sd, strict=False)
    model = model.to('cuda:0')
    return model


def create_model_selfsup_trained(net='resnet18', dataset='cifar10', checkpoint=''):
    checkpoint = torch.load(checkpoint)
    model = SupCEResNet(net, num_classes=10)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to('cuda:0')
    return model


def create_model_dm_trained(net='resnet18', dataset='cifar10', checkpoint=''):
    checkpoint = torch.load(checkpoint)
    model = ResNet18(num_classes=10)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to('cuda:0')
    return model


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
data = torchvision.datasets.CIFAR10('./data', train=False, transform=transform_test, target_transform=None,
                                    download=True)

val_loader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=False, num_workers=8, pin_memory=True,
                                         drop_last=False)

# checkpoint = 'cifar10_selfsup_20'  # FF
# checkpoint = 'cifar10_selfsup_90' #TT
# checkpoint = None #TF
# checkpoint = 'cifar10_dividemix_90' #FT
# checkpoint = 'cifar10_dividemix_20' #FF

checkpoint = 'warmup_cifar10_selfsup_20'  # FT
# checkpoint = 'warmup_cifar10_selfsup_90' #TF
# checkpoint = 'warmup_cifar10_dividemix_90' #TF
# checkpoint = 'warmup_cifar10_dividemix_20' # TF
if checkpoint is not None:
    if 'dividemix' in checkpoint:
        model = create_model_dm_trained(checkpoint='final_checkpoints/{}.pth.tar'.format(checkpoint))
    elif 'selfsup' in checkpoint:
        model = create_model_selfsup_trained(checkpoint='final_checkpoints/{}.pth.tar'.format(checkpoint))
else:
    checkpoint = 'pretrained'
    model = create_model_selfsup()  # SupCEResNet('resnet18', num_classes=100).cuda()
f_list, t_list = [], []
for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
    if 'dividemix' in checkpoint:
        features = model(inputs.cuda()).detach().cpu().numpy()
    elif 'selfsup' in checkpoint:
        features = model.encoder(inputs.cuda()).detach().cpu().numpy()
    f_list.append(features)
    t_list.append(targets.numpy())

features = np.concatenate(f_list)
targets = np.concatenate(t_list)

mapper = umap.UMAP(n_neighbors=20, min_dist=0.1, n_epochs=500, negative_sample_rate=10).fit(features)

colors = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
          # (70, 240, 240),
          (240, 50, 230),
          (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
          (170, 255, 195), (0, 0, 128), (128, 128, 128), (255, 255, 255), (0, 0, 0)]
colors = [[x[0] / 255., x[1] / 255., x[2] / 255.] for x in colors]
ax, legend_elements = points(mapper, labels=targets, show_legend=True, label_names=data.classes, color_key=colors,
                             rot_x=False, rot_y=False)

fig = ax.figure
export_legend(legend_elements, filename="legend.pdf")

fig.patch.set_visible(False)
ax.axis('off')
ax.get_legend().remove()
fig.tight_layout()
fig.savefig('umap_{}.pdf'.format(checkpoint))
# print('1')
