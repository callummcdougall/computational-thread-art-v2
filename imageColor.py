# ====== Includes all functions related to producing full-colour images (like the stag) ======

import numpy as np
import pandas as pd
import os
import math
import cairo
from itertools import product, groupby
import time
from collections import Counter, defaultdict
from pathlib import Path
from IPython.display import clear_output
from PIL import Image, ImageFilter, ImageDraw
from io import BytesIO
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import webcolors
import copy
from PyPDF2 import PdfReader, PdfMerger, PdfWriter
from reportlab.lib.colors import black, red, blue, gray, orange
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics 
from collections import OrderedDict
import io
import ipywidgets as wg
from IPython.display import display, HTML
import torch as t
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Union
from torchtyping import TensorType as TT
import einops

# A4 = (595.2755905511812, 841.8897637795277)
# A4 = (596, 870)

# from local_funcs.coordinates import *
from coordinates import *
from misc import *

@dataclass
class ThreadArtColorParams():
    name: str
    x: int
    n_nodes: int
    filename: str
    palette: List[List[int]]
    n_lines_per_color: List[int]
    n_random_lines: int 
    darkness: float
    blur_rad: int
    group_orders: str
    line_width_multiplier: float
    w_filename: Optional[str]
    d_coords: dict = field(default_factory=dict)
    d_pixels: dict = field(default_factory=dict)
    d_joined: dict = field(default_factory=dict)
    d_sides: dict = field(default_factory=dict)
    t_pixels: t.Tensor = field(default_factory=lambda: t.Tensor())
    n_consecutive: int = 0
    shape: str = "Rectangle"
    seed: int = 0
    pixels_per_batch: int = 32
    num_overlap_rows: int = 6

    def __post_init__(self):

        self.color_names = list(self.palette.keys())
        self.color_values = list(self.palette.values())
        first_color_letters = [color[0] for color in self.color_names]
        assert len(set(first_color_letters)) == len(first_color_letters), "First letter of each color name must be unique."
        
        assert os.path.exists("images/" + self.filename), f"Image {self.filename} not found"
        img_raw = Image.open("images/" + self.filename)
        self.y = int(self.x * (img_raw.height / img_raw.width))

        if self.shape.lower() in ["circle", "ellipse", "round"]:
            self.shape = "Ellipse"
        else:
            self.shape = "Rectangle"
        
        self.d_coords, self.d_pixels, self.d_joined, self.d_sides, self.t_pixels = build_through_pixels_dict(
            self.x, self.y, self.n_nodes, shape=self.shape, critical_distance=14, only_return_d_coords=False, width_to_gap_ratio=1
        )

        if len(self.group_orders) == 1 and self.group_orders.isdigit():
            self.group_orders = "".join([str(i) for i in range(len(self.palette))]) * int(self.group_orders)

        self.img_dict = dict(
            x=self.x, 
            y=self.y, 
            filename=self.filename, 
            d_pixels=self.d_pixels, 
            palette=self.palette, 
            w_filename=self.w_filename, 

            wneg_filename=None,
            other_colors_weighting=0,
            dithering_params=["clamp"],
            pixels_per_batch=self.pixels_per_batch, 
            num_overlap_rows=self.num_overlap_rows
        )

    def get_img(self):
        return Img(**self.img_dict)

    def __repr__(self):
        for k, v in self.__dict__.items():
            if isinstance(v, t.Tensor):
                print(f"{k:>22} : tensor of shape {tuple(v.shape)}")
            elif isinstance(v, dict):
                print(f"{k:>22} : dict of length {len(v)}")
            elif k == "palette":
                s = f"<code>{'&nbsp;' * 13}palette : </code>" + palette_to_html(v.keys())
                display(HTML(s))
            elif k == "group_orders":
                color0_to_color = {color[0]: i for i, color in enumerate(self.palette.keys())}
                color_list = [self.palette[color0_to_color[char]] for char in v]
                s = f"<code>{'&nbsp;' * 8}group_orders : </code>" + palette_to_html(color_list)
                display(HTML(s))
            elif not(k.startswith("color")):
                print(f"{k:>22} : {v}")
        return ""

# ===================================================================================================

# Class for images: contains Floyd-Steinberg dithering image function, histogram of colours, different versions of the image, etc
class Img:

    def __init__(self, x, y, filename, d_pixels=None, palette=None, w_filename=None, wneg_filename=None, other_colors_weighting=0, dithering_params=["clamp"], pixels_per_batch=None, num_overlap_rows=None):

        t0 = time.time()

        self.filename = "images/{}".format(filename)
        self.x = x
        self.y = y
        self.palette: Dict[str, Tuple[int, int, int]] = {color_name: tuple(color_value) for color_name, color_value in palette.items()}
        self.dithering_params = dithering_params

        base_image = Image.open(self.filename).resize((self.x, self.y))
        self.imageRGB = t.tensor((base_image).convert(mode="RGB").getdata()).reshape((self.y, self.x, 3))
        self.imageBW = t.tensor((base_image).convert(mode="L").getdata()).reshape((self.y, self.x))
        
        t_FS = 0
        if palette:
            self.image_dithered, t_FS = self.FS_dither(pixels_per_batch, num_overlap_rows)
            self.color_histogram, self.mono_images_dict = self.generate_mono_images_dict(d_pixels, other_colors_weighting)
        
        if w_filename:
            self.w_filename = "images/{}".format(w_filename)
            base_image_w = Image.open(self.w_filename).resize((self.x, self.y))
            self.w = 1 - (t.tensor((base_image_w).convert(mode="L").getdata()).reshape((self.y, self.x)) / 255)
        else:
            self.w = None

        if wneg_filename:
            self.wneg_filename = "images/{}".format(wneg_filename)
            base_image_wneg = Image.open(self.wneg_filename).resize((self.x, self.y))
            self.wneg = 1 - (t.tensor((base_image_wneg).convert(mode="L").getdata()).reshape((self.y, self.x)) / 255)
        else:
            self.wneg = None

        print(f"Other init operations complete in {time.time() - t0 - t_FS:.2f} seconds")

    # Performs FS-dithering with progress bar, returns the output (called in __init__)
    def FS_dither(self, pixels_per_batch, num_overlap_rows) -> Tuple[t.Tensor, float]:
        '''
        Currently, this doesn't implement the overlapping subsets idea. Will do that next!
        '''
        t_FS_misc = time.time()

        image_dithered: TT["y", "x", 3] = self.imageRGB.clone().to(t.float)
        y, x, _ = image_dithered.shape

        if pixels_per_batch is None:
            return self.FS_dither_batch(image_dithered.unsqueeze(-2))

        num_batches = math.ceil(y / pixels_per_batch)
        rows_to_extend_by = num_batches - (y % num_batches)

        # Add a batch dimension
        image_dithered = einops.rearrange(
            t.concat([image_dithered, t.zeros(rows_to_extend_by, self.x, 3)]), 
            "(batch y) x rgb -> y x batch rgb", 
            batch=num_batches
        )
        # Concat the last `num_overlap_rows` to the start of the image
        end_of_each_batch = t.concat([
            t.zeros(num_overlap_rows, x, 1, 3),
            image_dithered[-num_overlap_rows:, :, :-1],
        ], axis=-2)
        image_dithered = t.concat([end_of_each_batch, image_dithered], dim=0) 

        # # Plot slice images, to check that this is working correctly
        # for batch_no in range(image_dithered.size(2)):
        #     i = image_dithered[:, :, batch_no, :]
        #     px.imshow(i).show()

        image_dithered, t_FS = self.FS_dither_batch(image_dithered)

        image_dithered = einops.rearrange(
            image_dithered[num_overlap_rows-1: -1],
            "y x batch rgb -> (batch y) x rgb"
        )[1:y+1]

        t_FS_misc = time.time() - t_FS_misc - t_FS
        print(f"FS dithering (wrapper) complete in {t_FS_misc:.2f}s")

        return image_dithered, t_FS + t_FS_misc
        
    def FS_dither_batch(self, image_dithered: TT["y", "x", "batch", 3]) -> Tuple[t.Tensor, float]:

        AB = t.tensor([3, 5]) / 16
        ABC = t.tensor([3, 5, 1]) / 16
        BC = t.tensor([5, 1]) / 16

        is_clamp = ("clamp" in self.dithering_params)
        palette: TT["color", 3] = t.tensor(list(self.palette.values())).to(t.float)
        palette_sq = einops.rearrange(palette, "color rgb -> color 1 rgb")
        t0 = time.time()
        y, x, batch = image_dithered.shape[:3]

        # loop over each row, from first to second last
        for y_ in tqdm(range(y - 1), desc="Floyd-Steinberg dithering"):
            
            row: TT["x", "batch", 3] = image_dithered[y_].to(t.float)
            next_row: TT["x", "batch", 3] = t.zeros_like(row)

            # deal with the first pixel in the row
            old_color: TT["batch", 3] = row[0]
            # Will this broadcast correctly?
            color_diffs: TT["color", "batch"] = (palette_sq - old_color).pow(2).sum(axis=-1)
            color: TT["batch", 3] = palette[color_diffs.argmin(dim=0)]
            color_diff: TT["batch", 3] = old_color - color
            row[0] = color
            row[1] += (7/16) * color_diff
            next_row[[0, 1]] += einops.einsum(BC, color_diff, "two, batch rgb -> two batch rgb")
            
            # loop over each pixel in the row, from second to second last
            for x_ in range(1, self.x-1):
                old_color = row[x_]
                color_diffs = (palette_sq - old_color).pow(2).sum(axis=-1)
                color = palette[color_diffs.argmin(dim=0)]
                color_diff = old_color - color
                row[x_] = color
                row[x_+1] += (7/16) * color_diff
                next_row[[x_-1, x_, x_+1]] += einops.einsum(ABC, color_diff, "three, batch rgb -> three batch rgb")

            # deal with the last pixel in the row
            old_color = row[-1]
            color_diffs = (palette_sq - old_color).pow(2).sum(axis=-1)
            color = palette[color_diffs.argmin(dim=0)]
            color_diff = old_color - color
            row[-1] = color
            next_row[[-2, -1]] += einops.einsum(AB, color_diff, "two, batch rgb -> two batch rgb")

            # update the rows, i.e. changing current row and propagating errors to next row
            image_dithered[y_] = t.clamp(row, 0, 255)
            image_dithered[y_+1] += next_row
            if is_clamp: image_dithered[y_+1] = t.clamp(image_dithered[y_+1], 0, 255)

        # deal with the last row
        row = image_dithered[-1]
        for x_ in range(self.x-1):
            old_color = row[x_]
            color_diffs = (palette_sq - old_color).pow(2).sum(axis=-1)
            color = palette[color_diffs.argmin(dim=0)]
            color_diff = old_color - color
            row[x_] = color
            row[x_+1] += color_diff
            
        # deal with the last pixel in the last row
        old_color = row[-1]
        color_diffs = (palette_sq - old_color).pow(2).sum(axis=-1)
        color = palette[color_diffs.argmin(dim=0)]
        row[-1] = color
        if is_clamp: row = t.clamp(row, 0, 255)
        image_dithered[-1] = t.tensor(row)

        clear_output()
        t_FS = time.time() - t0
        print(f"FS dithering complete in {t_FS:.2f}s")

        return image_dithered.to(t.int), t_FS

    # Displays image output
    def display_output(self, height: int, width: int):
        px.imshow(self.imageRGB.to(t.float), height=height, width=width, template="plotly_dark").show()
        fig = px.imshow(
            t.stack(list(self.mono_images_dict.values())), height=height, width=width, template="plotly_dark", title="Images per color",
            color_continuous_scale="gray", animation_frame=0
        ).update_layout(coloraxis_showscale=False)
        fig.layout.sliders[0].currentvalue.prefix = "color = "
        for i, color_name in enumerate(self.palette.keys()):
            fig.layout.sliders[0].steps[i].label = color_name
        fig.show()

    # Takes FS output and returns a dictionary of monochromatic images (called in __init__)
    def generate_mono_images_dict(self, d_pixels, other_colors_weighting):

        # gets the pixels which are actually relevant
        boolean_mask = t.zeros(size=self.image_dithered.shape[:-1])
        pixels_y_all, pixels_x_all = list(zip(*d_pixels.values()))
        pixels_y_all = t.concat(pixels_y_all).long()
        pixels_x_all = t.concat(pixels_x_all).long()
        boolean_mask[pixels_y_all, pixels_x_all] = 1
        
        d_histogram = dict()           # histogram of frequency of colors (cropped to a circle if necessary)
        d_mono_images_pre = dict()     # mono-color images, before they've been processed
        d_mono_images_post = dict()    # mono-color images, after processing (i.e. adding weight to nearby colors)
        
        # loop through each color, creating the pre-processing mono images, and the histogram
        for color_name, color_value in self.palette.items():
            # mono_image = np.apply_along_axis(lambda pixel_col: tuple(pixel_col) == color, 2, self.image_dithered).astype(int)
            mono_image = (get_img_hash(self.image_dithered) == get_color_hash(t.tensor(color_value))).to(t.int)
            d_mono_images_pre[color_name] = mono_image
            d_histogram[color_name] = (mono_image * boolean_mask).sum() / boolean_mask.sum()
        
        # apply post-processing, if necessary
        if other_colors_weighting == 0:
            return d_histogram, d_mono_images_pre
        else:   
            for c0 in self.palette:
                d_mono_images_post[c0] = np.full_like(self.image_dithered, 0)
                for c1 in self.palette:
                    color_distance = (np.subtract(c0, c1) ** 2).sum() ** 0.5
                    scale_factor = 1 - (color_distance / (255 * (3 ** 0.5)))
                    d_mono_images_post[c1] += scale_factor * d_mono_images_pre[c1]
            return d_histogram, d_mono_images_post

    # Prints a suggested number of lines, in accordance with histogram frequencies (used in Juypter Notebook)
    def decompose_image(self, n_lines_total=10000):

        n_lines_per_color = [int(self.color_histogram[color] * n_lines_total) for color in self.palette]
        darkest_idx = [i for i, (color_name, color_values) in enumerate(self.palette.items()) if sum(color_values) == max([sum(cv) for cv in self.palette.values()])][0]
        n_lines_per_color[darkest_idx] += (n_lines_total - sum(n_lines_per_color))

        max_len_color_name = max(len(color_name) for color_name in self.palette.keys())
        for idx, (color_name, color_value) in enumerate(self.palette.items()):
            color_string = str(tuple(color_value))
            color_string = color_string + "&nbsp;" * (18 - len(color_string))
            color_name = color_name + "&nbsp;" * (max_len_color_name + 3 - len(color_name))
            s = f"<code>{color_string}{color_name}</code><text style='color:rgb{tuple(color_value)}'>████████</text><code> = {n_lines_per_color[idx]}</code>"
            display(HTML(s))

        print(f"`n_lines_per_color` for you to copy: {n_lines_per_color}")


# Displays a list of images in one row, similar but slightly different to the `imageBW.py` function (used by function below)
def display_img(im_list, width):
        
    for im_sublist in im_list:
    
        fig = make_subplots(rows=1, cols=len(im_sublist), horizontal_spacing=0.016)

        for idx, im in enumerate(im_sublist):
            
            if isinstance(im, t.Tensor):
                im = im.numpy()

            if isinstance(im, np.ndarray):
                im = Image.fromarray(im.astype(np.uint8))

            prefix = "data:image/png;base64,"
            with BytesIO() as stream:
                im.save(stream, format="png")
                base64_string = prefix + base64.b64encode(stream.getvalue()).decode("utf-8")

            fig.add_trace(go.Image(source=base64_string), row=1, col=idx+1)
            
        height = width * (im.height / im.width) * (1 / len(im_sublist))
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), width=width, height=height)
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

        fig.show(config={'doubleClick': 'reset', 'displayModeBar': False})


# Displays a grid of all the relevant images: original, monochrome, dithered, color-decomposed, and optionally the weighting (used in Jupyter Notebook)
def display_splashpage(I, size, w=False, d_coords=None, offset_pixels=1, coord_color=(255, 0, 0)):

    if w and I.w is None:
        print("You can't have `w=True` if `I` has no weighting.")
        return None

    offset_list = list(range(-offset_pixels, offset_pixels+1))

    img_list = [[I.imageRGB, I.imageBW]]

    if w:
        img_list.append([hsv_to_rgb_image(I)])

    if type(I.palette) != type(None):
        img_dithered = I.image_dithered.clone()

        # This next bit of code highlights the nodes in red - useful check if I'm not using a rectangular shape for them
        if d_coords is not None:
            for idx, (y, x) in d_coords.items():
                y, x = int(y), int(x)
                for y_offset, x_offset in product(offset_list, repeat=2):
                    y_true = max(min(y + y_offset, I.y - 1), 0)
                    x_true = max(min(x + x_offset, I.x - 1), 0)
                    img_dithered[y_true, x_true, :] = t.tensor(coord_color)

        img_list.append([img_dithered])
        # img_list.append([blur_image((255 * i).to(t.int), rad=1) for i in I.mono_images_dict.values()])

    display_img(img_list, size)


# Blurs the monochromatic images (used in the function below)
def linear_blur_image(image: t.Tensor, rad: int, threeD=False):
    
    if rad == 0:
        return image
    
    # define the matrix, and normalise it
    mat = t.zeros(2*rad+1, 2*rad+1)
    std = 0.5*rad
    for x in range(2*rad+1):
        x_offset = x - rad
        for y in range(2*rad+1):
            y_offset = y - rad
            value = 1 - (abs(y_offset)+abs(x_offset))/(2*rad+1) # 1/(abs(y_offset)+abs(x_offset)+1)
            mat[y, x] = value
    mat = mat / mat.sum()
    
    # create a canvas larger than the image, then loop over the matrix adding the appropriate copies of the canvas

    if threeD:
        image_size_y, image_size_x, _ = image.shape
        canvas_size_y = image.size(0) + 2*rad
        canvas_size_x = image.size(1) + 2*rad
        canvas = t.zeros(canvas_size_y, canvas_size_x, 3)
        for x in range(2*rad+1):
            for y in range(2*rad+1):
                canvas[y:y+image_size_y, x:x+image_size_x, :] += mat[y, x] * image
    else:
        image_size_y, image_size_x = image.shape
        canvas_size_y = image.size(0) + 2*rad
        canvas_size_x = image.size(1) + 2*rad
        canvas = t.zeros(canvas_size_y, canvas_size_x)
        for x in range(2*rad+1):
            for y in range(2*rad+1):
                canvas[y:y+image_size_y, x:x+image_size_x] += mat[y, x] * image
    
    # crop the canvas, and return it
    return canvas[rad : -rad, rad : -rad]


# Performs either linear blurring (image above) or Gaussian (used in `create_canvas` function, for image processing before creating output)
def blur_image(img: t.Tensor, rad: int, mode="linear", **kwargs):

    if mode == "linear":
        return linear_blur_image(img, rad, **kwargs)
    
    elif mode == "gaussian":
        return t.from_numpy(np.asarray(Image.fromarray(img.astype("uint8")).filter(ImageFilter.GaussianBlur(radius=rad)))).to(t.float)


# Generates a bunch of random lines and chooses the best one
def choose_and_subtract_best_line(m_image: t.Tensor, i: int, w: Optional[t.Tensor], n_random_lines: List[int], darkness: float, d_joined: dict, t_pixels: t.Tensor) -> int:

    j_random = t.from_numpy(np.random.choice(d_joined[i], min(len(d_joined[i]), n_random_lines), replace=False))

    coords_yx: TT["j", "yx": 2, "pixels"] = t_pixels[i, j_random.long()].long()
    is_zero: TT["j", "pixels"] = (coords_yx.sum(1) == 0)
    coords_yx: TT[2, "j_and_pixels"] = einops.rearrange(coords_yx, "j yx pixels -> yx (j pixels)")

    pixels_in_lines: TT["j_and_pixels"] = m_image[coords_yx[0], coords_yx[1]]
    pixels_in_lines: TT["j", "pixels"] = einops.rearrange(
        pixels_in_lines, "(j pixels) -> j pixels", 
        j=j_random.size(0)
    ).masked_fill(is_zero, 0)

    if isinstance(w, t.Tensor):
        weighting_of_pixels_in_lines: TT["j", "pixels"] = einops.rearrange(
            w[coords_yx[0], coords_yx[1]],
            "(j pixels) -> j pixels", j=j_random.size(0)
        ).masked_fill(is_zero, 0)

        w_sum: TT["j"] = weighting_of_pixels_in_lines.sum(dim=-1)
        scores: TT["j"] = (pixels_in_lines * weighting_of_pixels_in_lines).sum(-1) / w_sum

    else:
        lengths: TT["j"] = (~is_zero).sum(-1)
        scores: TT["j"] = pixels_in_lines.sum(-1) / lengths

    best_j = j_random[scores.argmax()].item()

    coords_yx: TT[2, "pixels"] = t_pixels[i, best_j].long()
    is_zero: TT["pixels"] = (coords_yx.sum(0) == 0)
    coords_yx = coords_yx[:, ~is_zero]
    m_image[coords_yx[0], coords_yx[1]] -= darkness

    return best_j

def create_canvas(I: Img, args: ThreadArtColorParams):
    """
    n_consecutive is to break up the lines, so they don't reveal IP when sent as an svg
    full_sweep_freq gives me stats (currently it's set up to return stats that generate line plots of score as you move one end of a line, trying to understand conceptually how alg works)
    """

    assert len(I.palette) == len(args.n_lines_per_color), "Palette and lines per color don't match. Did you change the palette without re-updating params?"
    
    t0 = time.time()
    line_dict = dict()
    w = I.w

    if type(args.darkness) in [float, np.float64]:
        darkness_dict = defaultdict(lambda: args.darkness)
    else:
        darkness_dict = args.darkness
    
    mono_image_dict = {color: blur_image(mono_image, args.blur_rad) for color, mono_image in I.mono_images_dict.items()}
    
    # setting a random seed at the start of this function ensures the lines will be the same (unless the parameters change)
    t.manual_seed(args.seed)
    np.random.seed(args.seed)

    max_color_name_len = max(len(i) for i in I.palette)
    
    for i, (color_name, color_value) in enumerate(I.palette.items()):

        n_lines = args.n_lines_per_color[i]
        m_image = mono_image_dict[color_name]
        line_dict[color_name] = []
        i = list(args.d_joined.keys())[t.randint(0, len(args.d_joined), (1,))]
          
        for n in range(n_lines): #, desc=f"Progress for color {color}"): #, leave=False):

            # Choose and add line
            j = choose_and_subtract_best_line(m_image, i, w, args.n_random_lines, darkness_dict[color_name], args.d_joined, args.t_pixels)
            line_dict[color_name].append((i, j))
            # Get the outgoing node
            i = j+1 if (j % 2 == 0) else j-1

            if args.n_consecutive != 0 and ((n+1) % args.n_consecutive) == 0:
                i = list(args.d_joined.keys())[t.randint(0, len(args.d_joined), (1,))]

            if n+1 % 50 == 0:
                print(f"Color {color_name+',':{max_color_name_len+1}} line {n+1:4}/{n_lines:<4} done.", end="\r")

        print(f"Color {color_name+',':{max_color_name_len+1}} line {n_lines:4}/{n_lines:<4} done.")

    print(f"total time = {time.time() - t0:.2f}")

    return line_dict




# Takes the line_dict, and uses it to create an svg of the output, then saves it
def paint_canvas(
    line_dict: dict,
    I: Img,
    args: ThreadArtColorParams,
    mode: str = "svg",
    filename_override: Optional[str] = None,
    rand_perm: float = 0.0025,
    fraction: Union[Tuple, Dict] = (0, 1),
    background_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
    show_individual_colors: bool = False,
    img_width=800,
    sf=8,
    verbose: bool = False,
):

    assert mode == "svg", "Only svg mode is supported right now."

    # if fraction != (0, 1), it means you're not plotting all the lines, only a subset of them - useful to see how many lines are actually needed to make the image look good
    # precise syntax: fraction = (a, b) means you're plotting between the a and bth lines, e.g. (0, 0.5) means the best half
    if type(fraction) == tuple:
        line_dict_ = {
            color: lines[int(fraction[0] * len(lines)):int(fraction[1] * len(lines))]
            for color, lines in line_dict.items()
        }
    else:
        line_dict_ = {
            color: lines[int(fraction.get(color, (0, 1))[0] * len(lines)):int(fraction.get(color, (0, 1))[1] * len(lines))]
            for color, lines in line_dict.items()
        }
    

    # find a name to save the file with
    if type(filename_override) == type(None):
        counter = 1
        while True:
            img_filename = f"outputs/{args.name}_{counter:02}.{mode}"
            if not Path(img_filename).exists():
                break
            counter += 1
    else:
        img_filename = f"outputs/{filename_override}.{mode}"
    
    # change the group orders to string, if they're integers
    group_orders = args.group_orders
    if type(group_orders) in [int, np.int64]:
        single_color_batch = "".join([color_name[0] for color_name in I.palette])
        group_orders = single_color_batch * group_orders
    # change the group orders to indices of colours
    if type(group_orders) == str:
        if group_orders[0].isdigit():
            group_orders = [int(i) for i in group_orders]
        else:
            d = {color_name[0]: idx for idx, (color_name, color_value) in enumerate(I.palette.items())}
            group_orders = [d[char] for char in group_orders]

    print(f"Saving to {img_filename!r}")
    
    with cairo.SVGSurface(img_filename, I.x, I.y) as surface:
                    
        context = cairo.Context(surface)
        context.scale(I.x, I.y)
        context.set_line_width(0.0002 * args.line_width_multiplier)

        if background_color is None:
            context.set_source_rgba(0.0, 0.0, 0.0, 0.0)
        else:
            context.set_source_rgb(*[c/255 for c in background_color])
        context.paint()
        
        for (i_idx, i) in enumerate(group_orders):
            
            color_name = list(I.palette.keys())[i]
            color_value = list(I.palette.values())[i]
            lines = line_dict_[color_name]
            
            context.set_source_rgb(*[c/255 for c in color_value])

            n_groups = len([j for j in group_orders if j == i])
            group_order = len([j for j in group_orders[:i_idx] if j == i])

            n = int(len(lines) / n_groups)
            lines_to_draw = lines[::-1]
            lines_to_draw = lines_to_draw[n*group_order : n*(group_order+1)]
            
            if verbose: print(f"{i_idx+1:2}/{len(group_orders)}: {len(lines_to_draw):4} {color_name}")

            current_node = -1

            for line in lines_to_draw:

                starting_node = line[1]
                if starting_node != current_node:
                    y, x = args.d_coords[starting_node] / t.tensor([I.y, I.x])
                    y, x = hacky_permutation(y, x, rand_perm)
                    context.move_to(x, y)

                finishing_node = line[0]
                y, x = args.d_coords[finishing_node] / t.tensor([I.y, I.x])
                y, x = hacky_permutation(y, x, rand_perm)
                context.line_to(x, y)

                current_node = finishing_node

            context.stroke()


    if show_individual_colors:

        for color_name, color_value in I.palette.items():

            lines = line_dict_[color_name]
        
            with cairo.SVGSurface(img_filename.replace(".", f"_{color_name}."), I.x, I.y) as surface:
                            
                context = cairo.Context(surface)
                context.scale(I.x, I.y)
                context.set_line_width(0.0002 * args.line_width_multiplier)

                if sum(color_value) > 255 * 2 - 5:
                    context.set_source_rgba(0.0, 0.0, 0.0, 1.0)
                else:
                    context.set_source_rgba(1.0, 1.0, 1.0, 0.0)
                context.paint()
                    
                context.set_source_rgb(*[c/255 for c in color_value])

                current_node = -1

                for line in lines:

                    starting_node = line[1]
                    if starting_node != current_node:
                        y, x = args.d_coords[starting_node] / t.tensor([I.y, I.x])
                        y, x = hacky_permutation(y, x, rand_perm)
                        context.move_to(x, y)

                    finishing_node = line[0]
                    y, x = args.d_coords[finishing_node] / t.tensor([I.y, I.x])
                    y, x = hacky_permutation(y, x, rand_perm)
                    context.line_to(x, y)

                    current_node = finishing_node

                context.stroke()



# Permutes coordinates, to stop weird-looking line pattern effects (used by `paint_canvas` function)
def hacky_permutation(y, x, r):
    
    R = r * (2 * np.random.random() - 1)
    
    if (x < 0.01) or (x > 0.99):
        return y + R, x
    else:
        return y, x + R


# Generate instructions for physically creating artwork (assuming rectangular shape)        
def generate_instructions_pdf(line_dict, I: Img, args: ThreadArtColorParams, font_size, num_cols, num_rows, true_x, show_stats=True, version="n+1", true_thread_diameter=0.25):

    try:
        font_file = 'lines/courier-prime.regular.ttf'
        prime_font = TTFont('Courier Prime', font_file)
        pdfmetrics.registerFont(prime_font)
        using_font = True
    except:
        using_font = False

    # A4 = (596, 870)
    width = A4[0]
    height = A4[1]
    
    # store instructions, in printable form
    lines = []
    
    # function which takes a node (between 0 and n_nodes), and returns instructions (i.e. side, node identifier, and parity)
    # Note that we differentiate between ellipse and rectangle here
    def format_node(node_idx):
        node = int(node_idx / 2)
        parity = node_idx % 2
        return f"{node // 10 : 3} {node % 10}", str(parity)

    total_nlines_so_far = 0
    total_nlines = sum([len(value) for value in line_dict.values()])

    if args.group_orders[0].isdigit():
        group_orders = args.group_orders
    else:
        d = {color_name[0]: str(idx) for idx, (color_name, color_value) in enumerate(I.palette.items())}
        group_orders = "".join([d[char] for char in args.group_orders])

    idx = 0
    curr_color = "not_a_color"
    color_count = {color: [0, 0] for color in set(group_orders)}
    group_len_list = []
    while idx < len(group_orders):
        if group_orders[idx] == curr_color:
            group_len_list[-1][1] += 1
        else:
            group_len_list.append([group_orders[idx], 1])
            curr_color = group_orders[idx][0]
            color_count[group_orders[idx]][1] += 1
        idx += 1
    
    # group_len_list looks like [['0', 3], ['3', 1], ['1', 2], ... for ███ █ ██ ...
    for idx, (color_idx, group_len) in enumerate(group_len_list):
        
        # figures out how many groups of this particular colour there are, and which group we're currently on
        n_groups = sum([j[1] for j in group_len_list if j[0] == color_idx])
        group_start = sum([j[1] for j in group_len_list[:idx] if j[0] == color_idx])
        group_end = group_start + group_len
        
        # gets the color and color description, from the first letter of the color description (i.e. color_idx="0" gets (0,0,0), "black")
        color_name = list(I.palette.keys())[int(color_idx)]
        
        # get the line range we need to draw
        lines_colorgroup = line_dict[color_name]
        line_range = get_range_of_lines(len(lines_colorgroup), n_groups, (group_start, group_end))

        # extend the dictionary of instructions, with titles and instructions
        color_count[color_idx][0] += 1
        thiscolor_group = f"{color_count[color_idx][0]}/{color_count[color_idx][1]}"
        lines.extend(["=================", f"ByNow = {total_nlines_so_far}/{total_nlines}", f"ByEnd = {total_nlines_so_far + len(line_range)}/{total_nlines}", f"NOW   = {color_name} {thiscolor_group}", "================="])
        if group_start == 0:
            lines.append(format_node(lines_colorgroup[-1][-1]))
        for idx_ in line_range:
            lines.append(format_node(lines_colorgroup[-idx_][0]))
        lines.extend(["=================", f"DONE  = {color_name} {thiscolor_group}"])
        total_nlines_so_far += len(line_range)

    group_page_list = []
    page_counter = 0
    
    while len(lines) > 0:
        
        next_lines, lines = lines[:num_rows*num_cols], lines[num_rows*num_cols:]
    
        filename = f"lines/lines-{args.name}-{page_counter}.pdf"
        canvas = Canvas(filename, pagesize=A4)
        
        page_counter += 1

        canvas.setLineWidth(0.4)
        canvas.setStrokeGray(0.8)
        canvas.setStrokeColor(gray)

        for col_no in range(num_cols):
            for row_no in range(num_rows):
                
                x = 0.5*cm + col_no*(width / num_cols)
                y = height * (1 - (1 + row_no) / (0.6 + num_rows))
                if next_lines:
                    next_line = next_lines.pop(0)
                else:
                    break

                if isinstance(next_line, str):

                    to = canvas.beginText()
                    if using_font:
                        to.setFont("Courier Prime", 14 if num_cols == 3 else 18)
                    to.setTextOrigin(x, y)
                    to.setFillColor(black)
                    to.textLine(next_line)
                    canvas.drawText(to)

                    if next_line.startswith("NOW"):
                        group_page_list.append([page_counter, next_line[6:]])

                else:
                    
                    tens, units = next_line

                    to = canvas.beginText()
                    if using_font:
                        to.setFont("Courier Prime", font_size) # "Symbola"

                    to.setTextOrigin(x, y)
                    to.setFillColor(black)
                    to.textLine(tens)

                    to.setTextOrigin(x+2.9*(font_size/20)*cm, y)
                    to.setFillColor(blue)
                    to.textLine(units)

                    canvas.drawText(to)

                    canvas_path = canvas.beginPath()
                    canvas_path.moveTo(x, y-.25*(font_size/20)*cm)
                    canvas_path.lineTo(x+3.65*(font_size/20)*cm, y-.25*(font_size/20)*cm)
                    canvas.drawPath(canvas_path, fill=0, stroke=1)
                
        canvas.setLineWidth(3)
        canvas.setStrokeGray(0.5)
        canvas.setStrokeColor(black)
        canvas_path = canvas.beginPath()
        canvas_path.moveTo(width/num_cols, 0)
        canvas_path.lineTo(width/num_cols, height)
        canvas_path.moveTo(2*width/num_cols, 0)
        canvas_path.lineTo(2*width/num_cols, height)

        canvas.drawPath(canvas_path, fill=0, stroke=1)

        canvas.save()

    merger = PdfMerger()
    for g in range(page_counter):
        filename = f"lines/lines-{args.name}-{g}.pdf"
        with open(filename, "rb") as f:
            merger.append(PdfReader(f))
        os.remove(filename)
    for idx, (pagenum, desc) in enumerate(group_page_list):
        merger.add_outline_item(title=f"{idx + 1}/{len(group_page_list)} {desc}", pagenum=pagenum-1)
        # .add_outline_item
    
    if version is None:
        pdf_filename = f"lines/lines-{args.name}.pdf"
    elif isinstance(version, int):
        pdf_filename = f"lines/lines-{args.name}-{version:02d}.pdf"
    elif version == "n+1":
        version = 1
        while os.path.exists(f"lines/lines-{args.name}-{version:02d}.pdf"):
            version += 1
        pdf_filename = f"lines/lines-{args.name}-{version:02d}.pdf"
    else:
        raise Exception("Invalid version argument. Should be None, int, or 'n+1'.")
    merger.write(pdf_filename)
    print(f"Wrote to {pdf_filename!r}")
        
    if show_stats:
        
        df_dicts = {}
        for color_name, lines in line_dict.items():
            nodes = [i[0]//2 for i in lines]
            counter = Counter(nodes)
            node_frequencies = np.array([counter.get(i, 0) for i in range(max(args.d_coords) // 2)])
            node_frequencies_averaged = np.convolve(node_frequencies, np.ones(4), 'valid') / 4
            df_dicts[color_name] = node_frequencies_averaged

        fig = px.line(pd.DataFrame(df_dicts), labels={"index": "node-pair", "value": "# lines", "variable": "Color"})
        fig.update_layout(template="ggplot2", width=800, height=450, margin=dict(t=60,r=30,l=30,b=40), title_text="Frequencies of lines at nodes, by color")
        fig.show()

        max_colorname_len = max(len(color_name) for color_name in line_dict.keys())
        image_x = max(coord[1] for coord in args.d_coords.values())
        image_y = max(coord[0] for coord in args.d_coords.values())
        sf = true_x / image_x
        true_y = image_y * sf
        for color_name, lines in line_dict.items():
            total_distance = sum([dist(args.d_coords[line[0]], args.d_coords[line[1]]) for line in lines])
            print(f"{color_name:{max_colorname_len}} | {total_distance*sf/1000:.2f} km")

        n_buckets = 200

        max_len = 0
        for lines in line_dict.values():
            for line in lines:
                line_len = int(args.d_pixels[tuple(sorted(line))].shape[1])
                max_len = max(max_len, line_len)

        true_area_covered_by_thread = 0
        df_dicts = {}
        for color_name, lines in line_dict.items():
            df_dicts[color_name] = [0 for i in range(n_buckets + 1)]
            for line in lines:
                line_len = dist(args.d_coords[line[0]], args.d_coords[line[1]])
                true_area_covered_by_thread += (line_len * (sf * 1000)) * true_thread_diameter
                line_len_bucketed = int(line_len * n_buckets / max_len)
                df_dicts[color_name][line_len_bucketed] += 1

        df_from_dict = pd.DataFrame(df_dicts, index=np.arange(n_buckets+1) * 100/n_buckets)
                
        fig = px.line(df_from_dict, labels={"index": "distance (as % of max)", "value": "# lines", "variable": "Color"})
        fig.update_layout(template="ggplot2", width=800, height=450, margin=dict(t=60,r=30,l=30,b=40), title_text="Distance of lines, by color")

        fig.add_vline(x=100*min(image_x, image_y)/max_len, line_width=3)
        if args.shape == "Rectangle":
            fig.add_vline(x=100*max(image_x, image_y)/max_len, line_width=3)
            fig.add_trace(go.Scatter(
                x=[100*min(image_x, image_y)/max_len + 3, 100*max(image_x, image_y)/max_len + 3],
                y=[0.9 * df_from_dict.values.max(), 0.9 * df_from_dict.values.max()],
                text=["x", "y"] if image_x < image_y else ["y", "x"],
                mode="text"
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[100*image_x/max_len + 3], y=[0.9 * df_from_dict.values.max()],
                text=["r"], mode="text"
            ))
        fig["data"][-1]["showlegend"] = False
        fig.show()

        true_area = 10e6 * true_x * true_y
        if args.shape == "Ellipse":
            true_area *= math.pi / 4
        print(f"Total area covered (counting overlaps) = {true_area_covered_by_thread/true_area:.3f}")
        print(r"\nBaseline:\nMDF: around 0.2 is enough. The stag_large sent to Ben was 0.209. This is with true diameter = 0.25, width 120cm, 10000 lines total. And I think down to 0.2 wouldn't have fucked it up, but smaller values might have.\nWheel: 0.264 is what I used for stag, probably 0.3 would have been better (it's got a transparent background!). This would mean about 7200 threads for a standard wheel.")






def create_list_of_all_lines(line_dict, args: ThreadArtColorParams):

    all_lines = []

    color_names = list(args.palette.keys())
    color_counters = {k: 0 for k in color_names}

    for g in args.group_orders:
        num_instances = len([c for c in args.group_orders if c == g])
        matching_color = [c for c in color_names if c[0] == g][0]
        color_value = args.palette[matching_color]
        color_counters[matching_color] += 1
        color_len = len(line_dict[matching_color])
        start = int(color_len * (color_counters[matching_color] - 1) / num_instances)
        end = int(color_len * color_counters[matching_color] / num_instances)
        next_lines = line_dict[matching_color][start: end]
        for line in next_lines:
            all_lines.append((line, color_value))

    return all_lines



def render_animation(
    I: Img,
    args: ThreadArtColorParams,
    line_dict: dict,
    x_output,
    gif_duration,
    n_frames_total,
    rand_perm=0.0015,
    background_color=(255,255,255),
    d_coords_output=None
):
    if not(os.path.exists("animations")):
        os.mkdir("animations")
    all_lines = create_list_of_all_lines(line_dict, args)[::-1]
    frames_list = []

    sf = x_output / I.x
    y_output = int(sf * I.y)
    canvas = np.tile(background_color, (y_output, x_output, 1)).astype("uint8")
    frame_0 = Image.fromarray(canvas)

    if d_coords_output is None:
        d_coords_output = build_through_pixels_dict(
            x_output, y_output, args.n_nodes, shape=args.shape, critical_distance=10, only_return_d_coords=True, width_to_gap_ratio=1
        )
        print("Generated output pixels dict.")
    
    frame_points = [int(len(all_lines) * i / n_frames_total) for i in range(n_frames_total)]
    frame_point_counter = 0
    
    for line, color in all_lines:

        y0, x0 = np.array(d_coords_output[line[0]]) / [y_output, x_output]
        y0, x0 = np.array(hacky_permutation(y0, x0, rand_perm)) * [y_output, x_output]
        y0, x0 = np.clip([y0, x0], 0, [y_output - 1, x_output - 1])

        y1, x1 = np.array(d_coords_output[line[1]]) / [y_output, x_output]
        y1, x1 = np.array(hacky_permutation(y1, x1, rand_perm)) * [y_output, x_output]
        y1, x1 = np.clip([y1, x1], 0, [y_output - 1, x_output - 1])

        pixels_y, pixels_x = through_pixels([y0, x0], [y1, x1]).astype("int")
        canvas[pixels_y, pixels_x] = color
        
        if frame_point_counter in frame_points:
            frames_list.append(Image.fromarray(canvas))
        frame_point_counter += 1

    frames_list.append(Image.fromarray(canvas))
    print("Created frames list.")

    filename = f"animations/animated-{args.name}.gif"
    frame_0.save(fp=filename, format="GIF", append_images=frames_list, save_all=True, duration=gif_duration)

    print(f"Saved animation as: {filename!r}")

    # return d_coords_output
