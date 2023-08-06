# ====== Includes all functions hard to categorise elsewhere ======

import re
from functools import reduce
from PIL import Image, ImageDraw, ImageOps
from IPython.display import display, HTML
import numpy as np
from pathlib import Path
import pandas as pd
# import qgrid
import cairo
import webcolors
import torch as t

def scale_down(n_lines_per_color, target):
    sf = target if (0 <= target <= 2) else (target / sum(n_lines_per_color))
    print((np.array(n_lines_per_color) * sf).astype(int).tolist())

def get_color_hash(c):
    assert isinstance(c, t.Tensor)
    return 256*256*c[0] + 256*c[1] + c[2]

def get_img_hash(i):
    assert isinstance(i, t.Tensor)
    return 256*256*i[:,:,0] + 256*i[:,:,1] + i[:,:,2]

def hsv_to_rgb_pixel(h, s, v):
    if s == 0.0: 
        return (v, v, v)
    i = int(h*6.)
    f = (h*6.)-i
    p , q, t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f))
    i %= 6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)

def hsv_to_rgb_image(I):

    hue = np.full_like(I.w.numpy(), 0)
    saturation = I.w.numpy()
    value = 1 - (I.imageBW.numpy() / 255)

    hsv_img = np.array([hue, saturation, value]).T.swapaxes(0, 1)
    rgb_img = np.array([[hsv_to_rgb_pixel(*p) for p in row] for row in hsv_img]) * 255

    return rgb_img

def hex_to_rgb(c):
    c = c.lstrip('#')
    lv = len(c)
    return tuple(int(c[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_description(c):
    if isinstance(c, t.Tensor):
        c = c.numpy()
    min_colours = {}
    for name, hex_value in webcolors.CSS3_NAMES_TO_HEX.items():
        c_rgb = hex_to_rgb(hex_value)
        print(c)
        print(c_rgb)
        d = (np.subtract(c, c_rgb) ** 2).sum()
        min_colours[d] = name
    color_desc = min_colours[min(min_colours.keys())]
    color_name_replacement_dict = {"dodgerblue": "midblue", "darkturquoise": "cyan", "midnightblue": "navyblue", "darkorange": "orange", "lemonchiffon": "pale-lemon"}
    color_desc = color_name_replacement_dict.get(color_desc, color_desc)
    for prefix in ["light", "dark"]:
        if color_desc.startswith(prefix):
            color_desc = color_desc[len(prefix):] + "-" + prefix
    for prefix in ["deep"]:
        if color_desc.startswith(prefix):
            color_desc = color_desc[len(prefix):]
    if color_desc == "green" and c[1] < 140:
        color_desc = "pinegreen"
    return color_desc

def palette_to_html(color):
    color = literal_eval(color)
    html = ""
    for c in color:
        html += f"<span style='color:rgb{tuple(c)}'>██</span>"
    return html

# =================================================================

def insert_line_breaks(text, max_line_length=80):
    words = text.split()
    result = []
    current_line = []

    for word in words:
        if len(' '.join(current_line + [word])) <= max_line_length:
            current_line.append(word)
        else:
            result.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        result.append(' '.join(current_line))

    return '\n'.join(result)

# self-explanatory!
def swap_parity(i):
    if (i % 2) == 0:
        return i + 1
    else:
        return i - 1

# evaluates a string like "[[1, 2, 3], [4, 5, 6]]", and returns Python object (useful when working with qgrid)
def literal_eval(s):
    """
    This evaluates a string e.g. "[[1, 2, 3], [4, 5, 6]]", and returns the corresponding Python object.
    It's useful because when I edit qgrid, the grid interpretes that object as a string even though it's often a list, so I need to turn it back into a list.
    It works with:
        > integers, e.g. "[[1, 2, 3], [0, 0, 0]]" -> [[1, 2, 3], [0, 0, 0]]
        > floats, e.g. "[4.0]" -> [4.0]
        > strings, e.g. "[[butterfly.png, butterfly_orig.png]]" -> [["butterfly.png", "butterfly_orig.png"]]
        > combinations, e.g. [[0, color]] -> [[0, "color"]]
            
    It also just returns the item, if there's no suitable transformation to perform
    """
    # case: non-string element
    if type(s) != str:
        return s
    
    # case: list of lists
    if "[[" in s:
        list_of_lists = s[2:-2].split("], [")
        return [literal_eval("[" + sublist + "]") for sublist in list_of_lists]
    
    # case: simple list
    elif "[" in s:
        if s == "[]": return []
        sublist = s[1:-1].split(", ")
        sublist = [literal_eval(el) for el in sublist]
        return sublist
    
    # case: string element - this is the "base case"
    else:
        if s.isdigit():
            return int(s)
        elif s.replace(".", "").isdigit():
            return float(s)
        else:
            return (s.replace("'", "").replace('"', ''))


# applies the function above for every row of the dataframe (used in write_df_to_pickle)
def literal_eval_df(df):
    
    df_copy = df.copy()
    for i in df.index:
        for c in df.columns:
            df_copy.at[i, c] = literal_eval(df.at[i, c])
            
    return df_copy



# takes list of lists and concats them together (useful in many places, because I use this operation a lot)
def concat_lists(list_of_lists):
    """
    This function is just used for drawing coordinates (clearer to have it be a separate func than to keep calling `reduce`)
    """

    return reduce(lambda x, y: x+y, list_of_lists)


# splits lines into ranges corresponding to the number of groups (used to create threading instructions)
def get_range_of_lines(n_lines, n_groups, group_number, return_slice=False):
    
    # Remember: we count backwards, so we do the best lines last. This is why we go from 1 to (n_lines + 1)
    
    n_lines_per_group = (n_lines // n_groups) + 1

    if type(group_number) == tuple:
        start = (n_lines_per_group * group_number[0]) + 1
        end = min(n_lines_per_group * group_number[1], n_lines) + 1
    else:
        print("using `get_range_of_lines`, not a tuple")
        start = (n_lines_per_group * group_number) + 1
        end = min(n_lines_per_group * (group_number + 1), n_lines) + 1
    
    return slice(start, end) if return_slice else range(start, end)


# draws a diagram of vertices and lines (useful for physical visualisation)
def draw_diagram(x, y, primary_coords=[], secondary_coords=[], primary_paths=[], secondary_paths=[], img_width=700, array_coordinate_convention=False):

    # Note
    # array_coordinate_convention: top-left is origin, coords are (y, x)
    # not euclidean_coordinate_convention: bottom-left is origin, coords are (x, y)

    # both need to be converted to PIL convention: top-left is origin, coord are (x, y)
    
    if img_width == 0:
        return None
    
    if array_coordinate_convention:
        primary_coords = [(x_, y - y_) for (y_, x_) in primary_coords]
        secondary_coords = [(x_, y - y_) for (y_, x_) in secondary_coords]
        primary_paths = [[(x_, y - y_) for (y_, x_) in path] for path in primary_paths]
        secondary_paths = [[(x_, y - y_) for (y_, x_) in path] for path in secondary_paths]

    # this funky looking formula is just to make sure the border adjusts well to the size of the image
    all_coords = concat_lists([primary_coords] + [secondary_coords] + primary_paths + secondary_paths)
    min_x = min(coord[0] for coord in all_coords)
    max_x = max(coord[0] for coord in all_coords)
    min_y = min(coord[1] for coord in all_coords)
    max_y = max(coord[1] for coord in all_coords)
    required_border = max(-min_x, max_x-x, -min_y, max_y-y, 0)
    padding = 0.05 * img_width
    border = int(((required_border * img_width / x) + padding) / (1 + 2 * required_border / x))
    
    X = img_width
    Y = int(((img_width - 2 * border) * y / x) + (2 * border))
    sf = (X - 2 * border) / x

    img = Image.new("RGB", (X, Y), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    for secondary_path in secondary_paths:
        if len(secondary_path) >= 2:
            for (x0, y0), (x1, y1) in zip(secondary_path[:-1], secondary_path[1:]):
                xy = list(border + sf * np.array([x0, y0, x1, y1]))
                draw.line(xy=xy, fill=(0, 180, 0), width=1)

    for primary_path in primary_paths:
        if len(primary_path) >= 2:
            for (x0, y0), (x1, y1) in zip(primary_path[:-1], primary_path[1:]):
                xy = list(border + sf * np.array([x0, y0, x1, y1]))
                draw.line(xy=xy, fill=(255, 0, 0), width=1)
    
    draw.rectangle(xy=[border, border, X-border, Y-border], outline=(0, 0, 0), width=2)
    
        
    for (x_, y_) in secondary_coords:
        x_, y_ = int(border + sf*x_), int(border + sf*y_)
        draw.ellipse(xy=[x_-3, y_-3, x_+3, y_+3], fill=(0, 180, 0), outline=(0, 0, 0)) 
        
    for (x_, y_) in primary_coords:
        x_, y_ = int(border + sf*x_), int(border + sf*y_)
        draw.ellipse(xy=[x_-3, y_-3, x_+3, y_+3], fill=(255, 0, 0), outline=(0, 0, 0)) 

    display(ImageOps.flip(img))





def create_background(colors, x, y, line_width_multiplier, max_line_distance, n_lines_total, filename, scaling_factors=(1, 1, 1, 1)):
    
    """
    Creates an SVG background for my Squarespace thread art website
    
        colors: list of RGB tuples (I have them plus black)
        x, y: integers, represent image dimensions
        line_width_multiplier: scales the line thickness
        max_line_distance: float in range [0, 1], determines maximum amount lines can protrude
        n_lines_total: integer
    """
    
    def convert_point_to_coords(p):
    
        if p < x:
            c = (p, 0)
            s = 3
        elif p < x + y:
            c = (x, p - x)
            s = 0
        elif p < 2*x + y:
            c = (p - x - y, y)
            s = 1
        else:
            c = (0, p - 2*x - y)
            s = 2

        return (np.array(c), s)

    with cairo.SVGSurface("color/" + filename + ".svg", x, y) as surface:

        context = cairo.Context(surface)
        context.set_line_width(line_width_multiplier)
        
        context.rectangle(0, 0, x, y)
        context.set_source_rgb(1, 1, 1)
        context.fill()

        n_lines_per_color = int(n_lines_total / (len(colors) + 1))

        for color in colors + [(0, 0, 0)]:

            context.set_source_rgb(*[c/255 for c in color])

            line_count = 0

            while line_count < n_lines_per_color:

                p1, p2 = np.random.randint(2 * (x + y), size=2)
                (c1, s1), (c2, s2) = [convert_point_to_coords(p) for p in [p1, p2]]

                if np.any(np.abs(c1 - c2) < 0.01 * max(x, y)):
                    continue

                line_distance = np.random.random() * max_line_distance * scaling_factors[s1]

                c2 = c1 + line_distance * (c2 - c1)
                
                context.move_to(c1[0], c1[1])
                context.line_to(c2[0], c2[1])

                context.stroke()
                
                line_count += 1