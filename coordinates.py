# ====== Includes all funcs related to coordinate placement ======

import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display, SVG
from tqdm import tqdm
import torch as t

# ================================================================

# Returns distance between two points
def dist(p0, p1, return_δ=False):
    """
    Given two coordinates, returns distance between them
    """
    
    δ = np.subtract(p1, p0)
    distance = (δ**2).sum()**0.5
    
    return (distance, δ / distance) if return_δ else distance

# Truncates a single pair of coordinates, to make sure that none are outside the bounds of the image (used in `build_through_pixels_dict` function)
def truncate_coords(coords, limits):
    """
    Truncates coordinates at some limit
    """
    
    for i in range(2):
        coords[i] = max(0, min(coords[i], limits[i]))
    
    return coords

# Truncates an array of pixel coordinates, to make sure that none are outside the bounds of the image (used in `build_through_pixels_dict` function)
def truncate_pixels(pixels, limits):
    """
    Truncates a pixels array (i.e. that was generated from the through_pixels function), to avoid index errors
    """

    for i in range(2):
        pixels[i][pixels[i] < 0] = 0
        pixels[i][pixels[i] > limits[i]] = limits[i]

    return pixels

# Gets array of pixels going through any two points (used in lots of other functions)
def through_pixels(p0, p1):
    """
    Given a numpy array [p1y, p1x, p2y, p2x], returns the pixels that the line connecting p1 & p2 passes through
    
    Returns it as float rather than integer, so that it can be reflected / translated accurately before being converted to int
    """
    
    δ = np.subtract(p1, p0)
    
    distance = (δ**2).sum()**0.5
            
    assert distance > 0, f"Error: {p0} and {p1} have distance zero."
    
    pixels_in_line = p0 + np.outer((np.arange(int(distance) + 1) / distance), δ)
    
    return pixels_in_line.T

def get_thick_line(p0, p1, all_coords, thickness=1):
    
    p0y, p0x = p0
    p1y, p1x = p1
    
    if p0x == p1x:
        a = 0
        b = 1
    else:
        a = (p1y - p0y) / (p1x - p0x)
        ab_norm = np.sqrt(1 + a**2)
        a /= ab_norm
        b = -1 / ab_norm
                
    c_p = a*p0x + b*p0y
    c_q = a*all_coords[1] + b*all_coords[0]
    
    return all_coords[:, np.abs(c_p - c_q) < thickness]

# ================================================================

def build_through_pixels_dict(x, y, n_nodes, shape, critical_distance=14, only_return_d_coords=False, width_to_gap_ratio=1):

    if shape == "Rectangle" and type(n_nodes) == int:
        assert (n_nodes % 4) == 0, f"n_nodes = {n_nodes} needs to be divisible by 4, or else there will be an error"

    d_coords = {}
    d_pixels_archetypes = {}
    d_pixels = {}
    d_joined = {}
    d_sides = {}

    if shape == "Rectangle":
    
        # we either read the number of nodes and divide them proportionally between sides, or the number of nodes is externally specified
        if type(n_nodes) in [int, np.int64]:
            nx = 2 * int(n_nodes * 0.25 * x / (x + y))
            ny = 2 * int(n_nodes * 0.25 * y / (x + y))

            while 2 * (nx + ny) < n_nodes:
                if ny >= nx: ny += 2
                else: nx += 2
            while 2 * (nx + ny) > n_nodes:
                if ny >= nx: ny -= 2
                else: nx -= 2
        elif isinstance(n_nodes, tuple):
            nx, ny = n_nodes
            n_nodes = 2 * (nx + ny)
        else:
            raise TypeError(f"n_nodes = {n_nodes} is of type {type(n_nodes)}, which is not supported")

        nodes_per_side_list = [ny, nx, ny, nx]
        
        starting_idx_list = np.cumsum([0] + nodes_per_side_list)

        x -= 1
        y -= 1

        xd = x / nx
        yd = y / ny
        X0_list = t.tensor([(y, x), (0, x), (0, 0), (y, 0)])

        Xd_list = t.tensor([(-yd, 0), (0, -xd), (yd, 0), (0, xd)])
        n0, n1, n2, n3, n4 = starting_idx_list
        
        # =============== get all the coordinates of the nodes, and their sides ===============
        
        for (side, starting_idx, X0, Xd) in zip(range(4), starting_idx_list, X0_list, Xd_list):
            
            nodes_per_side = nodes_per_side_list[side]

            δ = (width_to_gap_ratio - 1) / (2 * (width_to_gap_ratio + 1))
            
            for i in range(nodes_per_side):
                
                idx = starting_idx + i
                coords_raw = X0 + (i + 0.5) * Xd
                if (i % 2) == 0:
                    coords_raw -= δ * Xd
                else:
                    coords_raw += δ * Xd
                d_coords[idx] = truncate_coords(coords_raw, limits=[y, x])
                d_sides[idx] = side

        if only_return_d_coords:
            return d_coords
                
        # =============== get the joined pixels (i.e. the ones not on the same side) ===============

        for i in d_sides:
            d_joined[i] = [j for j in range(n4) if d_sides[i] != d_sides[j]]
        
        # =============== get all the archetypes (i.e.lines not related via translation/reflection/rotation) ===============
        
        # ==== first, get the vertical ones ====
        for δ in range(nx + 1):
            key = f"vertical_{δ}"
            i = n2
            j = (n3 + δ) % n4
            d_pixels_archetypes[key] = through_pixels(d_coords[i], d_coords[j])
            
        # ==== then, get the horizontal ones ====
        for δ in range(ny + 1):
            key = f"horizontal_{δ}"
            i = n1
            j = n2 + δ
            d_pixels_archetypes[key] = through_pixels(d_coords[i], d_coords[j])
            
        # ==== finally, get the diagonal ones ====
        n_min = min(nx, ny)
        n_max = max(nx, ny)
        for adj in range(n_min + 1):
            for opp in range(max(adj, 1), n_max + 1):
                key = f"diagonal_{adj}_{opp}"
                if nx >= ny:
                    i = n2 + adj
                    j = n2 - opp
                else:
                    i = n2 - adj
                    j = n2 + opp
                d_pixels_archetypes[key] = through_pixels(d_coords[i], d_coords[j])
        
        # =============== use the archetypes to fill in the actual lines ===============

        progress_bar = tqdm(desc="Building pixels dict", total=sum([len(d_joined[i]) for i in d_joined]) // 2)

        for idx, i in enumerate(d_joined):

            for j in d_joined[i]:
                
                if i >= j:
                    continue
                    
                # === first, check if they're opposite vertical, if so then populate using the archetypes ===
                elif (d_sides[i], d_sides[j]) == (1, 3):
                    # this makes sure the node with vertex 0 is seen as being on side 3, not side 0
                    if i == 0:
                        i_, j_ = j, n4
                    else:
                        i_, j_ = i, j
                    δ = (i_ + j_) - (2*n2 + ny)               
                    key = f"vertical_{abs(δ)}"
                    pixels = d_pixels_archetypes[key].clone()
                    if δ < 0:
                        pixels[1] = -pixels[1]
                    pixels[1] += (n2 - i_) * xd
                    d_pixels[(i, j)] = truncate_pixels(pixels.to(t.int), [y, x]).long()
                    
                # === then, check if they're opposite horizontal, if so then populate using the archetypes ===
                elif (d_sides[i], d_sides[j]) == (0, 2):
                    δ = (i + j) - (2*n1 + nx)
                    key = f"horizontal_{abs(δ)}"
                    pixels = d_pixels_archetypes[key].clone()
                    if δ < 0:
                        pixels[0] = -pixels[0]
                    pixels[0] += (n1 - i) * yd
                    d_pixels[(i, j)] = truncate_pixels(pixels.to(t.int), [y, x]).long()
                    
                # === finally, the diagonal case ===
                else:
                    i_side = d_sides[i]
                    j_side = d_sides[j]
                    
                    x_side = i_side if (i_side % 2 == 1) else j_side
                    y_side = i_side if (i_side % 2 == 0) else j_side
                    
                    if i_side == 0 and j_side == 3:
                        i_, j_ = j, i
                        i_side, j_side = 3, 0
                    else:
                        i_, j_ = i, j
                    
                    i_len = starting_idx_list[i_side + 1] - i_
                    j_len = j_ - starting_idx_list[j_side]
                    
                    x_len = i_len if (i_side % 2 == 1) else j_len
                    y_len = i_len if (i_side % 2 == 0) else j_len
                    
                    adj = min(i_len, j_len)
                    opp = max(i_len, j_len)
                    
                    key = f"diagonal_{adj}_{opp}"
                    pixels = d_pixels_archetypes[key].clone()
                    
                    # flip in x = y
                    if ((x_len > y_len) != (x > y)) and (x_len != y_len):
                        pixels = pixels.flip(0)
                    # flip in x
                    if x_side == 3:
                        pixels[0] = y - pixels[0]
                    # flip in y
                    if y_side == 0:
                        pixels[1] = x - pixels[1]
                        
                    d_pixels[(i, j)] = truncate_pixels(pixels.to(t.int), [y, x]).long()

                progress_bar.update(1)

        progress_bar.n = sum([len(d_joined[i]) for i in d_joined]) // 2

    elif shape == "Ellipse":

        assert x % 2 == 0, "x must be even to take advantage of symmetry"

        angles = np.linspace(0, 2 * np.pi, n_nodes + 1)[:-1]
            
        x_coords = 1 + ((0.5*x) - 2) * (1 + np.cos(angles))
        y_coords = 1 + ((0.5*y) - 2) * (1 - np.sin(angles))
        
        coords = t.stack([t.from_numpy(y_coords), t.from_numpy(x_coords)]).T
        
        d_sides = None
        d_joined = {n: [] for n in range(n_nodes)}
        for i, coord in enumerate(coords):
            d_coords[i] = coord
            # the line below is an efficient way of saying "d_joined[i] = all nodes at least `critical_distance` from i"
            d_joined[i] = sorted(np.mod(range(i + critical_distance, i + (n_nodes + 1 - critical_distance)), n_nodes))

        # The second half are added via symmetry
        total=sum([len(d_joined[i]) for i in d_joined]) // 4
        progress_bar = tqdm(desc="Building pixels dict", total=total)

        for i1 in d_joined:
            p1 = d_coords[i1]
            for i0 in d_joined[i1]:
                # Avoid double counting: only consider (i0, i1) for i0 < i1
                if i0 > i1:
                    break
                # Check if the reflection of this line is already in the dict
                reflection = (n_nodes + 1 - i1, n_nodes + 1 - i0)
                if reflection in d_pixels: # and False:
                    y_reflected, x_reflected = d_pixels[reflection]
                    d_pixels[(i0, i1)] = t.stack([(y - y_reflected).flip(0), x_reflected.flip(0)])
                # If reflection isn't in the dict, add it
                else:
                    p0 = d_coords[i0]
                    d_pixels[(i0, i1)] = through_pixels(p0, p1).to(t.int)
                    progress_bar.update(1)
        
        if only_return_d_coords:
            return d_coords
        
    # =============== populate the tensor ===============

    max_pixels = max([pixels.size(1) for pixels in d_pixels.values()])
    t_pixels = t.zeros((n_nodes, n_nodes, 2, max_pixels), dtype=t.int)
    for (i, j), pixels in d_pixels.items():
        t_pixels[i, j, :, :pixels.size(1)] = pixels
        t_pixels[j, i, :, :pixels.size(1)] = pixels

    output = [d_coords, d_pixels, d_joined, d_sides, t_pixels]

    return output
