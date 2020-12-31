def read_image_scheme(addr):
    start_rows = []
    start_columns = []
    patches = []
    for filename in glob(addr):
        s_r, s_c, l_r, l_c = [int(s) for s  in os.path.basename(filename)[:-6].split('-')]
        with open(filename) as f:
            patch = np.array(f.read().split('\n')).reshape((l_r, l_c))
        patches.append(patch)
        start_rows.append(s_r)
        start_columns.append(s_c)

    df = pd.DataFrame({'s_r':start_rows, 's_c':start_columns, 'patch':patches})
    rows = []
    for row_n, one_row in df.sort_values(by=['s_r', 's_c']).groupby('s_r'):
        rows.append(np.concatenate(one_row.patch.tolist(), axis=1))

    img = np.concatenate(rows, 0)

    return img

from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000
import pandas as pd
from itertools import combinations_with_replacement, combinations
import numpy as np
from sklearn.cluster import KMeans
from copy import deepcopy

rgb_fx = lambda fx: (int(fx[1:3], 16), int(fx[3:5], 16), int(fx[5:7], 16))
lab_fx = lambda fx: rgb2lab(np.array((int(fx[1:3], 16), int(fx[3:5], 16), int(fx[5:7], 16))).astype(np.uint8))


def get_palettes(multithread=2):
    general_palette = pd.read_csv('basic_DMC_palette.csv', index_col=0)

    clear_palette = {str(k): lab_fx(v) for k,v in general_palette['rgb_hex'].to_dict().items()}
    mix_palette = {str(k): np.stack(v).mean(0) for k,v in zip(combinations_with_replacement(clear_palette.keys(), multithread), combinations_with_replacement(clear_palette.values(), multithread))}

    return clear_palette, mix_palette

def get_image_loss(new_image, old_image):
    return np.sqrt(((new_image - old_image) ** 2).sum(-1)).mean()

def get_quantization_loss(image, model):
    return get_image_loss(model.cluster_centers_[model.predict(image.reshape(-1, 3))], image.reshape(-1, 3))

def best_removal(model, image):
    centers_backup = deepcopy(model.cluster_centers_)
    initial_loss = get_quantization_loss(image, model)

    losses = []

    for cid in range(len(model.cluster_centers_)):
        model.cluster_centers_ = np.concatenate([model.cluster_centers_[:cid], model.cluster_centers_[cid+1:]])
        losses.append(get_quantization_loss(image, model) - initial_loss) # hope we can replace it with more optimal some day
        model.cluster_centers_ = deepcopy(centers_backup)

    return losses

def best_replacement(model, image, swap_centers):
    centers_backup = deepcopy(model.cluster_centers_)
    initial_loss = get_quantization_loss(image, model)

    gains = []

    for cid in range(len(model.cluster_centers_)):
        model.cluster_centers_[cid] = swap_centers[cid]
        gains.append(initial_loss - get_quantization_loss(image, model))
        model.cluster_centers_ = deepcopy(centers_backup)

    return gains

def bin_colors(image, total_colors, too_small=1, multithread=2):
    image_lab = rgb2lab(image)
    vector_lab = image_lab.reshape(-1, 3)

    clustr = KMeans(n_clusters=total_colors)
    clustr.fit(vector_lab)

    clear_palette, mixed_palette = get_palettes(multithread)

    clear_colors = np.array(list(clear_palette.values()))
    mixed_colors = np.array(list(mixed_palette.values()))

    nearest_clear_internal = [np.argmin(deltaE_ciede2000(clear_colors, c)) for c in clustr.cluster_centers_]
    nearest_mixed_internal = [np.argmin(deltaE_ciede2000(mixed_colors, c)) for c in clustr.cluster_centers_]

    shifted_centers_clear = clear_colors[nearest_clear_internal]
    shifted_centers_mixed = mixed_colors[nearest_mixed_internal]

    shifted_ids_clear = np.array(list(clear_palette.keys()))[nearest_clear_internal].astype(np.object)
    shifted_ids_mixed = np.array(list(mixed_palette.keys()))[nearest_mixed_internal].astype(np.object)

    cluster_center_ids = deepcopy(shifted_ids_clear)

    cluster_centers_backup = deepcopy(clustr.cluster_centers_)
    clustr.cluster_centers_ = shifted_centers_clear
    number_of_clear_colors = len(clustr.cluster_centers_)

    while number_of_clear_colors > 0:
        losses_of_removal = best_removal(clustr, vector_lab)
        gains_of_swap = best_replacement(clustr, vector_lab, shifted_centers_mixed)

        if np.min(losses_of_removal) < np.max(gains_of_swap):
            # worthy to remove one old and replace with better new
            center_to_remove = np.argmin(losses_of_removal)
            center_to_swap = np.argmax(gains_of_swap)

            if center_to_remove == center_to_swap:
                # special case, we should actually try to remove second worst
                if np.sort(losses_of_removal)[1] < np.max(gains_of_swap):
                    center_to_remove = np.argsort(losses_of_removal)[1]
                else:
                    break # than it's finished -- we can not remove this center to add two instead

            clustr.cluster_centers_[center_to_swap] = shifted_centers_mixed[center_to_swap]
            clustr.cluster_centers_ = np.concatenate([clustr.cluster_centers_[:center_to_remove], clustr.cluster_centers_[center_to_remove+1:]])

            cluster_center_ids[center_to_swap] = shifted_ids_mixed[center_to_swap]
            cluster_center_ids = np.concatenate([cluster_center_ids[:center_to_remove], cluster_center_ids[center_to_remove+1:]])

            number_of_clear_colors -= 1
        else:
            break # it's finished, gain from swap is lesser than loss from removal

    new_image_ids = clustr.predict(vector_lab)
    new_image_code = cluster_center_ids[new_image_ids].reshape(image.shape[:2])

    palette = {k:v for k,v in zip(cluster_center_ids, clustr.cluster_centers_)}

    final_loss = get_quantization_loss(image, clustr)

    return new_image_code, palette, final_loss

colorize_scheme = lambda scheme, palette: lab2rgb(np.array([palette[i] for i in scheme.reshape(-1)]).reshape((*scheme.shape, 3)))

def convert_small_pixels(scheme, palette, radius, thr_count, max_shift=None):
    conversion_result = np.zeros(scheme.shape)
    image = deepcopy(scheme)
    for i in range(radius, image.shape[0]-radius):
        for j in range(radius, image.shape[1]-radius):
            pixel_value = image[i,j]

            fr_i = max(0,i-radius)
            to_j = min(image.shape[1], j+radius+1)
            fr_j = max(0,j-radius)
            to_i = min(image.shape[0], i+radius+1)
            search_area = image[fr_i:to_i, fr_j:to_j]

            if (search_area == pixel_value).sum() < thr_count+1: #+1 for pixel itself
                conversion_result[i,j] = 1
                k, p = np.unique(search_area, return_counts=True)
                candidates = k[(p >= thr_count) & (k != pixel_value)]
                distances = deltaE_ciede2000(palette[pixel_value], [palette[c] for c in candidates])
                candidate = candidates[np.argmin(distances)]
                distance = np.min(distances)
                if max_shift is not None:
                    if distance < max_shift:
                        image[i, j] = candidate
                        conversion_result[i,j] = 2
                else:
                    image[i, j] = candidate
                    conversion_result[i,j] = 2

    return image, conversion_result

import os

canvas_correction = {11: 396, 14:510, 16:590, 18:681, 20:786, 22:912}
def calc_length(crosses, canvas_size, threads=1):
    meters = crosses / canvas_correction[canvas_size]
    meters *= threads
    pasm = int(np.ceil(meters/8))
    return meters, pasm

from collections import defaultdict
def get_shopping_list(scheme, multithread=2, aida=18):
    colors, stitches = np.unique(scheme, return_counts=True)

    cart_m = defaultdict(float)
    for c,s in zip(colors, stitches):
        if c[0] == '(':
            m,p = calc_length(s, aida, 1)
            c = eval(c)
            for sc in c:
                cart_m[sc] += m
        else:
            m,p = calc_length(s, aida, multithread)
            cart_m[c] += m
    return cart_m, {k: int(np.ceil(v/8)) for k,v in cart_m.items()}

from jinja2 import Environment, FileSystemLoader
import skimage.io

def get_pdf(scheme, filename, project_name, colorisation_palette, multi_threading=2, aida=18):
    abbreviations = np.array([i for i in 'abcefghikmnoprstuvxyz1234567890'])
    c_p_p = 50 #columns per page
    r_p_p = 50 #rows per page

    color_image = colorize_scheme(scheme, colorisation_palette)
    skimage.io.imsave('tex_result/scheme.png', color_image)

    palette = {}
    for abbr, color in zip(abbreviations, np.unique(scheme)):
        if color[0] == '(':
            palette[abbr] = ' + '.join(eval(color))
        else:
            palette[abbr] = color

    wimg = np.vectorize({v:i for i,v in zip(abbreviations, np.unique(scheme))}.get)(scheme)

    row_tiles = int(np.ceil(wimg.shape[0]/r_p_p))
    column_tiles = int(np.ceil(wimg.shape[1]/c_p_p))

    shopping_meters, _ = get_shopping_list(scheme, multithread=multi_threading, aida=aida)

    env = Environment(loader=FileSystemLoader('.'))
    templ = env.get_template('page_template.jinja')
    for r in range(row_tiles):
        for c in range(column_tiles):
            with open(f'tex_result/page_{r}_{c}.tex', 'w') as f:
                print(templ.render(patch=wimg[r*r_p_p:(r+1)*r_p_p, c*c_p_p:(c+1)*c_p_p], start_row=r*r_p_p+1, start_column=c*c_p_p+1,
                                   col_fill=len(str(wimg.shape[1])), row_fill=len(str(wimg.shape[0]))), file=f)

    templ = env.get_template('main_template.jinja')
    with open('tex_result/main.tex', 'w') as mf:
        print(templ.render(multi_threading=multi_threading, shape=scheme.shape, palette=palette, aida=aida, project_name=project_name,
                            shopping_cart=shopping_meters, column_number=column_tiles, row_number=row_tiles), file=mf)

    print(os.system('cd tex_result; pdflatex main.tex'))
    print(os.system(f'cp tex_result/main.pdf {filename}.pdf'))
