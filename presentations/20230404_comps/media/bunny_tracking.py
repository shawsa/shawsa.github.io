

import imageio
import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
import os
from tqdm import tqdm

FILE_NAME = 'bun_tracking.gif'

bun = image.imread('cartoon_bunny.png')
outline = image.imread('cartoon_bunny_outline.png')

bun_width = 4
bun_height = 0.77*bun_width
grass_width = 3*bun_width
grass_height = grass_width * 315/920

def pos_tup(pos):
    return (0+pos, bun_width+pos, 0, bun_height)


speed = 1
bun_shift = .25*bun_width

final_time = 10
aquire_time = 2.5
shift_time = 5

time = np.linspace(0, final_time, 201)
dt = time[1] - time[0]

fig, ax = plt.subplots(figsize=(10, 3))
ax.fill_between([-5*bun_width, 20*bun_width],
                [-20]*2,
                [bun_height*.9]*2,
                color='forestgreen', zorder=-10)
ax.fill_between([-5*bun_width, 20*bun_width],
                [bun_height*.9]*2,
                [bun_height*10],
                color='deepskyblue', zorder=-10)
ax.fill_between([bun_width*1.2, bun_width*3], [-.5]*2, [bun_height+.5]*2, color='saddlebrown')
ax.fill_between([bun_width*3.1, bun_width*5], [-.5]*2, [bun_height+.5]*2, color='saddlebrown')
ax.axis('off')
ax.set_xlim(-.5*bun_width, 6*bun_width)
ax.set_ylim(-.5*bun_height, 3*bun_height)
ax.set_aspect('equal', 'box')
plt.tight_layout()
with imageio.get_writer(FILE_NAME, mode='I', duration=.05) as writer:
    for t in tqdm(time):
        pos = speed*t
        outline_pos = pos
        if t >= shift_time:
            pos += bun_shift
        bun_obj = ax.imshow(bun, aspect='auto', extent=pos_tup(pos), zorder=0)
        if t >= aquire_time:
            outline_obj = ax.imshow(outline, aspect='auto',
                                    extent=pos_tup(outline_pos),
                                    zorder=2)
        ax.set_xlim(-.5*bun_width, 6*bun_width)
        ax.axis('off')
        ax.set_aspect('equal', 'box')
        plt.tight_layout()
        plt.savefig(FILE_NAME + '.png')
        image = imageio.imread(FILE_NAME + '.png')
        writer.append_data(image)
        plt.pause(0.01)
        try:
            bun_obj.remove()
            outline_obj.remove()
        except:
            pass
os.remove(FILE_NAME + '.png')
plt.close()
