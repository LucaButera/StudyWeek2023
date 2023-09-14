import random
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

if __name__ == "__main__":
    images = []
    for d in Path.cwd().joinpath('dataset').iterdir():
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file() and f.suffix == '.png':
                    images.append(f.absolute())
    random.shuffle(images)

    nrows, ncols = 10, 20
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(nrows, ncols),
                     axes_pad=0.1,
                     share_all=True,
                     )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    for i, ax in enumerate(grid):
        img = plt.imread(images[i])
        ax.imshow(img)

    plt.savefig('dataset_grid.jpg')
