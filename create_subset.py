from config import *
from glob import glob
from tqdm import tqdm
import random
import os
from PIL import Image

object_ids = random.choices(list(range(1,101)), k=subset_size)
print(f"{object_ids = } {len(object_ids) = }")
# The samples used for training
# object_ids = [28, 7, 51, 33, 14, 42, 60, 74, 47, 48, 6, 17, 25, 37, 46, 56, 78, 83, 92, 5] len(object_ids) = 20

if not os.path.isdir(f"{data_dir}/subset"):
    os.makedirs(f"{data_dir}/subset/images")

for object_id in tqdm(object_ids):
    for src_file in glob(f"{data_dir}/coil-100/obj{object_id}__*.png"):
        dest_file = src_file.replace("coil-100", "subset/images").replace("png", "jpg")
        with Image.open(src_file) as img:
            img = img.convert("RGB")
            img.save(dest_file)
