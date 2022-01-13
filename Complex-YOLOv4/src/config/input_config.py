from easydict import EasyDict as edict

# unit: m
RANGE = 60

def complete_info(d):
    OVERLAP = (float(d.PATCH_NUM) * d.PATCH_SIZE - img_size) / (d.PATCH_NUM - 1.) if d.PATCH_NUM > 1 else 0
    # Front side (of vehicle) Point Cloud boundary for BEV
    d.boundary = [{"minX": RANGE * (d.PATCH_SIZE * i - OVERLAP * i) / img_size,
                   "maxX": RANGE * (d.PATCH_SIZE * (i + 1) - OVERLAP * i) /img_size ,
                   "minY": -RANGE/2 + RANGE * (d.PATCH_SIZE * j - OVERLAP * j) / img_size ,
                   "maxY": -RANGE/2 + RANGE * (d.PATCH_SIZE * (j + 1) - OVERLAP * j) / img_size ,
                   "minZ": -2.73,
                   "maxZ": 1.27} for i in range(d.PATCH_NUM) for j in range(d.PATCH_NUM)]
    d.DISCRETIZATION = [(d.boundary[i]["maxX"] - d.boundary[i]["minX"]) / d.BEV_HEIGHT for i in range(len(d.boundary))]
    return

# unit: pixel
img_size = 880.
patch_size = 256.
downsample = 4
patch_num = 4  # (1d)

LR_IMAGE_CFG = edict({
    "PATCH_NUM": 1,
    "PATCH_SIZE": img_size,
    "BEV_WIDTH": int(img_size / downsample), "BEV_HEIGHT": int(img_size / downsample),
    "color": [199, 76, 129]
})
complete_info(LR_IMAGE_CFG)

HR_IMAGE_CFG = edict({
    "PATCH_NUM": 1,
    "PATCH_SIZE": img_size,
    "BEV_WIDTH": int(img_size), "BEV_HEIGHT": int(img_size),
    "color": [207, 150, 81]
})
complete_info(HR_IMAGE_CFG)

LR_PATCH_CFG = edict({
    "PATCH_NUM": patch_num,
    "PATCH_SIZE": patch_size,
    "BEV_WIDTH": int(patch_size / downsample), "BEV_HEIGHT": int(patch_size / downsample),
    "color": [199, 76, 129] # purple
})
complete_info(LR_PATCH_CFG)

HR_PATCH_CFG = edict({
    "PATCH_NUM": patch_num,
    "PATCH_SIZE": patch_size,
    "BEV_WIDTH": int(patch_size), "BEV_HEIGHT": int(patch_size),
    "color": [207, 150, 81] # blue
})
complete_info(HR_PATCH_CFG)

# Back back (of vehicle) Point Cloud boundary for BEV
# boundary_back = {
#     "minX": -50,
#     "maxX": 0,
#     "minY": -25,
#     "maxY": 25,
#     "minZ": -2.73,
#     "maxZ": 1.27
# }

if __name__ == "__main__":
    print(LR_IMAGE_CFG)
    print(LR_PATCH_CFG)
    print(HR_IMAGE_CFG)
    print(HR_PATCH_CFG)
