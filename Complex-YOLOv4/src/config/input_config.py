from easydict import EasyDict as edict


def complete_info(d):
    OVERLAP = (float(d.PATCH_NUM) * d.PATCH_SIZE - d.IMAGE_SIZE) / (d.PATCH_NUM - 1.) if d.PATCH_NUM > 1 else 0
    # Front side (of vehicle) Point Cloud boundary for BEV
    d.boundary = [{"minX": 50. / d.IMAGE_SIZE * (d.PATCH_SIZE * i - OVERLAP * i),
                   "maxX": 50. / d.IMAGE_SIZE * (d.PATCH_SIZE * (i + 1) - OVERLAP * i),
                   "minY": -25. + 50. / d.IMAGE_SIZE * (d.PATCH_SIZE * j - OVERLAP * j),
                   "maxY": -25. + 50. / d.IMAGE_SIZE * (d.PATCH_SIZE * (j + 1) - OVERLAP * j),
                   "minZ": -2.73,
                   "maxZ": 1.27} for i in range(d.PATCH_NUM) for j in range(d.PATCH_NUM)]
    d.DISCRETIZATION = [(d.boundary[i]["maxX"] - d.boundary[i]["minX"]) / d.BEV_HEIGHT for i in range(len(d.boundary))]
    return


img_size = 608.

LR_IMAGE_CFG = edict({
    "PATCH_NUM": 1, "IMAGE_SIZE": img_size, "PATCH_SIZE": img_size,
    "BEV_WIDTH": int(img_size), "BEV_HEIGHT": int(img_size)
})
complete_info(LR_IMAGE_CFG)

LR_PATCH_CFG = edict({
    "PATCH_NUM": 2, "IMAGE_SIZE": img_size, "PATCH_SIZE": 320.,
    "BEV_WIDTH": 64, "BEV_HEIGHT": 64
})
complete_info(LR_PATCH_CFG)

HR_PATCH_CFG = edict({
    "PATCH_NUM": 2, "IMAGE_SIZE": img_size, "PATCH_SIZE": 320.,
    "BEV_WIDTH": 320, "BEV_HEIGHT": 320
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
