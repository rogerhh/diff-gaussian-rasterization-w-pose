import torch
import numpy as np
import csv

args = torch.load("args.pth")

bg, means3D, colors_precomp, opacities, scales, rotations, scale_modifier, cov3Ds_precomp, viewmatrix, projmatrix, projmatrix_raw, tanfovx, tanfovy, image_height, image_width, sh, sh_degree, campos, prefiltered, debug = args

# Save each data to a csv file
data = {
    "bg": bg,
    "means3D": means3D,
    "colors_precomp": colors_precomp,
    "opacities": opacities,
    "scales": scales,
    "rotations": rotations,
    "scale_modifier": scale_modifier,
    "cov3Ds_precomp": cov3Ds_precomp,
    "viewmatrix": viewmatrix,
    "projmatrix": projmatrix,
    "projmatrix_raw": projmatrix_raw,
    "tanfovx": tanfovx,
    "tanfovy": tanfovy,
    "image_height": image_height,
    "image_width": image_width,
    "sh": sh,
    "sh_degree": sh_degree,
    "campos": campos,
}

for key, value in data.items():
    print("Saving", key)
    if key == "sh":
        value = value.squeeze()
    if isinstance(value, int):
        np.savetxt(f"{key}.csv", [value], delimiter=" ", fmt='%d')
    elif isinstance(value, float):
        np.savetxt(f"{key}.csv", [value], delimiter=" ", fmt='%.8f')
    elif isinstance(value, torch.Tensor):
        np.savetxt(f"{key}.csv", value.detach().cpu().numpy(), delimiter=" ", fmt='%.8f')

np.savetxt("prefiltered.csv", [int(prefiltered)], delimiter=" ", fmt='%d')
np.savetxt("debug.csv", [int(debug)], delimiter=" ", fmt='%d')


