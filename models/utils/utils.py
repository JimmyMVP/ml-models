import numpy as np


#Voxelize the pointcloud
def voxel_grid(data, res=(32.0,32.0,32.0), limits=(25,25,25)):
    
    X_RES = res[0]
    Y_RES = res[1]
    Z_RES = res[2]
    limit = limits[0]
    voxel_grid = np.zeros(shape=(X_RES, Y_RES, Z_RES))

    for p in data:

        x,y,z = p[0],p[1],p[2]
        if abs(x) > limit:
            continue
        elif abs(y) > limit:
            continue
        elif abs(z) > limit:
            continue

        grid_x = int(np.ceil(((p[0]) / (2*limit))*X_RES))
        grid_y = int(np.ceil(((p[1]) / (2*limit))*Y_RES))
        grid_z = int(np.ceil(((p[2])) / (2*limit)*Z_RES))

        voxel_grid[grid_x][grid_y][grid_z]+=1.0

    return voxel_grid