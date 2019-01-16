# -*- coding:utf-8 -*-
import numpy as np, cv2

mat = np.array(
    [
        [[0,0],[0,1],[0,2]],
        [[1,0],[1,1],[1,2]],
        [[2,0],[2,1],[2,2]]
    ]
)
mat2 = np.array(
    [
        [[0,0],[0,1],[0,2]],
        [[1,0],[1,1],[1,2]],
        [[2,0],[2,2],[2,2]]
    ]
)

print(np.where((mat2-mat==0).all(axis=2), np.array([0,0]), mat2))

