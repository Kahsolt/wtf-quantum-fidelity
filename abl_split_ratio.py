#!/usr/bin/env python3
# Author: FlowerWater1019
# Create Time: 2024/10/25 

# 用不同的 split_ratio 训模型，看偏差程度

import matplotlib.pyplot as plt

# split_ratio: 0.1 -> 0.9
exp_stats = [
  # (theoretical, actual)
  (0.010433080768173172,  0.009859287040916489),
  (0.00605502999781651,   0.006521112107830001),
  (0.006467882813028192,  0.006177191819242997),
  (0.00566861063607098,   0.00616089931307233),
  (0.007061314019913397,  0.0071140949913510365),
  (0.00890664899041532,   0.007958781489462082),
  (0.005009275990004733,  0.003680053137767045),
  (0.004769508631794176,  0.005253707317322401),
  (0.0072126369410407155, 0.003818922366149628),
]

X = [e[1] for e in exp_stats]
Y = [e[0] for e in exp_stats]
plt.scatter(X, Y)
plt.xlabel('actual')
plt.ylabel('theoretical')
plt.suptitle('Ablation on split_ratio (ResNet18)')
plt.tight_layout()
plt.savefig('./img/abl_split_ratio.png', dpi=400)
