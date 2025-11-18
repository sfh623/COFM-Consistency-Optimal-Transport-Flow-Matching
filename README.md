# COFM-Consistency-Optimal-Transport-Flow-Matching
We introduce Consistency-Optimal-Transport-Flow-Matching (COFM) which extends optimal Flow Matching algorithm of transport ODE vector field.
This repository provides the official implementation for our paper: https://arxiv.org/abs/2511.06042
# Requirements
The code has been tested with Python 3.7 and the following packages:

- packaging  
- imageio  
- numpy  
- scipy  
- tqdm  
- dlutils  
- bimpy >= 0.1.1  
- dareblopy >= 0.0.5  
- torch >= 1.3  
- torchvision  
- scikit-learn  
- yacs  
- matplotlib  
# 😊Getting Started
The notebooks for Illustrative 2D Gaussian->Eight Gaussians experiments:
```bash
.\8gaussiantest.ipynb
```

The test notebook for Wasserstein-2 benchmark:
```bash
.\train_hdbm.ipynb
```

The notebooks for ALAE experiment:
```bash
.\train_ALAE.ipynb
.\teat_ALAE.ipynb
```

## 📘 Citation

If you find this work useful, please cite:

```bibtex
@misc{song2025physicsinformeddesigninputconvex,
      title={Physics-Informed Design of Input Convex Neural Networks for Consistency Optimal Transport Flow Matching}, 
      author={Fanghui Song and Zhongjian Wang and Jiebao Sun},
      year={2025},
      eprint={2511.06042},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.06042}, 
}
```
If you have any question about this repository, please feel free to reach me at fanghuisong@stu.hit.edu.cn.
