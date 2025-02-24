# Tumor augmentation

## Task 1 - Tumor cell staining + Missing tumor cell part generation
### Architecture
![](assets/losses.png)
### Consistency objectives
#### [ ] Neighborhood consistency
#### [ ] Global consistency
#### [ ] Local consistency
## Task 2 - Refine + Resolution augmentation 




## Keys
- The backbone of VM is a GAN-based generator G, specifically a CUT model, that consists of 2 sequential components, an encoder G_enc and a decoder G_dec

- The VM is trained under the supervision of 3 levels of consistency objectives: local, neighborhood and global consistency


# Create conda virtual environment
```bash
conda env create -f environment.yml
```

# Install test data (HE and NKX3)

```bash
# At the lab
python -m bin.main --data_dir /storage/apps/hoangthuy.vu/tumor-augmentation-main/data --data_type_src HE --data_type_dst NKX3 --csv_summary_path /storage/apps/hoangthuy.vu/tumor-augmentation-main/data/HE_NKX3.csv

# At home
python -m bin.main --data_dir /Users/vuhoangthuyduong/Documents/icm/tumor-augmentation/data --data_type_src HE --data_type_dst NKX3 --csv_summary_path /Users/vuhoangthuyduong/Documents/icm/tumor-augmentation/data/HE_NKX3.csv
```

# Test model with training_demo
```bash
python -m i2iTranslation.training_demo
```


# References
@article{pati2023multiplexed,
  title={Multiplexed tumor profiling with generative AI accelerates histopathology workflows and improves clinical predictions},
  author={Pati, Pushpak and Karkampouna, Sofia and Bonollo, Francesco and Comperat, Eva and Radic, Martina and Spahn, Martin and Martinelli, Adriano and Wartenberg, Martin and Kruithof-de Julio, Marianna and Rapsomaniki, Maria Anna},
  journal={bioRxiv},
  pages={2023--11},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
} 

@inproceedings{
lipman2023flow,
title={Flow Matching for Generative Modeling},
author={Yaron Lipman and Ricky T. Q. Chen and Heli Ben-Hamu and Maximilian Nickel and Matthew Le},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=PqvMRDCJT9t}
}

https://medium.com/thedeephub/learning-rate-and-its-strategies-in-neural-network-training-270a91ea0e5c

@misc{Gordić2020nst,
  author = {Gordić, Aleksa},
  title = {pytorch-neural-style-transfer},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gordicaleksa/pytorch-neural-style-transfer}},
}