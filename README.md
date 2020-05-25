# Linknet-NNFL-Project
Implementation of paper: __Linknet: Exploiting Encoder Representations for Efficient Semantic Segmentation__

# Instructions
The linknet_model.py file contains the code for the implemented linknet model. The dataloaders and training codes are in the respective train notebooks.

Run all cells in a notebook to train a linknet model and evaluate it.

# Camvid results
| Model | IoU |
|:--------:|:---------:|
| Segnet| 65.2|
| Enet| 68.3|
| Dilation8 | 65.3|
| Linknet(original) | 68.3|
| Linknet(ours) | 67.6 |
| Linknet(with modifications)| 65.2|
