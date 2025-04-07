Note : The size of trained model in form of (.keras or .h5) file is ~90 MB which is more than the permissible file size on Github so trained_model.keras file could not be uploaded on this platform , it can be found on [google drive](https://drive.google.com/drive/folders/1-23_a5ajQ7BmwtvDMAbw13_KQIeu8z3d) , also the size of dataset exceeds 2.5 GB as it contains 70000+ images so a compensating test folder is provided which contains some 33 test images

# Disease Detection in Plants Using Leaf Images

## Project Overview
This project implements a deep learning solution to automatically detect diseases in plants by analyzing images of their leaves. The system can classify plant leaves as healthy or identify specific diseases, enabling early intervention to protect crops.

## Key Features
-  Automated disease detection from leaf images
-  Supports multiple plant species and disease types
-  High-accuracy classification using deep learning
-  Interactive visualizations of model performance
-  Data augmentation for robust training

## Technologies Used
| Category        | Technologies |
|-----------------|--------------|
| Core Framework  | Python 3, TensorFlow, Keras |
| Visualization   | Matplotlib, Seaborn |
| Environment     | Google Colab |
| Data Source     | [PlantVillage Kaggle Dataset]([https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)) |
| Key Algorithms  | Convolutional Neural Networks (CNN) |

## Dataset
We use the [PlantVillage dataset from Kaggle]([https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)):
- 87000+ images of healthy and diseased leaves
- 14 plant species (Apple, Soybean, Corn, Tomato, Potato ,etc.)
- 38 disease categories
- Image resolution: 256×256 pixels

Example classes:
- Tomato Early Blight
- Apple Cedar Rust
- Corn Common Rust
- Potato Early Blight

## Model Architecture
**CNN Architecture Diagram**  
![](https://miro.medium.com/v2/resize:fit:1400/0*LeK_gmCf3DfO3gj_.jpeg)


The core model consists of:
```python
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 128, 128, 32)        │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 126, 126, 32)        │           9,248 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 63, 63, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 63, 63, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_3 (Conv2D)                    │ (None, 61, 61, 64)          │          36,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 30, 30, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_4 (Conv2D)                    │ (None, 30, 30, 128)         │          73,856 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_5 (Conv2D)                    │ (None, 28, 28, 128)         │         147,584 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_2 (MaxPooling2D)       │ (None, 14, 14, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_6 (Conv2D)                    │ (None, 14, 14, 256)         │         295,168 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_7 (Conv2D)                    │ (None, 12, 12, 256)         │         590,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_3 (MaxPooling2D)       │ (None, 6, 6, 256)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_8 (Conv2D)                    │ (None, 6, 6, 512)           │       1,180,160 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_9 (Conv2D)                    │ (None, 4, 4, 512)           │       2,359,808 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_4 (MaxPooling2D)       │ (None, 2, 2, 512)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 2, 2, 512)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 2048)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 1500)                │       3,073,500 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 1500)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 38)                  │          57,038 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 7,842,762 (29.92 MB)
 Trainable params: 7,842,762 (29.92 MB)
 Non-trainable params: 0 (0.00 B)
