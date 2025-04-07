Note : The size of trained model in form of (.keras or .h5) file is ~90 MB which is more than the permissible file size on Github so trained_model.keras file could not be uploaded on this platform , it can be found on [google drive](https://drive.google.com/drive/folders/1-23_a5ajQ7BmwtvDMAbw13_KQIeu8z3d) , also the size of dataset exceeds 2.5 GB as it contains 70000+ images so a compensating test folder is provided which contains some 33 test images

# Disease Detection in Plants Using Leaf Images 🌿🔍

![Project Banner](https://via.placeholder.com/800x300?text=Disease+Detection+in+Plants+Using+Leaf+Images)  
*(Consider adding a real banner image here)*

## 📌 Project Overview
This project implements a deep learning solution to automatically detect diseases in plants by analyzing images of their leaves. The system can classify plant leaves as healthy or identify specific diseases, enabling early intervention to protect crops.

## ✨ Key Features
- 🖼️ Automated disease detection from leaf images
- 🌱 Supports multiple plant species and disease types
- � High-accuracy classification using deep learning
- 📊 Interactive visualizations of model performance
- 🔄 Data augmentation for robust training

## 🛠️ Technologies Used
| Category        | Technologies |
|-----------------|--------------|
| Core Framework  | Python 3, TensorFlow, Keras |
| Visualization   | Matplotlib, Seaborn |
| Environment     | Google Colab |
| Data Source     | [PlantVillage Kaggle Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) |
| Key Algorithms  | Convolutional Neural Networks (CNN) |

## 📂 Dataset
We use the [PlantVillage dataset from Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset):
- 54,305 images of healthy and diseased leaves
- 14 plant species (Tomato, Potato, Pepper, etc.)
- 38 disease categories
- Image resolution: 256×256 pixels

Example classes:
- Tomato Early Blight
- Apple Cedar Rust
- Corn Common Rust
- Grape Black Rot

## 🧠 Model Architecture
**CNN Architecture Diagram**  
*(Insert architecture diagram here)*

The core model consists of:
```python
Model: "Sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 Conv2D (32 filters)         (None, 254, 254, 32)      896       
 MaxPooling2D                (None, 127, 127, 32)      0         
 Conv2D (64 filters)         (None, 125, 125, 64)      18496     
 MaxPooling2D                (None, 62, 62, 64)        0         
 Flatten                     (None, 246016)            0         
 Dense (128 units)           (None, 128)               31490176  
 Dropout (0.5 rate)          (None, 128)               0         
 Dense (38 units)            (None, 38)                4902      
=================================================================
Total params: 31,513,470
Trainable params: 31,513,470
