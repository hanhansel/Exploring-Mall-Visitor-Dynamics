<a id="top"></a>

<div align="center">

# **Exploring Mall Visitor Dynamics**

<b><ins>Crowd counting in shopping malls with density maps: training and comparing CSRNet, MCNN, and ResNet101 to estimate the number of people in a camera frame.</ins></b>

<br>

![Status](https://img.shields.io/badge/status-completed-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

</div>

<br>

<details open>
<summary><b>Table of Contents</b></summary>

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [How It Works](#how-it-works)
4. [Skills Learned](#skills-learned)
5. [Challenges and Solutions](#challenges-and-solutions)
6. [Project Structure](#project-structure)
7. [Getting Started](#getting-started)
8. [Extra Things](#extra-things)
9. [Credits and Contact](#credits-and-contact)

</details>

---

## <ins>Project Overview</ins>

> **TL;DR:** We turned annotated mall CCTV frames into density maps and trained three deep learning models to count people. The smallest model, **MCNN**, won with an **MSE of 8.34**, against 19.00 for ResNet101 and 22.21 for CSRNet.

### ▸ Problem
Shopping malls draw large numbers of visitors. Crowded areas can signal strong business opportunities, but they can also pose safety risks. Understanding foot traffic helps mall operators plan, run daily operations more efficiently, and improve the shopping experience. Counting people by hand from camera footage doesn't scale, so the question is which deep learning model counts people in a frame most accurately.

### ▸ Objectives
- *Convert crowd images with head annotations into density maps* that a model can learn from.
- *Train and compare three architectures* (CSRNet, MCNN and ResNet101) on the same data.
- *Evaluate them with MSE and GAME* to find the most accurate model for estimating crowd counts.

### ▸ Context
Built as a group project for the **Deep Learning** course in my fourth semester at Bina Nusantara University. We presented the results as a slide deck and a research poster.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## <ins>Tech Stack</ins>

> The tools I **used** and what I used them **for**.

| Category | Tool | What I Used It For |
|:---|:---|:---|
| **Language** | `Python` | Data preparation, model training and evaluation |
| **Data Processing** | `pandas`, `NumPy` | Loading the annotation CSV, parsing head coordinates and building density maps |
| **AI / ML** | `TensorFlow` / `Keras`, `OpenCV`, `scikit-learn` | Building and training the three models; reading, resizing and Gaussian-blurring images; splitting the data into train, validation and test sets |
| **Visualization** | `matplotlib` | Plotting annotated frames, density maps, loss curves and predicted vs. ground-truth maps |
| **Libraries** | `SciPy` | Reading `.mat` annotation files for the cross-dataset test |
| **Tools** | Jupyter | Running the experiments in one notebook |

<p align="right">(<a href="#top">back to top</a>)</p>

---

## <ins>How It Works</ins>

> Annotated frames → density maps → three models trained on the same data → compared on MSE and GAME.

```
Mall Frames + Head Annotations → Density Maps → Train CSRNet / MCNN / ResNet101 → Evaluate (MSE, GAME) → Best Model
```

1. **Load the data.** 2,000 mall CCTV frames (640×480), each with the coordinates of every person's head and a total count.
2. **Build density maps.** Each frame is resized to 224×224, the head coordinates are rescaled to match, and a small Gaussian kernel (11×11, σ = 3) is placed on every head. Summing the density map gives the number of people, which avoids having to detect each person separately in a crowd where people overlap.
3. **Split.** 70% training (1,400 frames), 10% validation (200), and 20% test (400). A custom Keras `Sequence` generator loads images and builds their density maps in batches of 4.
4. **Train three models** with Adam (learning rate 1e-4) and MSE loss, for up to 2,000 epochs each, with a callback that stopped training when the model stopped improving:
   - **CSRNet:** a frozen VGG16 front end, dilated convolutions to widen the receptive field without losing resolution, and a decoder that upsamples back to full size.
   - **MCNN:** three convolution columns with different filter sizes, trained from scratch, so it can detect heads that look large near the camera and small far away.
   - **ResNet101:** a frozen, 101-layer pretrained backbone with a decoder, to test whether a much deeper network helps.
5. **Evaluate** on the test set with **MSE** of the predicted counts and **GAME**, then test the models on images from a different dataset to check how well they generalize.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## <ins>Skills Learned</ins>

> What this project taught me, technically and beyond.

- **Density map generation:** Turning point annotations into Gaussian density maps, and rescaling coordinates correctly when images are resized, so the map still sums to the right count. See `DeepLearning_CrowdCounting_SourceCode.ipynb` (inside the zip file).
- **Custom data pipelines:** Writing a Keras `Sequence` generator that loads, resizes, and pairs each image with its density map on the fly instead of holding everything in memory. See `DeepLearning_CrowdCounting_SourceCode.ipynb`.
- **Transfer learning:** Using frozen ImageNet backbones (VGG16 and ResNet101) as feature extractors and training only a new decoder on top. See `DeepLearning_CrowdCounting_SourceCode.ipynb`.
- **CNN architecture design:** Building dilated convolutions, a multi-column network and `Conv2DTranspose` decoders, and understanding why each design suits crowd images. See `DeepLearning_CrowdCounting_SourceCode.ipynb`.
- **Model evaluation:** Implementing MSE and GAME from scratch and comparing models fairly on the same test set. See `DeepLearning_CrowdCounting_SourceCode.ipynb`.
- **Diagnosing overfitting:** Reading training and validation loss curves to spot when a model starts memorizing, and using a custom callback to plot predictions every 100 epochs. See `DeepLearning_CrowdCounting_SourceCode.ipynb`.
- **Presenting research:** Turning the experiments into a slide deck and a research poster. See `PPT.pdf` and `Crowd Counting in Shopping Malls_Poster.png`.

<p align="right">(<a href="#top">back to top</a>)</p>

## <ins>Project Structure</ins>

```
Exploring-Mall-Visitor-Dynamics/
├── DeepLearning_CrowdCounting_SourceCode.zip      # the full Jupyter notebook
├── PPT.pdf                                        # presentation slides
├── Crowd Counting in Shopping Malls_Poster.png    # research poster
└── README.md
```

<p align="right">(<a href="#top">back to top</a>)</p>

---

## <ins>Getting Started</ins>

```bash
# 1. Clone and unzip the notebook
git clone https://github.com/hanhansel/Exploring-Mall-Visitor-Dynamics.git
cd Exploring-Mall-Visitor-Dynamics
unzip DeepLearning_CrowdCounting_SourceCode.zip

# 2. Install
pip install tensorflow==2.10 opencv-python pandas numpy matplotlib scikit-learn scipy jupyter

# 3. Run
jupyter notebook DeepLearning_CrowdCounting_SourceCode.ipynb
```

Download the dataset from [Kaggle](https://www.kaggle.com/code/chaozhuang/crowd-counting-via-density-map-optimized-cnn/input). The notebook expects `output.csv` (counts and head coordinates) and a `frames/` folder of images. Update the hard-coded dataset paths in the notebook (`D:/SEM-4/Deep Learning/GSLC 2/...`) to where you saved them.

> **Requirements:** Python with TensorFlow 2.10 and a GPU. Training all three models for up to 2,000 epochs takes several hours.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## <ins>Extra Things</ins>

<details>
<summary><b>Dataset</b></summary>
<br>

- **Source:** [Crowd counting dataset on Kaggle](https://www.kaggle.com/code/chaozhuang/crowd-counting-via-density-map-optimized-cnn/input)
- **Size:** 2,000 frames from a mall CCTV camera, 640×480 pixels, with about 20–50 people per frame
- **Labels:** *annotations* (x, y coordinates of each head) and *count* (total people per frame)
- **Preprocessing:** Resized frames to 224×224 and scaled pixel values to 0–1, rescaled the head coordinates, generated Gaussian density maps, and split the data 70 / 10 / 20 into train, validation, and test sets

</details>

<details>
<summary><b>Results and Key Findings</b></summary>
<br>

> **Headline:** **MCNN** reached an **MSE of 8.34**, less than half the error of ResNet101 (19.00) and CSRNet (22.21), with the fewest parameters of the three.

| Model | Parameters | MSE ↓ | GAME ↓ |
|:---|:---:|:---:|:---:|
| CSRNet (VGG16 + decoder) | 23.4M | 22.21 | 3.66 |
| ResNet101 + decoder | 53.7M | 19.00 | 3.48 |
| **MCNN** | **0.13M** | **8.34** | **2.12** |

1. **Smaller won:** MCNN's multi-column design handles heads at different sizes, and training from scratch let it fit the crowd task directly instead of reusing classification features.
2. **Pretrained models overfit:** CSRNet and ResNet101 started overfitting after roughly 150–200 epochs.
3. **Better transfer:** On images from a different dataset, MCNN's predictions held up better than the other two models.

</details>

<details>
<summary><b>Model Details</b></summary>
<br>

- **Input / Output:** 224×224×3 image → 224×224×1 density map; the predicted count is the sum of the map
- **Training setup:** Adam, learning rate 1e-4, MSE loss, batch size 4, up to 2,000 epochs with a stopping callback
- **CSRNet:** frozen VGG16 (ImageNet) + 6 dilated convolution layers + `Conv2DTranspose` / upsampling decoder
- **MCNN:** 3 parallel columns with 9×9, 7×7 and 5×5 starting filters, merged into one density map, trained from scratch
- **ResNet101:** frozen ResNet101 (ImageNet) + convolution layers + `Conv2DTranspose` / upsampling decoder

</details>

<details>
<summary><b>Poster</b></summary>
<br>

<p align="center">
  <img src="Crowd%20Counting%20in%20Shopping%20Malls_Poster.png" alt="Research poster" width="650">
  <br><sub><i>Research poster: Uncovering Visitor Trends: Crowd Counting in Shopping Malls</i></sub>
</p>

</details>

<p align="right">(<a href="#top">back to top</a>)</p>

---

## <ins>Credits and Contact</ins>

**Acknowledgements:** Built with my teammates **Alexander Rafael**, **Derrick Vericho**, **Jonathan Sebastian Moiras** and **Jonathan William Gunawan**. Dataset from [Kaggle](https://www.kaggle.com/code/chaozhuang/crowd-counting-via-density-map-optimized-cnn/input). Special thanks to our lecturer, **Mr. Gregorius Natanael Elwirehardja**, for teaching us the fundamentals of deep learning and guiding us through this project.

**License:** Personal Use Only.

**Author:** **Hansel Sebastian**, *Data Science Student @ Bina Nusantara University*

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hanhansel)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/hansel-sebastian-5a19b8245)

<p align="right">(<a href="#top">back to top</a>)</p>
