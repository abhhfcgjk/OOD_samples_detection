# Detect Out-Of-Distribution samples

## Stage 1
Generate **Out-Of-Distribution** for selected model according Euclidean-norm and Mahalanobis distance.</br>
To shift samples from the distribution use an **adversarial attack**.</br>
![PCA](image.png)

## Stage 2
To obtain a degree of confidence that the data are in the original distribution, a modification of the original model is used.</br>
The modified model returns an embedding vector and In-Distribution probability.</br>
To find the ID probability the Mahalanobis distance is used.</br>
| ![ROC curve](image2.png) |
|:--:|
| *ROC curve with OOD predictions* |