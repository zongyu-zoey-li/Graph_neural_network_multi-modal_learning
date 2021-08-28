
# Graph neural network multi-modal learning

## Motivation:
* How to use graph neural network to both extract the feature information and to integrate the videos and kinematics data for error detection has yet to be explored. In this project, we address the problem of integrating both kinematics and videodata for erroneous gesture classification with GNN models.
* We propose a pipeline based on graph neural network to integrate both the videos and the kinematics information for erroneous gesture detection. 
## Background

Mathematically, the GCN model follows the following formula. 

<img src="https://render.githubusercontent.com/render/math?math=H^{(l+1)}=\sigma(D^{-\frac{1}{2}}AD^{-\frac{1}{2}}H^{(l)}W^{{l}})">

where <img src="https://render.githubusercontent.com/render/math?math=H^{(l)}"> denotes the <img src="https://render.githubusercontent.com/render/math?math=l^{th}"> layer in the network, <img src="https://render.githubusercontent.com/render/math?math=\sigma"> is the non-linearity, and <img src="https://render.githubusercontent.com/render/math?math=W"> is the weight matrix for this layer. D and A represent degree matrix and adjacency matrix.

In this work, we use a two layered GCN, and the input graph is a fully connected graph with 5 nodes. The last layer's embeddings of the 5 nodes are then concatenated and fed into a regular neural network to classify the input as either erroneous or normal. 

<img src="https://render.githubusercontent.com/render/math?math=Class=\sigma(concat[h_{1},h_{2},h_{3},h_{4},h_{5}]W+b)">
	     
where <img src="https://render.githubusercontent.com/render/math?math=h_{1},h_{2},h_{3},h_{4},h_{5}"> are the embeddings of the 5 nodes. We use the binary cross-entropy loss as our loss function shown below.

<img src="https://render.githubusercontent.com/render/math?math=L=-1/N\sum_{i=1}^{N}y_{i}\cdot log(p(y_{i}))+(1-y_{i})\cdot log(1-p(y_{i}))">
	     
## Results
We evaluated our proposed system onthe JIGSAWS data set. In the experimental evaluation, the proposed method achieved 0.71 F1-Score.
