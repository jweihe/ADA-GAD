## [ADA-GAD: Anomaly-Denoised Autoencoders for Graph Anomaly Detection](https://arxiv.org/abs/2312.14535)
[comment]: <> ([Paper]&#40;https://arxiv.org/abs/2312.14535&#41)
<a href="https://arxiv.org/abs/2312.14535"><img src="https://img.shields.io/badge/arXiv-2312.14535-b31b1b.svg" height=22.5></a>

<!-- This repository contains a PyTorch implementation for our paper "ADA-GAD: Anomaly-Denoised Autoencoders for Graph Anomaly Detection". -->

![Teaser image](./main_photo.png)
> **Abstract:** *Graph anomaly detection is crucial for identifying nodes that deviate from regular behavior within graphs, benefiting various domains such as fraud detection and social network. Although existing reconstruction-based methods have achieved considerable success, they may face the \textit{Anomaly Overfitting} and \textit{Homophily Trap} problems caused by the abnormal patterns in the graph, breaking the assumption that normal nodes are often better reconstructed than abnormal ones. Our observations indicate that models trained on graphs with fewer anomalies exhibit higher detection performance. Based on this insight, we introduce a novel two-stage framework called Anomaly-Denoised Autoencoders for Graph Anomaly Detection (ADA-GAD). In the first stage, we design a learning-free anomaly-denoised augmentation method to generate graphs with reduced anomaly levels. We pretrain graph autoencoders on these augmented graphs at multiple levels, which enables the graph autoencoders to capture normal patterns. In the next stage, the decoders are retrained for detection on the original graph, benefiting from the multi-level representations learned in the previous stage. Meanwhile, we propose the node anomaly distribution regularization to further alleviate \textit{Anomaly Overfitting}. We validate the effectiveness of our approach through extensive experiments on both synthetic and real-world datasets.*


