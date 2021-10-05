# FGCNN 
> Feature Generation by Convolutional Neural Networks.


Abstract taken from [paper](https://arxiv.org/abs/1904.04447)
> Click-Through Rate prediction is an important task in recommender systems, which aims to estimate the probability of a user to click on a given item. Recently, many deep models have been proposed to learn low-order and high-order feature interactions from original features. However, since useful interactions are always sparse, it is difficult for DNN to learn them effectively under a large number of parameters. In real scenarios, artificial features are able to improve the performance of deep models (such as Wide & Deep Learning), but feature engineering is expensive and requires domain knowledge, making it impractical in different scenarios. Therefore, it is necessary to augment feature space automatically. In this paper, We propose a novel Feature Generation by Convolutional Neural Network (FGCNN) model with two components:Feature Generation and Deep Classifier. Feature Generation leverages the strength of CNN to generate local patterns and recombine them to generate new features. Deep Classifier adopts the structure of IPNN to learn interactions from the augmented feature space. Experimental results on three large-scale datasets show that FGCNN significantly outperforms nine state-of-the-art models. Moreover, when applying some state-of-the-art models as Deep Classifier, better performance is always achieved, showing the great compatibility of our FGCNN model. This work explores a novel direction for CTR predictions: it is quite useful to reduce the learning difficulties of DNN by automatically identifying important features.

## Install

`pip install your_project_name`

## How to use

Fill me in please! Don't forget code examples:

```python
#slow


# prepare data loaders
dls     = get_dl()

# prepare embedding matrix
emb_szs = get_emb_sz(dls.train_ds, k=40)

# prepare model architecture
m = FGCNN(emb_szs=emb_szs, 
             conv_kernels=[14, 16, 18, 20], 
             kernels=[3, 3, 3, 3],
             dense_layers=[4096, 2048, 1024, 512],
             h=7,
             hp=2
            )

# create tabular learner
learn = TabularLearner(dls, m, loss_func=BCELossFlat(), opt_func=ranger)

# train and validate
learn = train(dls, m, lr, loss_func=BCELossFlat(), n_epochs=1)
```
