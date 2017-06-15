# AnimeRecommendation
This is an implementations of various recommender systems [using a MyAnimeList (MAL) dataset from Kaggle](https://www.kaggle.com/CooperUnion/anime-recommendations-database) using the latent factor model (matrix factorization), using tensorflow. Before I begin, let me note where to acquire the data from.

## Getting the pre-requisite data and models for this project.
There are two sets of files that you'll need. The easier one is obtaining the actual anime data. That can be downloaded from Kaggle via the link above. It's probably easiest to leave it in the root directory of this repo and then place a copy of it in the subdirectories, just because the pathing and such is in need of refactoring. I'll do that when I have the chance. The other set of data is a little bit more difficult, and it's only relevant if you are interested in doing the part of this project with deep CNNs.

### Getting the CNN models (into tensorflow)
This repository uses the [Illustration2Vec](http://illustration2vec.net/) CNN model (Saito & Matsui, 2015) which is a CNN trained on anime images. Specifically, it uses the pre-trained model that gets a feature representation of an input image. That process is done in `create_anime_image_data` in `util/data`. Illustration2Vec has a convenient Python interface if you use caffe or chainer, but I do not. Consequently, I converted the caffe model to a tensorflow model using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) on an AWS instance, as tf is built into my laptop. If you have caffe or chainer installed, you can modify `create_anime_image` to use caffe or chainer. If you not, you'll have to install one, or convert it to tensorflow. The data file is quite large (nearly 1 GB), so getting caffe or chainer to work is the easiest way. If not, familiarize yourself with `caffe-tensorflow` and ping me to send you the trained model.

You only need to run `create_anime_image_data` once and after that it'll save a `dict` as a `pickle` file so that you can simply load it up.

## Understanding the layout of this repository 
In the root repository lies `recsys.py`. This file (which will need to be refactored) holds the actual implementation of the latent factor model for this project. It is not an API (yet), i.e. you can't use it as a generic implementation of a latent factor model for another project. But it does let you choose to build a model and then make predictions for a specific user. You can see this as the 'core' file. There are also three subdirectories of note.

`util` is exactly as it sounds: it is a subdirectory that holds various utility files, which includes creating data and extracting information from a user's MAL profile.

`Illustration2Vec` is the code needed to run the CNN. Specifically, there are the relevant `caffe-tensorflow` files needed to run, and they also hold the model and the data (which are not on Github but on my local machine). 

`notebooks` is a directory that holds my Jupyter (iPython) notebooks. Some of these are exploratory notebooks for the convience of prototyping various python functionality (e.g. how to use MAL's API) and others are essentially implementations of various latent factor models (just testing against a validation set -- no user-specific predictions as in `recsys.py`).

-----

# An overview of the Collaborative Filtering Latent Factor Methods (via Matrix Factorization)
In this portion of the README, I will be discussing the various latent factor methods (a collaborative filtering method also known in the literature as matrix factorization) that are used in this project. I really recommend Koren, Bell, and Volinsky (2009) for an overview of the primary techniques. Note that whereas in the Netflix Challenge, the goal is to minimize MSE, here I try to minimize MAE. This is for no reason other than MAE is very easily interpretable. I will at some point update this code to work in MSE as that's standard.

### The trivial model: returning the global mean
The most trivial recommender system is one that simply predicts the global mean. Specifically:

`R(u,i) = α`

This recommender system doesn't care who the user is or what the item is. It just predicts the global mean. That is the absolute baseline, and I didn't actually write a notebook for it. I only list it here for comparisons.

### Adding in user-specific and item-specific bias terms
`R(u,i) = α + βᵢ + βᵤ`

This new model expands from the previous term by adding in an item-specific and user-specific bias (offset) term. So if a user is a harsh user and/or if an item sucks, they'll have strongly negative offsets so that the predicted rating falls far below the global mean `α`. 

This model actually works well in minimizing MSE or MAE. The problem with this model is that there is no actual recommendation going on. Each person is recommended the same item. So while it is a decent predictive system, it's a poor recommendation system. Finally, note that this model can run into the problem of overfitting. Consequently, we add L2 regularization to our objective function.

`Loss(u,i) = |Rᵤ,ᵢ - (α + βᵢ + βᵤ)| + λ * (βᵢ² + βᵤ²)`

λ is a hyper-parameter that controls the amount of regularization. I often set it to be 1, although a more principled choice would be to use a validation set to choose its value. Finally, note that L2 regularization on a variable is well known to be equivalent to putting a Gaussian prior on it. There are actually probabilistic  interpretations of our models that I may explore (e.g. Salakhutdinov & Mnih, 2008 and 2009).

### Adding in latent factors
Remember that in the previous weakness, there's no actual recommendation going on. There's no interaction between users and items. In the latent factor model, that issue is remedied.

In the latent factor model, our goal is to estimate the complete M x N matrix of ratings, where M is the number of users and N is the number of items. In other words, we want to estimate the complete matrix of every user/ item rating pair. There are a few ways to do this (e.g. SVD-based), but the approach we take here is fairly simple. We approximate the M x N matrix as the product of two low rank submatrices that are M x k and k x N (k is a hyper-parameter). The M x k matrix can be seen as mapping a user's preferences into some k-dimensional space, and the k x N matrix can be seen as mapping an item's qualities into the same k-dimensional space. We measure the similarity of a user's preferences and item's qualities by the dot product, and we add that to our prediction of the rating. So now we have:

`R(u,i) = α + βᵢ + βᵤ + γᵤγᵢ`

where γᵤ represents a user's latent preferences and γᵢ represents an item's latent qualities. I use L2 regularization on these as well, so the loss function becomes:

`Loss(u,i) = |Rᵤ,ᵢ - (α + βᵢ + βᵤ + γᵤγᵢ)| + λ * (βᵢ² + βᵤ² + γᵤ² + γᵢ²)`
