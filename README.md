# AnimeRecommendation
This is an implementations of various recommender systems [using a MyAnimeList (MAL) dataset from Kaggle](https://www.kaggle.com/CooperUnion/anime-recommendations-database) using the latent factor model (matrix factorization). Before I begin, let me note where to acquire the data from.

## Getting the pre-requisite data and models for this project.
There are two sets of files that you'll need. The easier one is obtaining the actual anime data. That can be downloaded from Kaggle via the link above. It's probably easiest to leave it in the root directory of this repo and then place a copy of it in the subdirectories, just because the pathing and such is in need of refactoring. I'll do that when I have the chance. The other set of data is a little bit more difficult, and it's only relevant if you are interested in doing the part of this project with deep CNNs.

#### Getting the CNN models (into tensorflow)
This repository uses the [Illustration2Vec] (http://illustration2vec.net/) CNN model (Saito & Matsui, 2015) which is a CNN trained on anime images. Specifically, it uses the pre-trained model that gets a feature representation of an input image. That process is done in `create_anime_image_data` in `util/data`. Illustration2Vec has a convenient Python interface if you use caffe or chainer, but I do not. Consequently, I converted the caffe model to a tensorflow model using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) on an AWS instance, as tf is built into my laptop. If you have caffe or chainer installed, you can modify `create_anime_image` to use caffe or chainer. If you not, you'll have to install one, or convert it to tensorflow. The data file is quite large (nearly 1 GB), so getting caffe or chainer to work is the easiest way. If not, familiarize yourself with `caffe-tensorflow` and ping me to send you the trained model.

You only need to run `create_anime_image_data` once and after that it'll save a `dict` as a `pickle` file so that you can simply load it up.

## Understanding the layout of this repository 
In the root repository lies `recsys.py`. This file (which will need to be refactored) holds the actual implementation of the latent factor model for this project. It is not an API (yet), i.e. you can't use it as a generic implementation of a latent factor model for another project. But it does let you choose to build a model and then make predictions for a specific user. You can see this as the 'core' file. There are also three subdirectories of note.

`util` is exactly as it sounds: it is a subdirectory that holds various utility files, which includes creating data and extracting information from a user's MAL profile.

`Illustration2Vec` is the code needed to run the CNN. Specifically, there are the relevant `caffe-tensorflow` files needed to run, and they also hold the model and the data (which are not on Github but on my local machine). 

`notebooks` is a directory that holds my Jupyter (iPython) notebooks. Some of these are exploratory notebooks for the convience of prototyping various python functionality (e.g. how to use MAL's API) and others are essentially implementations of various latent factor models (just testing against a validation set -- no user-specific predictions as in `recsys.py`).

-----

# An overview of the Latent Factor Methods (based on Matrix Factorization)

TBD
