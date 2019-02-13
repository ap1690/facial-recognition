# facial-recognition
Simple facial recognition with neural networks using numpy

### Script organization:

* `generate_facedata`: generates training data for facial recognition using OpenCV's haar cascade
* `pre_process`: a set of functions used for training and prediction scripts
* `process`: handles csv file generation. There is no particular reason for having separate `process` and `pre_process` scripts other than for easy understanding.
* `train_simplenn`: trains a simple feed forward neural network to detect faces.
* `predict`: uses trained neural network to make recognize faces in a video.

### Code maintainenace
This code was written while learning the basics of neural networks. This is a nice project for a beginner to train a neural network without using any libraries. The trained network can run in semi-realtime, even on a raspberry pi. As the focus of the project is learning, the code in this repo is not very well written and does not conform to best practices. Do not use this code in production, except if you are deploying on a raspberry pi. The script for inference on a raspberry pi may be provided on request. Please open an issue if you would like me to share the script. Other than that, this code will not be maintained.
