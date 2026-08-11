Gaitpattern Dataset
===================

The dataset contains skeleton sequences of people walking in a given pace and steplength behind a mobile robot, which was recording them with a Azure Kinect RGB-D camera.
Originally, there are MIRA-Tapes of the individual trials, from which the skeleton sequences have been extracted and stored as .npy files. There was some augmentation of speed and scale applied during export to yield five times more data samples.

There is a Dataset class for pytorch, which allows to read the sequences from the exported .npy files.

The data has been split into train, validation, and test, while the list of belonging samples is given in the train.txt, val.txt, and test.txt respectively.

The labels folder contains the global labels for the individual trials which were defined at recoding time. (augmentation has been applied to them as well)
The computed_labels_w30 contain a set of gait parameters which have been extracted from 30 frames long time windows of the original sequences.

The skeleton sequences are given as torch.Tensors of the shape [T,J,D], where T are the time frames, J are the joints in the order given in joint_order.txt, ande D are the x,y,z coordinates in a world centered reference frame.

The Dataset class yields samples which are sections of the sequences of a given length in frames. The Frames are recorded with 20Hz.
In addition the Dataset iterator provides the normalized samples, which are given in a human cenetered coordiante system.
The normalization.py describes this encoding.

