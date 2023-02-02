# PONG-atari-without-using-convolutional-networks-CNN

Dear friends. In this work, I want to show how I implemented the training of a neural network playing Pong without using convolutional neural networks.

A few words about the system requirements. To implement this project, I used a computer with Ubuntu 20.04 on board. All dependencies are packed into a file "requirements.txt ". So, let's go through the process of preparing for training and inference of the model step by step.

To begin with, we will create and activate a virtual environment.

```
python3 -m venv pong
source pong/bin/activate
```

`Update the package index`

```
sudo apt update
```

`Install pip and Python headers`

```
sudo apt install python3-pip python3-dev
```

`Finally install Jupyter`

```
pip install jupyter
```

Install dependencies and add virtual environment to the Python Kernel

```
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=pong
```

`Now you can run Jupiter with the command`

```
jupyter notebook
```

`After that, go to the menu and select: Kernel > Change kernel > pong`

`This is important - our notebook must work in a virtual environment created by us.`

`Now let's load all the dependencies from the file "requirements.txt ".`

```
pip install -r requirements.txt
```

`Okay, now we're ready to start. Open the file "PONG_hw.ipynb" in jupiter. We will skip the first point, since we have already installed all the dependencies.`

**The second point** is to install the Atari game package, which includes Pong.

Not all commands can work correctly. The main thing is to understand the essence and, if necessary, correct something. The bottom line is that we download the game package from the link. Unzip it. We get two folders - "ROMS" and "HC ROMS". Create a new folder "rars" and copy both unzipped folders into it. Next, from the directory with the folder "rars", run the command "python -m atari_py.import_roms rars".

By **the third and fourth point**, we simply import all the necessary libraries and install the game environment. Check the version of "gym" - it should be "0.15.3". This is important - newer versions will no longer work on this code.

**The fifth point** is one of the most important. This is the essence of my method of teaching an agent to play Pong.

Usually, convolutional neural networks are used for such training, that is, they process the entire picture corresponding to each frame of the game as a whole. This leads to a huge number of trainable parameters - about 3.5 million. Here is an example of such an architecture.

![CNN architecture](https://github.com/mic21053/PONG-atari-without-using-convolutional-networks-CNN-/blob/main/illustrations/%D0%9E%D0%B1%D1%80%D0%B0%D0%B7%D0%B5%D1%86%20%D0%B0%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80%D1%8B.png)

My idea is to translate each state of the game (that is, each frame) into an embedding vector, in which we encrypt only the data that matters for winning. So, we begin to get rid of excess. Here is our frame in its original form.

![](https://github.com/mic21053/PONG-atari-without-using-convolutional-networks-CNN-/blob/main/illustrations/1.png)

First, we will convert the color image to shades of gray - so we will get rid of the two dimensions of the channels. From the picture 210x160x3 we get a picture 210x160.

![](https://github.com/mic21053/PONG-atari-without-using-convolutional-networks-CNN-/blob/main/illustrations/2.png)

The field in which the rackets and the ball can move is not the whole picture. The strip on top with the score and the strip on the bottom do not affect the game in any way. Therefore, we will cut the picture from the top and bottom. We will get a 160x160 picture.

![](https://github.com/mic21053/PONG-atari-without-using-convolutional-networks-CNN-/blob/main/illustrations/3.png)

And in fact, in this frame, we only care about the position of our racket and the position of the ball. The position of the opponent's racket is not important to us - our task is to trace the trajectory of the ball flying in our direction and put our racket under it in time. And where this ball will fly from is not important to us. Looking at the brightness values of the current state, we will see that the brightness value of our racket is 0.57713255, and the brightness value of the ball is 0.92539765. Using "np.where" we will find the coordinates of our racket and ball (namely, the upper left corner and the lower right corner) and write them as the embedding vector. If there is no racket or ball on the screen at the moment, we will write (-1, -1, -1, -1). "gray_small" in the figure corresponds to "gray" in the code.

![](https://github.com/mic21053/PONG-atari-without-using-convolutional-networks-CNN-/blob/main/illustrations/4.png)

![](https://github.com/mic21053/PONG-atari-without-using-convolutional-networks-CNN-/blob/main/illustrations/5.png)

![](https://github.com/mic21053/PONG-atari-without-using-convolutional-networks-CNN-/blob/main/illustrations/6.png)

At the output, we also glue the batch dimension to the resulting vector.

Thus, we encrypted the state of the game through 8 digits.

In **the sixth point**, we set all the hyperparameters of the model.

In **the seventh point**, we will create a function that builds the architecture of our model. The most interesting layer is the first one. This is a recurrent layer. It accepts as input, in addition to the current one, several other previous states of the environment in the amount specified in the "RNN_LENGTH" hyperparameter. This allows us to take into account the dynamics of the ball's movement, that is, its trajectory.
Next, we will create two models with this architecture. They will be absolutely identical, except for the weights, which will be initialized randomly.
Let's take a look at the "summary" of this model.

![](https://github.com/mic21053/PONG-atari-without-using-convolutional-networks-CNN-/blob/main/illustrations/7.png)

As we can see, the number of parameters differs 455 times - 3358374 parameters for a model built on a convolutional network, against 7382 parameters for my model. Moreover, the efficiency can be adjusted very well by playing with hyperparameters. At the same time, the network is trained for quite a long time anyway - therefore, due to lack of time, I did not select ideal hyperparameters. If you want, do it. In this work, the main thing for me is to show the very principle of how my network works.

In **the eighth point**, we will make the weights of our two created networks the same. To do this, we will write a small function, since we will have to do this copying more than once.

**The ninth point** is divided into two sub-points - 9.1 and 9.2. Now the code in paragraph 9.1 is commented out. But when you start training for the first time, you need to uncomment and execute it. Since the training goes on for a very long time, and the computer can suddenly stop working for various reasons (for example, the lights are turned off), in order not to lose progress, our training saves these five parameters, as well as the model and its weights, every 100 iterations of our training. So if the work was interrupted, then in order to restore the training from the last checkpoint, we will have to comment out paragraph 9.2, and on the contrary, comment out paragraph 9.1 and we will not refer to it again in the future.

In **the tenth point**, we finally start learning. Moreover, let's pay attention to the "range" function in the cycle of iterations by episodes. During the initial training, it will look like "range(NUM_EPISODES)". If we resume work from a checkpoint, then we need to add an initial value equal to the checkpoint number + 1. That is, "range(<checkpoint + 1>, NUM_EPISODES)".

In **the eleventh point**, we save the model and weights in different formats to the "Finish_model" folder. Next, to load the finished model, you just need to run the command "model = load_model('Finish_Model/MODEL')".

Finally, in **the twelfth point**, we visualize the statistical data collected during the training and look at the independent game of the finished model.

![](https://github.com/mic21053/PONG-atari-without-using-convolutional-networks-CNN-/blob/main/illustrations/8.png)
