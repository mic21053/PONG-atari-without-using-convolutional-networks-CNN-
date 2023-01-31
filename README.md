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

![CNN architecture](https://github.com/mic21053/PONG-atari-without-using-convolutional-networks-CNN-/blob/main/Образец%20архитектуры.png)

My idea is to translate each state of the game (that is, each frame) into an embedding vector, in which we encrypt only the data that matters for winning. So, we begin to get rid of excess. Here is our frame in its original form.

![](media/38780debb562ddbb7e49e9fba5417e88.png)

First, we will convert the color image to shades of gray - so we will get rid of the two dimensions of the channels. From the picture 210x160x3 we get a picture 210x160.

![](media/156083e5b7533b5a5e97d2842145df97.png)

The field in which the rackets and the ball can move is not the whole picture. The strip on top with the score and the strip on the bottom do not affect the game in any way. Therefore, we will cut the picture from the top and bottom. We will get a 160x160 picture.

![](media/434d3ec74e566da76d612d849ebb2caa.png)

And in fact, in this frame, we only care about the position of our racket and the position of the ball. The position of the opponent's racket is not important to us - our task is to trace the trajectory of the ball flying in our direction and put our racket under it in time. And where this ball will fly from is not important to us. Looking at the brightness values of the current state, we will see that the brightness value of our racket is 0.57713255, and the brightness value of the ball is 0.92539765. Using "np.where" we will find the coordinates of our racket and ball (namely, the upper left corner and the lower right corner) and write them as the embedding vector. If there is no racket or ball on the screen at the moment, we will write (-1, -1, -1, -1). "gray_small" in the figure corresponds to "gray" in the code.

![](media/87f1b9db311f28563333dea16f9abdc5.png)

![](media/bdf45b664296789d9fb70c34b208690c.png)

![](media/51570c1b6de59ed1f288924c0480b98d.png)

At the output, we also glue the batch dimension to the resulting vector.

Thus, we encrypted the state of the game through 8 digits.
