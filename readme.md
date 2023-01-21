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
