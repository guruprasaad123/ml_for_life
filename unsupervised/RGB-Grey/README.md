# Grayscale Convertion using autoencoder

In this project we'll make use of autoencoders to learn the process of `grayscaling` the images that are provided to us as input.

# What is an Autoencoder ?

## A quick overview of Autoencoders

Autoencoder is a special type of neural-network that literally copies or mimicks our inputs i.e Not exactly copying but learning the internal representations of our input so that outputs can be generated or mimicked given an input.That is what makes it `Generative Model`.

# Why autoencoders ?

You might have wondered why not use openCV to Generate / Convert the images to `Greyscale` version.Thing is it works but it is not recommended because we want to make a Model to learn the Greyscaling process so that it can be futher enhanced to learn things like Color pop etc..,

In this project we are using CAE ( Convolutional-Autoencoders ) not the Traditional-Autoencoders because of the images are involved and the signals could be convolution of one another.So we can observe the internal representations effectively and make re-constructions of inputs atease.

## Dependencies

You'll need the following dependencies:

- tensorflow ==   1.13.1

- numpy == 1.16.2

- matplotlib ==  2.2.2

You can install all these dependencies using [pip](https://pypi.org/project/pip/) , And issue this command below.

```python
pip install -r requirements.txt
```

## Strategy

## Training

- Folder `training/colored` will be containing the images that our autoencoder takes as the input.

- Folder `training/grey` wil be containng the images that our autoencoder needs to generate as output.

## Testing

- Folder `flower_images` will be containing the images that our autoencoder has never seen. (new Images) which we will input as part of our testing and see if it can generate the `grey_scaled` images meaning that our autoencoder has learned its way of greyscale convertion.

- Folder `generated` is used to store the output ( i.e greyscaled version ) of our autoencoder as we provided the input above ie. from `flower_images` folder

## Working

You'll need to run `demo.py` file in the terminal.

```python
python demo.py
```

Resouces:

[Neural Networks Are Impressively Good At Compression | Probably Dance](https://probablydance.com/2016/04/30/neural-networks-are-impressively-good-at-compression/)

[Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)

[Autoencoders: Deep Learning with TensorFlow’s Eager API | Data Stuff](https://towardsdatascience.com/autoencoders-deep-learning-with-tensorflows-eager-api-data-stuff-378318784ae?source=extreme_main_feed---------6-58--------------------1557645033000)

[https://stackoverflow.com/questions/18870603/in-opencv-python-why-am-i-getting-3-channel-images-from-a-grayscale-image](https://stackoverflow.com/questions/18870603/in-opencv-python-why-am-i-getting-3-channel-images-from-a-grayscale-image)




