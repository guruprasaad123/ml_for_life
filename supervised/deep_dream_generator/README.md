# Deep Dream

## Description

We replicate Google's Deep Dream code in of Python using the Tensorflow machine learning library and opencv.
Then we visualize it at the end.

In Nullshell ,what we are doing here is that we using Conventional Neural Networks aka CNN to apply apply gradient-descent to each Frame in case of video and each image in case of Picture and Save it using [opencv](https://docs.opencv.org/3.0-beta/index.html).

## Dependencies

- numpy_1.16.2 ( [https://pypi.org/project/numpy/](https://pypi.org/project/numpy/) )

- tensorflow_1.13.1 ( [https://pypi.org/project/tensorflow-gpu/](https://pypi.org/project/tensorflow-gpu/) )
- opencv_4.0.0 ( [https://pypi.org/project/opencv-python/](https://pypi.org/project/opencv-python/))

- matplotlib_3.0.3( [https://pypi.org/project/matplotlib/](https://pypi.org/project/matplotlib/) )

## Running the app

Issue this command to run the project .

For video 

```python
python demo.py <input.mp4> --video <output.mp4>(optional) 
```

For Images

```python
python demo.py <input.jpg> --image <output.jpg>(optional) 
```

 Resources :

 [https://github.com/martinwicke/tensorflow-tutorial/blob/master/3_deepdream.ipynb](https://github.com/martinwicke/tensorflow-tutorial/blob/master/3_deepdream.ipynb)

[https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html)

[https://www.tutorialkart.com/opencv/python/opencv-python-save-image-example/](https://www.tutorialkart.com/opencv/python/opencv-python-save-image-example/)
