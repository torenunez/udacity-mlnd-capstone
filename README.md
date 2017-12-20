# udacity-mlnd-capstone
Capstone Project for the Udacity Machine Learning Nanodegree


## Project Overview

TBD


## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/torenunez/udacity-mlnd-capstone.git
cd udacity-mlnd-capstone
```

3. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```
	conda create --name dog-project python=3.6
	source activate dog-project
	```  
	- __Windows__: 
	```
	conda create --name dog-project python=3.6
	activate dog-project
	```

6. Install a few pip packages.
```
pip install -r requirements.txt
```

7. Install TensorFlow. 
	- Option 1: __To install TensorFlow with GPU support__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step and only need to install the `tensorflow-gpu` package:
		```
		pip install tensorflow-gpu==1.1.0
		```
	- Option 2: __To install TensorFlow with CPU support only__ on a local machine:
		```
		pip install tensorflow==1.1.0
		```

8. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

9. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment.  Open the notebook.
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
jupyter notebook dog_app.ipynb
```

10. Before running code, change the kernel to match the dog-project environment by using the drop-down menu (**Kernel > Change kernel > dog-project**). Then, follow the instructions in the notebook.

__NOTE:__ While some code has already been implemented to get you started, you will need to implement additional functionality to successfully answer all of the questions included in the notebook. __Unless requested, do not modify code that has already been included.__

## Evaluation

Your project will be reviewed by a Udacity reviewer against the CNN project [rubric](#rubric).  Review this rubric thoroughly, and self-evaluate your project before submission.  All criteria found in the rubric must meet specifications for you to pass.


## Project Submission

When you are ready to submit your project, collect the following files and compress them into a single archive for upload:
- The `dog_app.ipynb` file with fully functional code, all code cells executed and displaying output, and all questions answered.
- An HTML or PDF export of the project notebook with the name `report.html` or `report.pdf`.
- Any additional images used for the project that were not supplied to you for the project. __Please do not include the project data sets in the `dogImages/` or `lfw/` folders.  Likewise, please do not include the `bottleneck_features/` folder.__

Alternatively, your submission could consist of the GitHub link to your repository.
