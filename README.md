# udacity-mlnd-ev-capstone
Capstone Project for the Udacity Machine Learning Nanodegree


## Project Overview

TBD


## Project Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/torenunez/udacity-mlnd-ev-capstone.gitt
cd udacity-mlnd-ev-capstone
```

2. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```
	conda create --name ev-capstone python=3.6
	source activate ev-capstone
	```  
	- __Windows__: 
	```
	conda create --name ev-capstone python=3.6
	activate ev-capstone
	```

3. Install a few pip packages.
```
pip install -r requirements.txt
```

4. Install TensorFlow. 
	__To install TensorFlow with CPU support only__ on a local machine:
		```
		pip install tensorflow==1.1.0
		```

5. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

6. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `ev-capstone` environment.  Open the notebook.
```
python -m ipykernel install --user --name ev-capstone --display-name "EV Capstone Project"
jupyter notebook ev-capstone.ipynb
```

7. Before running code, change the kernel to match the ev-capstone environment by using the drop-down menu (**Kernel > Change kernel > ev-capstone**). 
