# udacity-mlnd-ev-capstone  
  
Capstone Project for the Udacity Machine Learning Nanodegree  
  
Salvador Joel Núñez Gastélum  
  

## Project Overview

This project applies data augmentation techniques and a combination of k-means clustering, K-Nearest Neigbors (KNN), and Gradient Boosted Decision Trees (GBDT) to predict whether there is an Electric Vehicle (EV) charging at a house during a specific time frame.

For more details, see original data set [here](https://www.gridcure.com/wp2/wp-content/uploads/2016/05/EV_files-2.zip). 



## Project Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/torenunez/udacity-mlnd-ev-capstone.git
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

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `ev-capstone` environment.  Open the notebook.
```
python -m ipykernel install --user --name ev-capstone --display-name "EV Capstone Project"
jupyter notebook ev-capstone.ipynb
```

5. Before running code, change the kernel to match the ev-capstone environment by using the drop-down menu (**Kernel > Change kernel > ev-capstone**). 
