General information
===================

This application is made for analyzing sensor data, and interpreting said data for result analysis. The main function of the application is to analyze datasets utilizing the Savitzky-Golay filter and provide visualizations based on analyzed data.
Expected input data is produced by two distinct sensors, each polling at 50 000 samples per second. The expected input is a .csv or .npy file containing either the raw unanalyzed data in two columns (adc1,adc2), or a previously analyzed dataset with two timestamp columns and a label column (startTime,endTime,label). The intended use is for analysis of data obtained from medical instruments, but the application can analyze any data as long as it is provided with correctly formatted columns.

The application can be used for analysis of the directly produced graphs, as well as converting the large direct datasets to a smaller, easily loadable size written to an .npy file before analysis.


This application is built using Python 3.12, and requires the following additional libraries to function:

Pandas, Numpy, MatPlotLib, Scipy

These libraries are present in the requirements.txt file, and should be installed before attempting to run the application.


Installation and execution
==========================

Running from code
-----------------

The directory includes all source code for the program, and can be used to run and edit the program as the user sees fit.


Step 1. Cloning the directory
-----------------------------

First navigate to your desired install directory using the `cd` command, for example

`cd C:\Users\Johnny\PycharmProjects\target`

Using git, clone the directory either directly with the command

`git clone https://github.com/LeeviMakinen/ICTProject1.git`

or using your IDE of choice.


Step 2. Verifying essentials
----------------------------

Ensure you have a functional python installation, for example with the command

`python --version` or other variants such as `py --version` depending on installation

Also ensure you have a functional version of python`s package manager pip with a similar command

`pip --version`


(Step 3. Installing a virtual environment)
------------------------------------------

Installing a virtual environment is not mandatory, but installing all dependencies system-wide could potentially cause permission issues.
To allow for a virtual environment, install virtualenv either with the command

`pip install virtualenv`

before activating it using the following commands

`python -m venv venv`

`venv\Scripts\activate`

Or create one using your IDE.

Exiting a virtual environment can be done with the command

`deactivate`


Step 4. Installing dependencies
-------------------------------

As the program needs additional libraries to run aside from base python, you need to install all dependencies listed above.
This can be done automatically using pip with the command

`pip install -r requirements.txt`

In case of errors, each dependency can be installed separately using the command

`pip install <name_of_dependecy>`, for example `pip install scipy`


Step 5. Running the program
---------------------------

Once all dependencies have been installed, run the program with the command

`python main.py` (in some installations shortened to `py main.py`)

or by launching the main.py file from your IDE



Usage of application
====================

The basic workflows of the application are as follows:

#Using all features

Flow 1: Open program > Load CSV-file > Convert to NPY > Load NPY > Adjust sliders for analysis > Update analysis (optionally > Export peaks)


#Analyzing previously converted data

Flow 2: Open program > Load NPY > Adjust sliders for analysis > Update analysis (optionally > Export peaks)


#Loading an exported set of peaks

Flow 3: Open program > Import peaks

**Note that the program does not support re-exporting a peak file that is loaded for visualization**





