=====================================================================
!!!THESE INSTRUCTIONS HAVE ONLY BEEN TESTED ON PYTHON -V == 3.11.1!!!
=====================================================================

To run:

0. git clone https://github.com/cameronbuster/cbuster6-supervised-learning.git
1. Create virtual environment in your terminal. "python -m venv venv"
2. Activate this venv. If Windows, ".\venv\Scripts\activate". If macOS, "source venv/bin/activate".
3. Install requirements. "pip install -r requirements.txt"
4. Run entrypoint.py.

All directories should be created for you. All files should be generated for you. All figures and
models should be placed intuitively for you. At the end of the execution, a text file containing
a list of various runtimes that were of interest to me - you can sift through those if you're
interested. Lastly, my hyperparameter grids are quite large. Running n_jobs=-1 on an i7-13700k
took just over 6 hours with this configuration of 50,000 grid search iterations for each dataset
and each model (500,000 total grid search iterations). This is still only a fraction of the 
grids I've setup. It is up to you how you choose to search. GridSearchCV is imported. You're
more than welcome to swap RandomizedSearchCV with GridSearchCV if you would like to run
entrypoint.py for a billion years. Let me know what your results are.