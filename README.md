# opticalc library
This library is intended to assist in transfer matrix method and effective medium theory calculations.

This repository contains the current working version of opticalc library and some example jupyter notebooks.

## Install notes
It is recommended that the repository be cloned, and a new python virtual enviroment be set up to install the prequisite packages.  A new jupyter kernel can then be created from that virtual environment.

```
mkdir opticalc_repo
git clone https://github.com/iandvaag/opticalc
cd opticalc/
python3 -m venv github_opticalc_myenv
source github_opticalc_myenv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=github_opticalc_myenv
jupyter notebook
```
Note that you sould select the kernel 'github_opticalc_myenv' as a kernel befor running the example notebooks.


## How to use the library
Documentation for use is viewable at: [https://iandvaag.github.io/opticalc/](https://iandvaag.github.io/opticalc/)
