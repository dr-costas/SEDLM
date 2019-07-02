# Language Modeling for Sound Event Detection with Teacher Forcing and Scheduled Sampling 
***
### Welcome to the repository of the SEDLM method. 

This is the repository for the method presented in the paper 
"Language Modeling for Sound Event Detection with Teacher Forcing and Scheduled Sampling", by 
[K. Drossos](https://tutcris.tut.fi/portal/en/persons/konstantinos-drosos(b1070370-5156-4280-b354-6291618bb965).html), 
S. Gharib,
[P, Magron](http://www.cs.tut.fi/~magron/), 
and 
[T. Virtanen](http://www.cs.tut.fi/~tuomasv/). 

Our paper is submitted to the 
[Detection and Classification of Acoustic Scenes and
Events (DCASE) Workshop 2019](http://dcase.community/workshop2019/index). 
You can find an online version of our paper at arXiv: 

**If you use our method, please cite our paper.**  

## Table of Contents

1. [Method introduction](#method-introduction)
    1. [Sound event detection](#sound-event-detection)
    2. [Teacher forcing and scheduled sampling](#teacher-forcing-and-scheduled-sampling)
2. [Dependencies, pre-requisites,
and setting up the project](#dependencies-pre-requisites-and-setting-up-the-project)
3. [Using SEDLM](#using-sedlm)
    1. [Data set-up](#data-set-up)
    2. [Hyper-parameters tuning](#hyper-parameters-tuning)
    3. [Running the system](#running-the-system)
4. [Acknowledgements](#acknowledgements)

***

## Method introduction

### Sound event detection

### Teacher forcing and scheduled sampling

## Dependencies, pre-requisites, and setting up the project

To start using our project, you have to: 

1. Use Python 3.6. The code in this repository is tested and works with Python 3.6. 
Probably using other Python 3.X versions will be OK, but please have in mind that this code
is for Python 3.6. 

2. Set-up the dependencies using either the ``pip`` ([pip_requirements.txt](requirements/pip_requirements.txt))
or ``conda`` ([conda_requirements.txt](requirements/conda_requirements.txt)) files. 
Navigate with your terminal inside the root directory of the project (i.e. the directory that is
created after cloning this repository) and then issue the proper command at the terminal: 

    1. **pip**: To set-up the dependencies with `pip` use: 
        ```bash
        $ pip install -r requirements/pip_requirements.txt
        ```
    2. **conda**: To set-up the dependencies with `conda`, you can issue the command
        ```bash
        $ conda install --yes --file requirements/conda_requirements.txt
        ```

3. Download the audio data. You can download the three audio datasets from: 

   1. TUT-SED Synthetic 2016 dataset is available
   from [here](http://www.cs.tut.fi/sgn/arg/taslp2017-crnn-sed/index).
   
      Download the audio files (i.e. the Audio 1/5, Audio 2/5, ..., Audio 5/5)
      and place them at the ### directory. 
      
   2. The TUT Sound Events 2016 is available from [here](https://zenodo.org/record/45759#.XRYTYHUzZGo).
   
      Download the audio files and place them in the ### directory. 
      
   3. The 

4. Now the project is set-up and you can use it with the data that you got from step 3. 

## Using SEDLM

### Data set-up

### Hyper-parameters tuning 

### Running the system


## Acknowledgements

* Part of the computations leading to these results were performed on a TITAN-X GPU
donated by [NVIDIA](https://www.nvidia.com/en-us/) to K. Drossos. 
* The authors wish to acknowledge [CSC-IT Center for Science](https://www.csc.fi/), 
Finland, for computational resources. 
* The research leading to these results has received funding from the [European Research 
Council](https://erc.europa.eu/) under the European Union’s H2020 Framework Programme 
through ERC Grant Agreement 637422 EVERYSOUND. 
* P. Magron is supported by the [Academy of Finland](http://www.aka.fi/en), project no. 290190.
