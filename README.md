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

1. [Dependencies, pre-requisites,
and setting up the project](#dependencies-pre-requisites-and-setting-up-the-project)

## Dependencies, pre-requisites, and setting up the project

To start using our project, you have to: 

1. Use Python 3.6. The code in this repository is tested and works with Python 3.6. 
Probably using other Python 3.X versions will be OK, but please have in mind that this code
is for Python 3.6. 

2. Set-p the dependencies using either the ``pip`` ([pip_requirements.txt](pip_requirements.txt))
or ``conda`` ([conda_requirements.txt](conda_requirements.txt)) files. 

3. Download the audio data. You can download the three audio datasets from: 

   1. TUT-SED Synthetic 2016 dataset is available
   from [here](http://www.cs.tut.fi/sgn/arg/taslp2017-crnn-sed/index).
   
      Download the audio files (i.e. the Audio 1/5, Audio 2/5, ..., Audio 5/5)
      and place them at the ### directory. 
      
   2. The TUT Sound Events 2016 is available from [here](https://zenodo.org/record/45759#.XRYTYHUzZGo).
   
      Download the audio files and place them in the ### directory. 
      
   3. The 

Our code is tested and works for Python 3.6. For the deep neural networks, we use the
[PyTorch framework](https://pytorch.org/) and the version 1.1.0. 

To set-up and use our project, you have to install  
