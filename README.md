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

Our paper is presented to the 
[Detection and Classification of Acoustic Scenes and
Events (DCASE) Workshop 2019](http://dcase.community/workshop2019/index). 
You can find an online version of our paper at arXiv: [https://arxiv.org/abs/1907.08506](https://arxiv.org/abs/1907.08506)

**If you use our method, please cite our paper.**  

**You can get the version of the code used in the paper from** [![DOI](https://zenodo.org/badge/194117423.svg)](https://zenodo.org/badge/latestdoi/194117423)

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

Sound event detection (SED) is the task of identifying activities of sound events from
short time representations of audio. For example, given an audio feature vector that
is extracted from 0.04 seconds, a SED method should identify the activities of different
sound events in this vector. Usually, SED is applied over a sequence of short time audio 
feature vectors and the identification of activities of sound events is performed for 
every input feature vector. That is, as an input is given a matrix
![equation](https://latex.codecogs.com/gif.latex?\inline&space;\mathbf{X}\in\mathbb{R}^{T\times&space;F}),
with `T` and `F` to be the amount of feature vectors and features, respectively, the output is the matrix
![equation](https://latex.codecogs.com/gif.latex?\inline&space;\hat{\mathbf{Y}}\in\mathbb{R}^{T\times&space;C}),
which holds the predictions for each of the `C` classes at every `t` feature vector.

In real-life recordings, the various sound events likely have **temporal structures within and across events**. 
For instance, a “footsteps” event might be repeated with pauses in between (intra-event structure). On the
other hand, “car horn” is likely to follow or precede the “car passing by” sound event (inter-events structure). 
Such temporal structures are employed and used in other machine learning tasks, for example in machine 
translation, image captioning, and speech recognition. In these tasks, the developed method also learns
a model of the temporal associations of the targeted classes. These associations usually are termed
as language model. 

SED methods can benefit from a language model. The method in this repository is about exactly this. 
A method to take advantage of language model for SED. 
 
### Teacher forcing and scheduled sampling

In order to take advantage of the above mentioned temporal structures, we use the *teacher forcing* [1] technique.
Teacher forcing is the conditioning of the input to an RNN with the activities of sound events at the previous time
step. That is, 

![equation](https://latex.codecogs.com/gif.latex?h'_{t}=RNN(h'_{t-1},h_{t},y'_{t-1}))

where ![equation](https://latex.codecogs.com/gif.latex?\inline&space;h'_{t}) is the output of the RNN at time-step *t*,
![equation](https://latex.codecogs.com/gif.latex?\inline&space;h_{t}) is the input to the RNN (from a previous layer)
and at time-step *t*, and 
![equation](https://latex.codecogs.com/gif.latex?\inline&space;y'_{t-1}) is the activities of the sound events at
the time-step *t-1*. 

If as ![equation](https://latex.codecogs.com/gif.latex?\inline&space;y'_{t-1}) are used the ground truth values, 
then the RNN will not be robust to cases where the
![equation](https://latex.codecogs.com/gif.latex?\inline&space;y'_{t-1}) is not a correct class activity. For example,
in the testing process where there are no ground truth values. 

If as ![equation](https://latex.codecogs.com/gif.latex?\inline&space;y'_{t-1}) are used the predictions of the
classifier, then the RNN will have a difficult time to learn any dependencies of the sound events, because during
training (and especially at the beginning of the training process) it will be fed incorrect class activities. 

To tackle both of the above, we employ the scheduled sampling technique [2]. That is, at the beginning of the training
we use as ![equation](https://latex.codecogs.com/gif.latex?\inline&space;y'_{t-1}) the ground truth values. As
the training proceeds and the classifier learns to predict more and more correct class activities, we gradually
employ the predictions of the classifier as ![equation](https://latex.codecogs.com/gif.latex?\inline&space;y'_{t-1}).

[1] R. J. Williams and D. Zipser, “A learning algorithm for continually running fully recurrent neural networks,”
Neural Computation, vol. 1, no. 2, pp. 270–280, June 1989.

[2] S. Bengio, O. Vinyals, N. Jaitly, and N. Shazeer, “Scheduled sampling for sequence prediction with recurrent
neural networks,” in Proceedings of the 28th International Conference on Neural Information Processing Systems, 
Volume 1, ser. NIPS’15. Cambridge, MA, USA:MIT Press, 2015, pp. 1171–1179. Online. 
Available: http://dl.acm.org/citation.cfm?id=2969239.2969370

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
   
      Download the audio files (i.e. the Audio 1/5, Audio 2/5, ..., Audio 5/5),
      do your feature extraction and follow the instructions at the
      [Data Set-up](#data-set-up) section. 
      
   2. The TUT Sound Events 2016 is available from [here](https://zenodo.org/record/45759#.XRYTYHUzZGo).
   
      Download the audio files, do your feature extraction and follow the instructions
      at the [Data Set-up](#data-set-up) section. 
      
   3. The TUT Sound Event 2017 is available from [here](https://zenodo.org/record/814831#.XTGbsnUzZGo)
   
      Download the audio files, do your feature extraction and follow the instructions
      at the [Data Set-up](#data-set-up) section.

4. Now the project is set-up and you can use it with the data that you got from step 3. 

## Using SEDLM
You can use SEDLM directly for your data, or you can check the code and adopt the SEDLM to your SED task, or repeat
the process described in our paper.

SEDLM code is based on [PyTorch](https://pytorch.org/), version 1.1.0.

In the current form, different variables of the code are specified in a YAML file, holding all the settings for the
code. All the YAML files are in the `settings` directory, and the YAML loading function searches in the `settings`
directory for YAML files. In general, you can just alter the values of settings in the YAML file and then run the
code.

### Data set-up

The data have to be in the `data` directory. 

If you want to use the existing data loaders, then you have to have your data organized in 
a specific way. First of all, you have to have different files for input features and target
values. For example, `input_features.npy` and `target_values.npy`. Then, depending on the 
dataset that you will use, you have to have your data in different directories. That is: 

  1. TUTSED Synthetic 2016.
   
      The data have to be in a directory called `synthetic`, in the `data` directory. That is, 
      `data/synthetic`. Then, the files for the training, validation, and testing data have to
      be in a different directory. That is:
       
        - `data/synthetic/training`   
        - `data/synthetic/validation`   
        - `data/synthetic/testing`   
      
      You have to have different **numpy files** for the input features and the target values.
      You can specify the name of each of the input or target files in the YAML settings file.
      For example, the training files should be like:
       
        - `data/synthetic/training/input_features.npy`
        - `data/synthetic/training/target_values.npy`
        
      The code will load the numpy files and use them for training the SEDLM method. You can to make 
      sure though that the input features and target values are properly ordered. That is, the 
      first element in the input features corresponds to the first element in the target values. 
       
  2. TUT Real Life 2016
   
      The data have to be in a directory called `real_life_2016`, in the `data` directory. That is, 
      `data/real_life_2016`. Then, the files for each of the folds have to be in a different 
      directory. That is:
       
        - `data/real_life_2016/fold_1`   
        - `data/real_life_2016/fold_2`   
        - `data/real_life_2016/fold_3`   
        - `data/real_life_2016/fold_4`   
      
      You have to have different **pickle files** for the input features and the target values, 
      and for the training and testing of each fold. Since there are multiple files per scene
      and per fold, you cannot have all features in a numpy array. Thus, you have to have all
      the data in a list and serialize (i.e. store to disk) that list using the pickle package.
      Also, there are files for training and testing in each fold.  
      
      For convenience, SEDLM uses automatically a pre-fix for the file names. That is, it 
      automatically adds "train" and "test" to the specified file name.  
      
      You can specify the name of each of the input or target files in the YAML settings file.
      For example, the files should be like:
       
        - `input_features.p`
        - `target_values.p`
        
      Then, SEDLM code will search for the proper files and for each fold. For example, for fold 1
      and home scene, the following files will be sought:
       
        - `data/real_life_2016/home/fold_1/train_input_features.p`
        - `data/real_life_2016/home/fold_1/train_target_values.p`
        - `data/real_life_2016/home/fold_1/test_input_features.p`
        - `data/real_life_2016/home/fold_1/test_target_values.p`
        
      The code will load the pickle files and use them for training the SEDLM method. You can to make 
      sure though that the input features and target values are properly ordered. That is, the 
      first element in the input features corresponds to the first element in the target values. 
      
  3. TUT Real Life 2017
   
      The data have to be in a directory called `real_life_2017`, in the `data` directory. That is, 
      `data/real_life_2017`. Then, the files for each of the folds have to be in a different 
      directory. That is:
       
        - `data/real_life_2017/fold_1`   
        - `data/real_life_2017/fold_2`   
        - `data/real_life_2017/fold_3`   
        - `data/real_life_2017/fold_4`   
      
      You have to have different **pickle files** for the input features and the target values, 
      and for the training and testing of each fold. Since there are multiple files per fold, 
      you cannot have all features in a numpy array. Thus, you have to have all
      the data in a list and serialize (i.e. store to disk) that list using the pickle package.
      Also, there are files for training and testing in each fold.  
    
      For convenience, SEDLM uses automatically a pre-fix for the file names. That is, it 
      automatically adds "train" and "test" to the specified file name.  
    
      You can specify the name of each of the input or target files in the YAML settings file.
      For example, the files should be like:
     
        - `input_features.p`
        - `target_values.p`
      
      Then, SEDLM code will search for the proper files and for each fold. For example, for fold 1,
      the following files will be sought:
     
        - `data/real_life_2017/fold_1/train_input_features.p`
        - `data/real_life_2017/fold_1/train_target_values.p`
        - `data/real_life_2017/fold_1/test_input_features.p`
        - `data/real_life_2017/fold_1/test_target_values.p`
      
      The code will load the pickle files and use them for training the SEDLM method. You can to make 
      sure though that the input features and target values are properly ordered. That is, the 
      first element in the input features corresponds to the first element in the target values.
       
### Hyper-parameters tuning

The hyper-parameters can be tuned from the YAML settings files. Available hyper-parameters for tuning are: 

  1. Amount of CNN channels
  2. Dropout for CNNs and RNN
  3. Scheduled sampling parameters
  4. Learning rate of Adam optimizer
  5. Batch size 

### Running the system

You can run the system using a bash script. An example of such script are the files: 

  1. `example_bash_script_baseline.sh`, which runs the baseline configuration for the SEDLM
  2. `example_bash_script_tf.sh`, which runs the SEDLM with the TUT Real Life 2017 dataset. 


## Acknowledgements

* Part of the computations leading to these results were performed on a TITAN-X GPU
donated by [NVIDIA](https://www.nvidia.com/en-us/) to K. Drossos. 
* The authors wish to acknowledge [CSC-IT Center for Science](https://www.csc.fi/), 
Finland, for computational resources. 
* The research leading to these results has received funding from the [European Research 
Council](https://erc.europa.eu/) under the European Union’s H2020 Framework Programme 
through ERC Grant Agreement 637422 EVERYSOUND. 
* P. Magron is supported by the [Academy of Finland](http://www.aka.fi/en), project no. 290190.
