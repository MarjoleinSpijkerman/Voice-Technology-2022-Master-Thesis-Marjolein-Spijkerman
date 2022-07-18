# Voice Technology 2021/2022: Master Thesis by Marjolein Spijkerman

This GitHub page contains an overview of all the code that I used for the experiments. 

## The source of the base code and the data set

As base for the experiments, I used already existing for the MaskCycleGAN-VC. This can be found on [https://github.com/GANtastic3/MaskCycleGAN-VC](https://github.com/GANtastic3/MaskCycleGAN-VC).

The data that I used can be found on [http://www.isle.illinois.edu/sst/data/UASpeech/](http://www.isle.illinois.edu/sst/data/UASpeech/). To access this data you first need to ask for permission, this is why I cannot share the data on this GitHub page. 

## Changes I made to the code

First of all, no changes were made to model.py, train.py and test.py. <br/>

### Preprocessing function
The changes that were necessary for the experiments to work were made in the preprocessing function. 
The version of the preprocessing function that was used for the experiments is stored on this page as "preprocess_vcc2018_v2.py"<br/>

In this function three things were added:
1. An argument was added to the argument parser such that the speed rate for time stretching could be indicated. This was done in line 115.
2. The actual code needed for time-stretching. This part uses the code from [https://librosa.org/doc/main/generated/librosa.phase_vocoder.html](https://librosa.org/doc/main/generated/librosa.phase_vocoder.html). This was done in line 37-40.
3. As the preprocessing seems to shuffle the order of the wav files, I stored the order in which the wav files occured in a txt file. This txt file will then later be used to see whether the ASR made the correct predictions. This is done in line 96-100.

### Make predictions function
To test the performance on the ASR model, an additional python function was used called "make_predictions.py". In order to be able to run this function, some of the paths in the code may need to be changed to match the correct folders when running this code somewhere else. 

### Jobscript files
Will continue after dinner


