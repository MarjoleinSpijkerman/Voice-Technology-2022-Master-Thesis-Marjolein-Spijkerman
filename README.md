# Voice Technology 2021/2022: Master Thesis by Marjolein Spijkerman (Work in Progress)

This GitHub page contains an overview of all the code that I used for the experiments. <br/>
All the python files and jobscripts were run in the Peregrine cluster of the University of Groningen. <br/>
The shorter code bits in the jupyter notebook that were not computationally complex were run on my own laptop. 

## The source of the base code and the data set

As base for the experiments, I used already existing for the MaskCycleGAN-VC. This can be found on [https://github.com/GANtastic3/MaskCycleGAN-VC](https://github.com/GANtastic3/MaskCycleGAN-VC). To use this MaskCycleGAN-VC, I cloned the GitHub page to Peregrine and setup the conda environment in the same manner that they described on their page. To run the python commands that they describe on their page I used jobscript to make them work within Peregrine.

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

### Using the ASR for predictions and calculating the WER and CER. 
To test the performance on the ASR model, an additional python function was used called "make_predictions.py". In order to be able to run this function, some of the paths in the code may need to be changed to match the correct folders when running this code somewhere else. <br/>

Given a folder that contains separate folders each containing the audio files of one of the experimental settings, this code creates txt files containing the textual predictions for each audio file. Each experimental setting gets two txt files containing predictions, one for the original speech and one for the converted speech. Each audio files prediction gets its own line in these txt files. </br>

The folder structure looks like:
- Speaker M0X</br>
  - experiment 1</br>
    - all the audio files</br>
  - experiment 2</br>
    - all the audio files</br>
  - predictions</br>
    - all the txt files containing the predictions</br>

After creating all the text files containing the predictions, I calculated the Word Error Rate and Character Error Rate in a Jupyter Notebook on my own laptop. I added this notebook, which includes the output from running the cells, to this GitHub page. When running the notebook, the paths still need to be changed to match the map structure of the person running the code. The text files containing the transcriptions and the predictions of the experimental setups for both speakers and the transcriptions and predictions for the control speaker are also included in the files on GitHub. <br/>
The transcriptions are created by taking the txt files I created during the preprocessing and comparing the filenames to the speaker word list of the data set. This allowed me to create a list of transcriptions in the order of the audio files. 

### Jobscript files
Lastly, to run the code I used jobscripts. I added the jobscripts I used for preprocessing, the training, and the testing part of the MaskCycleGAN-VC. The preprocessing includes the part where you need to indicate the number of different time stretching experiments. The training part is mostly the same as it was in the original version of the MaskCycleGAN-VC. The testing part creates the audio files, here I needed to add all the different experiment's data. The preprocessing and testing jobscript show the jobscript for speaker M05, the jobscript for speaker M07 looks basically the same but has every instance of M05 replaced by M07. 
