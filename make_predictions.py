import os
import torchaudio
from datasets import load_dataset, load_metric
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
import torch

#Insert the name of the folder where the data can be found
main_folder = "results/experiments_M5"
#This is the ASR model that we are using
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

#It is preferable to run it on a gpu.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#I'm only evaluating on text data
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
processor = Wav2Vec2Processor.from_pretrained(model_name)

#Looping over all time-stretching settings
for setting in os.listdir(main_folder):
	#since we're saving the predictions in the same folder, we need to make sure we skip this data
    if setting != 'predictions':
		
		#save path name, we need this later
        path = os.path.join(main_folder, setting, "converted_audio/")
        original_audio_predictions = []
        transformed_audio_predictions = []
        
        new_name = setting[23:]
        new_name = new_name[:-5]
        
        #The folder contains both the original and converted speech data. So, we need to loop over half the length of items in the folder
        for i in range(0, int(len(os.listdir(path))/2)):
            if (i%100 == 0):
                print(i)
				
			#store path location of both the original and converted file
            file_name = []
            name = path + str(i) + "-converted_" + new_name + "_to_CM01.wav"
            file_name.append(name)
            name = path + str(i) + "-original_" + new_name + "_to_CM01.wav"
            file_name.append(name)
            
			#A separate prediction for both is needed
            for i in range(2):
                speech, _ = torchaudio.load(file_name[i])
                resampler = torchaudio.transforms.Resample(orig_freq=16_000, new_freq=16_000)   
                speech = resampler.forward(speech.squeeze(0))
                features = processor(speech, sampling_rate=16000, padding=True, return_tensors="pt")
                input_values = features.input_values.to(device)
                attention_mask = features.attention_mask.to(device)
        
                with torch.no_grad():
                    logits = model(input_values, attention_mask=attention_mask).logits
                    pred_ids = torch.argmax(logits, dim=-1)
                    
                    if i == 0:
                        transformed_audio_predictions.append(processor.batch_decode(pred_ids))
                    else:
                        original_audio_predictions.append(processor.batch_decode(pred_ids))
        
		#This needs to be manually changed for the experimental setting, in this case it's for speaker M5.
		#Change it to the folder where the data is located
        path_name = 'results/experiments_M5/predictions'
        original_name = new_name + '_predictions_original.txt'
        transformed_name = new_name + '_predictions_transformed.txt'
        
		#Save the prediction in txt files. Each prediction has it's own line. 
        with open(os.path.join(path_name, original_name), 'w') as output:
            for prediction in original_audio_predictions:
                output.write(prediction[0])
                output.write("\n")
    
        with open(os.path.join(path_name, transformed_name), 'w') as output:
            for prediction in transformed_audio_predictions:
                output.write(prediction[0])
                output.write("\n")