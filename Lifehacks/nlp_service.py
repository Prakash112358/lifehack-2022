from re import I
from typing import Iterable, List
from tilsdk.localization.types import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import parselmouth
from parselmouth.praat import call

import onnxruntime as ort

from scipy.stats.mstats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import statistics
import soundfile
import librosa
from librosa import display

import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
#!/usr/bin/env python
#final nlp function
#!pip install praat-parselmouth
#!pip install fuzzy-c-means

   
class NLPService:
	
    emotions = ['angry','sad','happy','neutral','fear'] #prediction index for emotions

	
    def measurePitch(self,wav_file, f0min, f0max, unit):
        sound = parselmouth.Sound(wav_file) # read the sound
        duration = call(sound, "Get total duration") # duration
        pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
        meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
        stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        return duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, apq5Shimmer, apq11Shimmer, ddaShimmer

    def measureFormants(self,sound, wav_file, f0min, f0max):
        sound = parselmouth.Sound(sound) # read the sound
        pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

        formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
        numPoints = call(pointProcess, "Get number of points")

        f1_list = []
        f2_list = []
        f3_list = []
        f4_list = []

        for point in range(0, numPoints):
            point += 1
            t = call(pointProcess, "Get time from index", point)
            f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
            f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
            f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
            f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)
            f4_list.append(f4)

        f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
        f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
        f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
        f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']

        f1_mean = statistics.mean(f1_list)
        f2_mean = statistics.mean(f2_list)
        f3_mean = statistics.mean(f3_list)
        f4_mean = statistics.mean(f4_list)

        f1_median = statistics.median(f1_list)
        f2_median = statistics.median(f2_list)
        f3_median = statistics.median(f3_list)
        f4_median = statistics.median(f4_list)

        f1_std = statistics.stdev(f1_list)
        f2_std = statistics.stdev(f2_list)
        f3_std = statistics.stdev(f3_list)
        f4_std = statistics.stdev(f4_list) 
            
        return f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median, f1_std, f2_std, f3_std, f4_std


    def getPraat_Preds(self,wav_file):
        sound = parselmouth.Sound(wav_file)
        (duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, apq5Shimmer, apq11Shimmer, ddaShimmer) = self.measurePitch(sound, 75, 300, "Hertz")
        (f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median, f1_std, f2_std, f3_std, f4_std) = self.measureFormants(sound, wav_file, 75, 300)
        zscore_f1_median = (f1_median - f1_mean) / f1_std
        zscore_f2_median = (f2_median - f2_mean) / f2_std
        zscore_f3_median = (f3_median - f3_mean) / f3_std
        zscore_f4_median = (f4_median - f4_mean) / f4_std
        pF = (zscore_f1_median + zscore_f2_median + zscore_f3_median + zscore_f4_median) / 4
        fdisp = (f4_median - f1_median) / 3
        avgFormant = (f1_median + f2_median + f3_median + f4_median) / 4
        mff = (f1_median * f2_median * f3_median * f4_median) ** 0.25
        fitch_vtl = ( (1 * (35000 / (4 * f1_median))) +
            (3 * (35000 / (4 * f2_median))) + 
            (5 * (35000 / (4 * f3_median))) + 
            (7 * (35000 / (4 * f4_median))) ) / 4
        xysum = (0.5 * f1_median) + (1.5 * f2_median) + (2.5 * f3_median) + (3.5 * f4_median)
        xsquaredsum = (0.5 ** 2) + (1.5 ** 2) + (2.5 ** 2) + (3.5 ** 2)
        delta_f = xysum / xsquaredsum
        vtl_delta_f = 35000 / (2 * delta_f)
        sc = StandardScaler()
        sc = pickle.load(open("sc.pkl", "rb"))
        sc_means = pickle.load(open("sc_means.pkl", "rb"))
        sc_std = pickle.load(open("sc_stds.pkl", "rb"))
        sc.mean_ = sc_means
        sc.std_  = sc_std
        pca = PCA(n_components=2)
        pca = pickle.load(open("pca.pkl", "rb"))
        measures = [localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter,
                    localShimmer, localdbShimmer, apq3Shimmer, apq5Shimmer, apq11Shimmer, ddaShimmer]
        msr = np.array(measures).transpose().reshape(1,-1)
        msr_norm = sc.transform(msr)
        results = pca.transform(msr)
        JitterPCA = results[0][0]
        ShimmerPCA = results[0][1]

        x_cmeans1 = [JitterPCA, ShimmerPCA]
        c_means_fitk1 = pickle.load(open("cmeans.pkl", "rb"))
        pred_cmeans1 = c_means_fitk1.predict(np.array(x_cmeans1))

        x_cmeans2 = [vtl_delta_f, hnr]
        c_means_fitk2 = pickle.load(open("cmeans2.pkl", "rb"))
        pred_cmeans2 = c_means_fitk2.predict(np.array(x_cmeans2))

        x_cmeans3 = [meanF0, fitch_vtl]
        c_means_fitk3 = pickle.load(open("cmeans3.pkl", "rb"))
        pred_cmeans3 = c_means_fitk3.predict(np.array(x_cmeans3))

        x_cmeans4 = [f1_mean, localShimmer]
        c_means_fitk4 = pickle.load(open("cmeans4.pkl", "rb"))
        pred_cmeans4 = c_means_fitk4.predict(np.array(x_cmeans4))

        x_cmeans5 = [f2_mean, f2_median]
        c_means_fitk5 = pickle.load(open("cmeans5.pkl", "rb"))
        pred_cmeans5 = c_means_fitk5.predict(np.array(x_cmeans5))

        x_cmeans6 = [f3_mean, f3_median]
        c_means_fitk6 = pickle.load(open("cmeans6.pkl", "rb"))
        pred_cmeans6 = c_means_fitk6.predict(np.array(x_cmeans6))


        x_kmeans_dnn = [meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, 
                localShimmer, localdbShimmer,apq3Shimmer,apq5Shimmer, apq11Shimmer, ddaShimmer,
                f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median, 
                JitterPCA, ShimmerPCA, pF, fdisp, avgFormant, mff, fitch_vtl, delta_f, vtl_delta_f]    
        x2 = np.array(x_kmeans_dnn).transpose().reshape(1,-1)
        k_means_fitk = pickle.load(open("kmeans.pkl", "rb"))
        pred_kmeans = k_means_fitk.predict(x2)

        #dnn_model = tf.keras.models.load_model('dnn.h5')
        #predictions = dnn_model.predict(x2)
        #pred_dnn = np.argmax(predictions, axis=1)
        
        sess = ort.InferenceSession("model_dnn.onnx", providers= ['CPUExecutionProvider'])
        predictions = sess.run(["dense_7"], {"input": x2})
        pred_dnn = np.argmax(predictions[0], axis=1)    
        

        return pred_cmeans1[0], pred_cmeans2[0], pred_cmeans3[0], pred_cmeans4[0], pred_cmeans5[0],  pred_cmeans6[0], pred_kmeans[0], pred_dnn[0]  

    def decode_audio(self,audio_binary):
        audio, _ = tf.audio.decode_wav(contents=audio_binary)
        return tf.squeeze(audio, axis=-1)

    def get_spectrogram(self,wav_file):
        input_len = 16000
        audio_binary = tf.io.read_file(wav_file)
        waveform = self.decode_audio(audio_binary)
        waveform = waveform[:input_len]
        zero_padding = tf.zeros(
            [16000] - tf.shape(waveform),
            dtype=tf.float32)
        waveform = tf.cast(waveform, dtype=tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)
        spectrogram = tf.signal.stft(
            equal_length, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    def getSpec_CNN_Pred(self,wav_file):
        spectrogram = self.get_spectrogram(wav_file)
        spectrogram = np.expand_dims(spectrogram, axis=0) 
        sess = ort.InferenceSession("model_cnn.onnx", providers= ['CPUExecutionProvider'])
        predictions = sess.run(["dense_1"], {"x": np.array(spectrogram)})
        pred_cnn = np.argmax(predictions[0], axis=1)
        return pred_cnn[0]
    

    def getSpec_RNN_Pred(self,wav_file):
        y2,sr2 = librosa.load(wav_file,duration=1)
        ps2 = librosa.feature.melspectrogram(y=y2,sr=sr2)
        ps2=librosa.util.fix_length(ps2,44)
        ps2 = np.expand_dims(ps2, axis=0)
        sess = ort.InferenceSession("model_rnn.onnx", providers= ['CPUExecutionProvider'])
        predictions = sess.run(["dense_4"], {"lstm_input": ps2})
        pred_rnn = np.argmax(predictions[0], axis=1)    
        return pred_rnn[0]        
    

    def getTpg_CNN_Pred(self,wav_file):
        img_height = 160
        img_width = 160
        x, sr = librosa.load(wav_file)
        hop_length = 200 # samples per frame
        onset_env = librosa.onset.onset_strength(x, sr=sr, hop_length=hop_length, n_fft=2048)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=400)
        plt.rcParams["figure.figsize"] = (10,10)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo')
        plt.savefig("out.jpg")
        plt.clf()
        plt.close()
        img = image.load_img("out.jpg", target_size=(img_height, img_width))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)      
        sess = ort.InferenceSession("model_tpg.onnx", providers= ['CPUExecutionProvider'])
        predictions = sess.run(["dense_1"], {"rescaling_input": x})
        pred_tpg = np.argmax(predictions[0], axis=1)    
        return pred_tpg[0]
        



    def getFinalNlpPred(self, wav_file):  
        data, samplerate = soundfile.read(wav_file)
        soundfile.write('./sample.wav', data, samplerate, subtype='PCM_16')
        wav_file = './sample.wav'
        (pred_cmeans1, pred_cmeans2, pred_cmeans3, pred_cmeans4, pred_cmeans5,  pred_cmeans6, pred_kmeans, pred_dnn) = self.getPraat_Preds(wav_file)
        pred_cnn = self.getSpec_CNN_Pred(wav_file)
        pred_rnn = self.getSpec_RNN_Pred(wav_file)
        pred_tpg = self.getTpg_CNN_Pred(wav_file)
        one_shot_x = np.zeros(55)
        one_shot_x[pred_kmeans] = 1 #kmeans (0-4)
        one_shot_x[pred_dnn+5] = 1 #dnn (5-9)
        one_shot_x[pred_cnn+10] = 1 #cnn (10-14)
        one_shot_x[pred_rnn+15] = 1 #rnn (15-19)
        one_shot_x[pred_tpg+20] = 1 #tpg (20-24)
        one_shot_x[pred_cmeans1+25] = 1 #c-means (25-29)
        one_shot_x[pred_cmeans2+30] = 1 #c-means (30-34)
        one_shot_x[pred_cmeans3+35] = 1 #c-means (35-39)
        one_shot_x[pred_cmeans4+40] = 1 #c-means (40-44)
        one_shot_x[pred_cmeans5+45] = 1 #c-means (45-49)
        one_shot_x[pred_cmeans6+50] = 1 #c-means (50-54)
        features = np.expand_dims(one_shot_x, axis=0)
        #final_model = tf.keras.models.load_model('combine7.h5')
        #predictions = final_model.predict(features)
        #pred_final = np.argmax(predictions,axis=1)
        
        sess = ort.InferenceSession("model_final.onnx", providers= ['CPUExecutionProvider'])
        predictions = sess.run(["dense_27"], {"input": features})
        pred_final = np.argmax(predictions[0], axis=1)            
        
        
        return pred_final[0]
    
    def __init__(self, model_dir:str):
        pass

    def locations_from_clues(self, clues:Iterable[Clue]) -> List[RealLocation]:
        emotions = ['angry','sad','happy','neutral','fear'] 
        locations = []
        predictions = []
        notimptlocations = []
        for i in clues:
            with open('clues.wav', 'wb') as f:
                f.write(i.audio)
            prediction = self.getFinalNlpPred('./clues.wav')
            print("==================================")
            print("emotion=",emotions[prediction])
            print("==================================")
            predictions.append(emotions[prediction])
            if prediction < 2:
                locations.append(RealLocation(i.location[0],i.location[1]))
            else:
                notimptlocations.append(Reallocation(i.location[0],i.location[1]))
        return predictions, locations, notimptlocations
        '''
        Process clues and get locations of interest.

        Parameters
        ----------
        clues
            Clues to process.

        Returns
        -------
        lois
            Locations of interest.
        '''

        # TODO: Participant to complete.
        #pass
