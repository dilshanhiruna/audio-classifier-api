import numpy as np
import scipy
import librosa
import librosa.display

def aggresive_sound_detected(filename):
    
    # Load audio file in 16-bit mono
    y, sr = librosa.load(filename, sr=16000, mono=True)

    # preprocess the audio data
    y = librosa.effects.trim(y, top_db=20)[0]

    # normalize the audio data 
    y = librosa.util.normalize(y)

    # remove unwanted frequencies using scipy
    y = scipy.signal.medfilt(y, kernel_size=5)
    

    # Define threshold for bark detection
    energy_threshold = np.median(y) + (np.std(y) * 30)
    
    # Find bark events using the energy threshold
    bark_events = librosa.effects.split(y, top_db=energy_threshold)


    # Compute the time intervals between barks
    intervals = []
    for i in range(1, len(bark_events)):
        start_time = bark_events[i-1][1] / sr
        end_time = bark_events[i][0] / sr
        interval = end_time - start_time
        intervals.append(interval)

    if(len(intervals)>0 and len(bark_events)>0):
        # calculate the average interval between barks
        avg_interval = sum(intervals) / len(intervals)
        # print("Average interval between barks:", avg_interval)

        # Identify the bark is aggressive or not from number of barks, interval between barks
        if len(bark_events) > 5 and avg_interval < 0.6: 
            # print("Aggressive")
            return True
        else:
            # print("Not Aggressive")
            return False
    else:
        # print("Not Aggressive")
        return False
        
