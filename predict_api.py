import pandas as pd
import numpy as np
from rule_base_api import aggresive_sound_detected
import tensorflow.lite as tflite
import librosa



yamnet_model_handle = 'models/lite-model_yamnet_tflite_1.tflite'
MODEL="models/yamnet_model_2.tflite"
classes = ["CH","GR","L-S1","L-S2"]
valid_classes = ["Animal","Domestic animals, pets", "Dog","Crying, sobbing","Whimper","Bark","Bow-wow","Growling","Whimper (dog)", "Livestock, farm animals, working animals","Groan","Grunt"]
WHINNING_INDEX=0
GROWLING_INDEX=1
NOR_BARK_INDEX=2
AGG_BARK_INDEX=3



preffered_aggression_index = -1 # refer final_aggression_detection() for more info


yamnet_model = tflite.Interpreter(model_path=yamnet_model_handle)
yamnet_model.allocate_tensors()



my_model = tflite.Interpreter(model_path=MODEL)
my_model.allocate_tensors()


# Get input and output tensors
yamnet_input_details = yamnet_model.get_input_details()
yamnet_output_details = yamnet_model.get_output_details()

my_model_input_details = my_model.get_input_details()
my_model_output_details = my_model.get_output_details()

class_map_path = pd.read_csv("models/yamnet_class_map.csv")
class_names = class_map_path['display_name'].tolist()


# get wav files and predict using model
def predict(file_path, chunk_size=5,USE_RULE_BASE=1):

    print("Predicting file: {}".format(file_path))

    # object to return
    results = []

    chartData = {
        "noOfAggressiveChunks": 0,
        "noOfNonAggressiveChunks": 0,
        "noOfGrowlingChunks": 0,
        "noOfWhinningChunks": 0,
        "noOfOther": 0,
    }

    try:

        # load wav file
        wav_data, sr = librosa.load(file_path, sr=16000, mono=True)

        chunk_size = 16000 * chunk_size
        num_chunks = wav_data.shape[0] // chunk_size

        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, wav_data.shape[0])
            chunk = wav_data[start:end]

            result = {
                "start_time": start,
                "end_time": end,
                "reliable": False,
                "main_sound": "",
                "main_sound_score": 0,
                "predicted_sound": "",
                "predicted_sound_score": 0,
                "is_aggressive": False,
                "aggression_index": -1
            }

            # Run YAMNet model
            yamnet_model.resize_tensor_input(yamnet_input_details[0]['index'], chunk.shape)
            yamnet_model.allocate_tensors()
            yamnet_model.set_tensor(yamnet_input_details[0]['index'], chunk)
            yamnet_model.invoke()
            scores = yamnet_model.get_tensor(yamnet_output_details[0]['index'])
            class_scores = np.mean(scores, axis=0)
            top_class = np.argmax(class_scores)
            inferred_class = class_names[top_class]
            top_score = class_scores[top_class]


            print(f'[YAMNet] The main sound is: {inferred_class} ({top_score})')
            result["main_sound"] = inferred_class
            result["main_sound_score"] = top_score.item()

            if inferred_class in valid_classes:

                my_model.resize_tensor_input(my_model_input_details[0]['index'], chunk.shape)
                my_model.allocate_tensors()
                my_model.set_tensor(my_model_input_details[0]['index'], chunk)
                my_model.invoke()
                my_model_results = my_model.get_tensor(my_model_output_details[0]['index'])
                top_class = np.argmax(my_model_results)
                inferred_class = classes[top_class]
                class_probabilities = np.squeeze(my_model_results)
                top_score = class_probabilities[top_class]

                if top_score:

                    result["reliable"] = True

                    print(f'Predicted sound is: {inferred_class} ({top_score})')
                    result["predicted_sound"] = inferred_class
                    result["predicted_sound_score"] = top_score.item()

                    if inferred_class == "CH":
                        chartData["noOfWhinningChunks"] += 1
                    elif inferred_class == "GR":
                        chartData["noOfGrowlingChunks"] += 1
                    else:
                        chartData["noOfOther"] += 1

                    # get predicted index
                    predicted_index = classes.index(inferred_class)

                    if predicted_index == NOR_BARK_INDEX or predicted_index == AGG_BARK_INDEX:

                        # if using the rule base then get the result
                        if USE_RULE_BASE == 2:

                            rule_base_result = aggresive_sound_detected(file_path)

                            print("Rule base result: Aggr : ", rule_base_result)

                            # get final prediction
                            is_aggressive, index = final_aggression_detection(rule_base_result, predicted_index)

                            if is_aggressive:
                                chartData["noOfAggressiveChunks"] += 1
                            else:
                                chartData["noOfNonAggressiveChunks"] += 1

                            result["aggression_index"] = index
                            
                            # final prediction
                            if is_aggressive and index > preffered_aggression_index:
                                print("\/\/\/\/\ Aggressive bark detected, index : ", index)
                                result["is_aggressive"] = True
                                result["predicted_sound"] = classes[AGG_BARK_INDEX]
                            else:
                                print("--------- Aggressive bark not detected, index : ", index)
                                result["is_aggressive"] = False
                                result["predicted_sound"] = classes[NOR_BARK_INDEX]
                        else:

                            if inferred_class == "L-S1":
                                chartData["noOfNonAggressiveChunks"] += 1
                            elif inferred_class == "L-S2":
                                chartData["noOfAggressiveChunks"] += 1

                            if predicted_index == AGG_BARK_INDEX:
                                print("\/\/\/\/\ Aggressive bark detected")
                                result["is_aggressive"] = True
                            elif predicted_index == NOR_BARK_INDEX:
                                print("--------- Aggressive bark not detected")
                                result["is_aggressive"] = False

                else:
                    print("Prediction is not reliable")
                    result["reliable"] = False
            
            else:
                chartData["noOfOther"] += 1
            
            results.append(result)
            
        
        return results , chartData

    
    except Exception as e:
        print("Error: ", e)

        return results , chartData

def final_aggression_detection(ruleBase, mlClass):

    # if LS1 and RB both true then aggressive bark detected
    # if LS1 is only true then aggressive bark not detected
    # if LS2 and RB both true then aggressive bark detected
    # if LS2 is only true then aggressive bark detected

    if mlClass==NOR_BARK_INDEX and ruleBase:
        return True, 1
    elif mlClass==NOR_BARK_INDEX and not ruleBase:
        return False, 0
    elif mlClass==AGG_BARK_INDEX and ruleBase:
        return True, 3
    elif mlClass==AGG_BARK_INDEX and not ruleBase:
        return True, 2
    else:
        return False, -1



if __name__ == "__main__":
    # for i, (dirpath, dirnames, filenames) in enumerate(os.walk("downloads\chunks")):
    #     for f in filenames:
    #         file_path = os.path.join(dirpath, f)
    #         predict(file_path)

    predict("downloads\event.wav")