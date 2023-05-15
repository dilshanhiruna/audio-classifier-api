import os
from fastapi import FastAPI, HTTPException, responses
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp
import soundfile as sf
import requests
from predict_api import predict

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

yt_url = 'https://www.youtube.com/watch?v='
yts_url = 'https://www.youtube.com/shorts/'

# uvicorn api:app --reload  
# Define the API endpoint
# endpoint call example: http://127.0.0.1:8000/classify_audio?url=https://www.youtube.com/watch?v=jYUtL-LcMsk&chunk_size=2
@app.get("/classify_audio")
async def classify_audio_from_url(url: str, chunk_size: int = 5,agg_level: int = 1):

    try:

        # check if the url is a youtube url
        if url.startswith(yt_url) or url.startswith(yts_url):

            if url.startswith(yts_url):
                url = url.replace(yts_url, yt_url)

            # get the id from the url
            video_id = url.split("=")[1]

            # check if the video exists
            r = requests.get("https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={}&format=json".format(video_id))
            if r.status_code != 200:
                raise HTTPException(status_code=404, detail="Video not found")


            # Use the youtube-dl library to download the audio file
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'outtmpl': './downloads/%(id)s.%(ext)s',

            }

            # check if the audio file already exists in the downloads folder with the same id
            if not (os.path.exists("./downloads/{}.wav".format(video_id))):

                # take the id 
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

            print(video_id)

            filename = "./downloads/{}.wav".format(video_id)

            # Classify the audio
            prediction, chartData = predict(filename, chunk_size, agg_level)

            # delete the file
            # os.remove(filename)

            return {"prediction": prediction, "chartData": chartData}

        # if the url is a direct link to the audio file (wav or mp3)
        elif url.endswith(".wav") or url.endswith(".mp3"):

            audio_filename = "dau"
            
            print("Predicting file: {}".format(url))
            # download the file in to downloads folder
            r = requests.get(url, allow_redirects=True)


            if url.endswith(".wav"):
                filename = "./downloads/{}".format(audio_filename + ".wav")
                with open(filename, 'wb') as f:
                    f.write(r.content)
            elif url.endswith(".mp3"): # convert mp3 to wav
                mp3filename = "./downloads/{}".format(audio_filename + ".mp3")
                with open(mp3filename, 'wb') as f:
                    f.write(r.content)
                
                # convert mp3 to wav
                data, samplerate = sf.read(mp3filename)
                sf.write("./downloads/{}".format(audio_filename + ".wav"), data, samplerate)
                filename = "./downloads/{}".format(audio_filename + ".wav")

                # delete the mp3 file
                os.remove(mp3filename)
                    
            # Classify the audio
            prediction, chartData = predict(filename, chunk_size, agg_level)

            # delete the file
            os.remove(filename)

            return {"prediction": prediction, "chartData": chartData}
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error")



@app.get("/convert_yt_to_audio")
async def convert_yt_to_audio(url: str):

    try:

        # first clear the folder
        for filename in os.listdir("./downloads"):
            os.remove("./downloads/{}".format(filename))

        if url.startswith(yts_url):
            url = url.replace(yts_url, yt_url)
        # get the id from the url
        video_id = url.split("=")[1]
        # check if the video exists
        r = requests.get("https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={}&format=json".format(video_id))
        if r.status_code != 200:
            return


        # Use the youtube-dl library to download the audio file
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': './downloads/%(id)s.%(ext)s',
            }

        # take the id 
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # get the id from the url
        video_id = url.split("=")[1]

        # convert mp3 to wav
        data, samplerate = sf.read("./downloads/{}.wav".format(video_id))
        sf.write("./downloads/{}.mp3".format(video_id), data, samplerate)

        # delete the wav file
        # os.remove("./downloads/{}.wav".format(video_id))

        file_path = "./downloads/{}.mp3".format(video_id)

        return responses.FileResponse(file_path)
    

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error")


# health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}

    