import re
import json
from faster_whisper import WhisperModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transcribe", tags=["STT"])
async def create_upload_file(file: UploadFile = File(...)):
    with open("uploaded_audio.mp3", "wb") as audio_file:
        audio_file.write(await file.read())
    try:
        answer = getSpeechToKlima("uploaded_audio.mp3")
        print(answer)
        return JSONResponse(content=answer)
    except:
        return JSONResponse(content="None")


def executeWhisper(audioFile):
    # Load model and audio
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audioFile, beam_size=5)

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" %
              (segment.start, segment.end, segment.text))
        result = segment.text

    # print the recognized text
    print(result.text)
    detected_text = result.text.lower()
    return detected_text


def GetRegexTime(text):
    hour_pattern = re.compile(r'(\w+)\s*stunde(n)?')
    minute_pattern = re.compile(r'(\w+)\s*minute(n)?')
    second_pattern = re.compile(r'(\w+)\s*sekunde(n)?')
    hour_match = hour_pattern.search(text)
    minute_match = minute_pattern.search(text)
    second_match = second_pattern.search(text)
    # Die Zahl aus dem Regex herausziehen Gruppe 1 = (|d+) Gruppe 2 = n
    hours = hour_match.group(1) if hour_match else 0
    minutes = minute_match.group(1) if minute_match else 0
    seconds = second_match.group(1) if second_match else 0

    return hours, minutes, seconds


def GetRegexIntensity(text):
    intensity_pattern = re.compile(r'(\d+)\s*(prozent?|%)')
    intensity_match = intensity_pattern.search(text)
    intensity = int(intensity_match.group(1)) if intensity_match else None
    return intensity


def GetRegexPlace(text):
    place_pattern = re.compile(r'(\bin\b|\bvon\b|\baus\b)\s*(\w+)')
    place_match = place_pattern.search(text)
    place = place_match.group(2) if place_match else ""
    return place


def GetCommandType(text, keywords):
    ContainKeyword = False
    if "in" in text:
        for keyword in keywords:
            if keyword in text:
                ContainKeyword = True
        commandType = 0 if ContainKeyword else 2
    else:
        commandType = 1
    return commandType


def GetFeature(text, keywords):
    features = []
    for keyword in keywords:
        if keyword in text:
            features.append(keyword)
    return features


def TimeToNumerical3(time):
    try:
        time = int(time)
        return time
    except ValueError:
        if time == "null":
            time = 0
        if time == "einer":
            time = 1
        if time == "zwei":
            time = 2
        if time == "drei":
            time = 3
        if time == "vier":
            time = 4
        if time == "fünf":
            time = 5
        if time == "sechs":
            time = 6
        if time == "sieben":
            time = 7
        if time == "acht":
            time = 8
        if time == "neun":
            time = 9
        if time == "zehn":
            time = 10
        if time == "elf":
            time = 11
        if time == "zwölf":
            time = 12
        if time == "dreizehn":
            time = 13
        if time == "vierzehn":
            time = 14
        if time == "fünfzehn":
            time = 15
        if time == "sechzehn":
            time = 16
        if time == "siebzehn":
            time = 17
        if time == "achtzehn":
            time = 18
        if time == "neunzehn":
            time = 19
        if time == "zwanzig":
            time = 20

        return time


def getSpeechToKlima(speechFile):
    detected_text = executeWhisper(speechFile)
    keywords = ["sonne", "regen", "temperatur", "luftfeuchtigkeit"]
    detected_numbers = re.findall(
        r'\d+', detected_text)  # optional (for testing)
    numbers = [int(num) for num in detected_numbers]  # optional (for tsting)
    print(f"numbers: {numbers}")
    # Type of the Command (Planning, Instant, Live)
    CommandType = GetCommandType(detected_text)
    print(f"Command Type: {CommandType}")
    # Feature (Sun, rain, temperature, humidity)
    Features = GetFeature(detected_text)
    print(f"Feature: {Features}")

    # Intensity of the Feature
    Intensity = GetRegexIntensity(detected_text)
    print(f"Intensity: {Intensity}")
    # When to plan
    hours, minutes, seconds = GetRegexTime(detected_text)
    hours = TimeToNumerical3(hours)
    minutes = TimeToNumerical3(minutes)
    seconds = TimeToNumerical3(seconds)
    print(f"hours: {hours}")
    print(f"minutes: {minutes}")
    print(f"seconds: {seconds}")
    relative_unixTime = (hours*60+minutes)*60+seconds
    print(f"unix time: {relative_unixTime}")

    # What place if Live
    Place = None
    if CommandType == 2:
        Place = GetRegexPlace(detected_text)
    print(f"Place: {Place}")

    # Write all data into a dictionary
    dictionary = {
        "command": CommandType,
        "feature": Features[0],
        "data": [
            {
                "value": Intensity,
                "Time": relative_unixTime
            },
            {
                "value": Intensity
            },
            {
                "place": Place
            }
        ]
    }
    # Write the dictionary in a json file
    json_object = json.dumps(dictionary, indent=0)

    return json_object
