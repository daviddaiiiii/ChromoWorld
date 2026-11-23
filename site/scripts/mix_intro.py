"""Create synchronized narration, music, and WebVTT for the overview film."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile
import textwrap

import imageio_ffmpeg
import numpy as np
from kokoro_onnx import Kokoro
from scipy import ndimage
from scipy.io import wavfile
from scipy.signal import resample_poly

FPS = 24
RATE = 48_000
DURATION = 233
VOICE = "af_heart"
CUES = Path(__file__).with_name("intro_narration.json")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def ffmpeg(*args, input_bytes=None):
    return subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
                          input=input_bytes, stdout=subprocess.PIPE, check=True).stdout


def speak(model, text, seconds):
    phones = model.tokenizer.phonemize(text, lang="en-us")
    # NCI pronunciation and standard letter-by-letter reading of AKT.
    for source, replacement in (("capivasertib", "kəpˌiːvəˈsɚtɪb"), ("A K T", "ˈeɪ kˈeɪ tˈiː")):
        phones = phones.replace(model.tokenizer.phonemize(source, lang="en-us"), replacement)
    sound, native = model.create(phones, voice=VOICE, is_phonemes=True)
    ratio = len(sound)/native/seconds
    if ratio > 1:
        if ratio > 1.27:
            raise ValueError(f"Cue needs too much time compression ({ratio:.2f}): {text}")
        sound = np.frombuffer(ffmpeg(
            "-f", "f32le", "-ar", native, "-ac", 1, "-i", "pipe:0",
            "-af", f"atempo={ratio*1.025:.5f}", "-f", "f32le", "pipe:1",
            input_bytes=np.asarray(sound, dtype="<f4").tobytes(),
        ), dtype="<f4")
    divisor = math.gcd(native, RATE)
    samples = resample_poly(sound, RATE//divisor, native//divisor).astype(np.float32)
    # Lightly match loudness between takes.
    active = samples[np.abs(samples) > .01]
    if not len(active):
        raise ValueError(f"Silence for cue: {text}")
    rms = np.sqrt(np.mean(active**2))
    gain = np.clip(.12/rms, .68, 1.6)
    samples *= min(gain, .82/np.max(np.abs(samples)))
    edge = min(int(RATE*.008),len(samples)//2)
    samples[:edge] *= np.linspace(0,1,edge)
    samples[-edge:] *= np.linspace(1,0,edge)
    if len(samples)/RATE > seconds+.025:
        raise ValueError(f"Cue failed to fit the authored window: {text}")
    return samples


def cue_time(t):
    ms = round(t*1000)
    return f"{ms//3_600_000:02}:{ms//60_000%60:02}:{ms//1000%60:02}.{ms%1000:03}"


def narration(model, work):
    dialogue = np.zeros(DURATION*RATE, dtype=np.float32)
    captions = {"en": [], "zh": []}
    for i,(start, stop, en, zh, *spoken) in enumerate(json.loads(CUES.read_text()), 1):
        audio = speak(model, spoken[0] if spoken else en, stop-start)
        start_index = round(start*RATE)
        dialogue[start_index:start_index+len(audio)] = audio
        finish = min(stop,start+len(audio)/RATE+.16)
        for lang, text in (("en", en), ("zh", zh)):
            captions[lang].append({"text":text, "startMs":round(start*1000), "endMs":round(finish*1000),
                                   "timestampMs":None, "confidence":None})
        print(f"{i:02} {start:6.2f}–{finish:6.2f} ({len(audio)/RATE:.2f}s)", flush=True)
    for lang, cues in captions.items():
        blocks = []
        for cue in cues:
            times = cue_time(cue["startMs"]/1000)+" --> "+cue_time(cue["endMs"]/1000)+" line:92%,end size:91%"
            caption = textwrap.fill(cue["text"],width=55) if lang == "en" else cue["text"]
            blocks.append(times+"\n"+caption)
        (work/f"intro-{lang}.vtt").write_text("WEBVTT\n\n"+"\n\n".join(blocks)+"\n",encoding="utf-8")
    return dialogue


def music_bed(score, dialogue):
    source = ffmpeg("-i",score,"-t",DURATION,"-ac",2,"-ar",RATE,"-f","f32le","pipe:1")
    music = np.frombuffer(source,dtype="<f4").reshape(-1,2).copy()
    if len(music) != len(dialogue):
        raise ValueError("The score must cover the full film.")
    # Measure speech in 20 ms blocks and ease the bed across phrase boundaries.
    hop = RATE//50
    time = (np.arange(DURATION*50)+.5)/50
    active = (np.max(np.abs(dialogue.reshape(-1,hop)),axis=1) > .018).astype(np.float32)
    occupied = ndimage.maximum_filter1d(active,size=35)
    duck = ndimage.gaussian_filter1d(occupied,sigma=7)
    fade = np.clip((DURATION-time)/3.4,0,1)*np.clip(time/.7,0,1)
    lift = np.interp(time,[0.,1.1,60.,61.,202.,203.,222.,228.,229.,233.],[1.25,1.,1.,.93,1.,.84,.84,1.2,1.5,1.5])
    envelope = fade*lift*(1-.49*duck)
    # Keep the score in the background even at the orchestral swell.
    block_rms = np.sqrt(np.mean(music.reshape(-1,hop,2)**2,axis=(1,2)))
    level = ndimage.gaussian_filter1d(block_rms,sigma=100)
    gain = .040/np.maximum(level,.095)
    amp = np.interp(np.arange(len(music))/RATE,time,envelope*gain).astype(np.float32)
    music *= amp[:,None]
    return music


def master(work, score, dialogue):
    music = music_bed(score, dialogue)
    speech = work/"speech.wav"
    background = work/"music.wav"
    wavfile.write(speech,RATE,dialogue)
    wavfile.write(background,RATE,music)
    wav = work/"master.wav"
    ffmpeg("-i",speech,"-i",background,"-filter_complex",
           "[0:a]highpass=f=75,acompressor=threshold=0.12:ratio=1.8:attack=20:release=130,"
           "pan=stereo|c0=c0|c1=c0[vo];[vo][1:a]amix=inputs=2:normalize=0,"
           "loudnorm=I=-16:TP=-1.5:LRA=9[a]",
           "-map","[a]","-ar",RATE,"-c:a","pcm_s24le",wav)
    return wav


def encode_overview(video, credits, sound, output):
    ffmpeg("-i",video,"-i",credits,"-i",sound,
           "-filter_complex","[0:v]setpts=PTS-STARTPTS[film];"
           "[1:v]setpts=PTS-STARTPTS[credit];[film][credit]concat=n=2:v=1:a=0,"
           "pad=iw:ih+112:0:0:black[v]",
           "-map","[v]","-map","2:a","-t",DURATION,"-r",FPS,"-c:v","libx264","-preset","fast",
           "-crf",18,"-threads",4,"-pix_fmt","yuv420p","-c:a","aac","-b:a","160k","-movflags","+faststart",
           "-metadata","comment=Music: A Kind Of Hope by Scott Buckley (CC BY 4.0); excerpt and mix.",output)
    ffmpeg("-xerror","-i",output,"-f","null","-")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="228-second silent overview")
    parser.add_argument("credits", type=Path, help="Five-second Remotion credits clip")
    parser.add_argument("music", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("voices", type=Path)
    parser.add_argument("output", type=Path, help="Directory for intro.mp4 and both VTT files")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = Kokoro(str(args.model), str(args.voices))
    with tempfile.TemporaryDirectory(dir=args.output) as temporary:
        work = Path(temporary)
        sound = master(work,args.music,narration(model,work))
        output = work/"intro.mp4"
        encode_overview(args.video,args.credits,sound,output)
        for asset in ("intro.mp4","intro-en.vtt","intro-zh.vtt"):
            (work/asset).replace(args.output/asset)


if __name__ == "__main__":
    main()
