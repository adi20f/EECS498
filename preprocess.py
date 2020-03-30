import torchaudio
import torchvision
import torch
import os

def get_ids():
    # Load sample ids
    ids = []
    with open("identifiers.txt", "r") as f:
        ids = f.read().splitlines()

    return ids

def load_audio(ids, directory="data/s1audio", max_length=280):
    print(len(ids), "files")
    audio_features = []

    # Extract MFCC features for each sample
    for filename in ids:
        filename = directory + "/" + filename + ".wav"
        audio, sample_rate = torchaudio.load(filename)
        m = torchaudio.transforms.MFCC(sample_rate)
        feat_vec = m(audio)
        feat_vec = torch.nn.functional.pad(feat_vec, (0, max_length-feat_vec.shape[2])).squeeze()
        print(feat_vec.shape)
        audio_features.append(feat_vec)


    audio_features = torch.stack(audio_features)

    return audio_features

def load_video(ids, directory="data/s1video", max_length=1000):

    video_features = []

    for idx, filename in enumerate(ids):
        if idx % 100 == 0:
            print("finsihed",idx)

        filename = directory + "/" + filename + ".mpg"
        # print(filename)
        try:
            vframes,_,_ = torchvision.io.read_video(filename, pts_unit="sec")
            video_features.append(vframes)
        except:
            print("oops")
            continue

    video_features = torch.stack(video_features)
    print(video_features.shape)

    return video_features

if __name__ == "__main__":
    ids = get_ids()
    # load_audio(ids",data/s1audio")
    load_video(ids, "data/s1video")
