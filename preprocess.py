import torchaudio
import torchvision
import torch
import os
import imutils
import dlib
import cv2
import numpy as np
from imutils import face_utils


def get_ids():
# Load sample ids
    ids = []
    with open("identifiers.txt", "r") as f:
        ids = f.read().splitlines()

    return ids

def load_audio(ids, directory="data/s1audio/", max_length=280):
    print(len(ids), "files")
    audio_features = []

    # Extract MFCC features for each sample
    for filename in ids:
        filename = directory + filename + ".wav"
        audio, sample_rate = torchaudio.load(filename)
        m = torchaudio.transforms.MFCC(sample_rate)
        feat_vec = m(audio)
        feat_vec = torch.nn.functional.pad(feat_vec, (0, max_length-feat_vec.shape[2])).squeeze()
        print(feat_vec.shape)
        audio_features.append(feat_vec)


    audio_features = torch.stack(audio_features)

    return audio_features

def crop_video(ids, directory="data/s1video/", output_dir="data/cropped/", offset=60):

    video_features = []

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    	for idx, id in enumerate(ids[200:]):
			if idx % 50 == 0:
				print("finsihed",idx)

			try:
				filename = directory + id + ".mpg"
				# print(filename)

				vframes,_,_ = torchvision.io.read_video(filename, pts_unit="sec")
				# video_features.append(vframes)

				cropped_frames = []
				for j in range(vframes.shape[0]):

					# load the input image, resize it, and convert it to grayscale
					image = vframes[j].numpy()
					# print("original shape", image.shape)
					image = imutils.resize(image, width=500)
					gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
					# detect faces in the grayscale image
					rect = detector(gray, 1)[0]

					# determine the facial landmarks for the face region, then
					# convert the landmark (x, y)-coordinates to a NumPy array
					shape = predictor(gray, rect)
					shape = face_utils.shape_to_np(shape)

					# Calculate center of bounding rect and use to resize image around mouth
					# New shape is 120 x 120 like in the paper
					(x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]]))
					center_x = x + w//2
					center_y = y + h//2

					roi = image[center_y-offset:center_y+offset, center_x-offset:center_x+offset]
					cropped_frames.append(roi)
					# print("new shape", roi.shape)

					# show the particular face part
					# cv2.imshow("ROI", roi)
					# cv2.waitKey(0)
					# cv2.destroyAllWindows()

				# Convert to tensor
				cropped_video = torch.LongTensor(cropped_frames)

				# print(cropped_video.shape)
				torchvision.io.write_video(output_dir + id + ".mp4", cropped_video, 21)

			except:
				print("error on ", id)
				error_files.append(id)

		print(error_files)


        # video_features = torch.stack(video_features)
        # print(video_features.shape)
        #
        # return video_features
def preprocess_frames(video,output_dir,num_frames=75):
    processed_frames = []
    for i in range(video.shape[0]):
        frame = video[i,:,:,:].numpy()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(output_dir + str(i) + ".png", gray_frame)
        processed_frames.append(gray_frame)

    black_frame = np.zeros((120,120))
    for i in range(num_frames - video.shape[0]):
        frame_num = i + video.shape[0]
        cv2.imwrite(output_dir + str(frame_num) + ".png", black_frame)
        processed_frames.append(black_frame)

    return np.array(processed_frames)

def get_word_splits(align_file):
    f = open(align_file, "r")
    lines = f.readlines()
    splits = []
    for line in lines:
        words = line.split()
        append = ((words[0],words[1]),words[2])
        splits.append(append)
    return splits

def load_video(ids, align_dir="data/align/",input_dir="data/cropped/", out_dir="data/processed/"):
    print("num of vids: ", len(ids))

    # loop through all ids
    for i,id in enumerate(ids[0:10]):
        # create video file and alignment filenames
        infile = input_dir + id + ".mp4"
        align_file = align_dir + id + ".align"

        #create the output directory if it does not already exist
        out_dir_id = out_dir + id
        try:
            os.mkdir(out_dir_id)
            print("directory created for id: ", id)
        except:
            print("directory already created for id: ", id)

        # load in the video and the audio/frame splits
        vframes,_,_ = torchvision.io.read_video(infile, pts_unit="sec")
        processed_frames = preprocess_frames(vframes, out_dir_id + "/")
        print(processed_frames.shape)

if __name__ == "__main__":
    ids = get_ids()
    # load_audio(ids,"data/s1audio/")
    # crop_video(ids, "data/s1video/")
    load_video(ids)
