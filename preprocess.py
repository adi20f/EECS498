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

def split_audio(audio, sample_rate=25000):
	# Splits audio tensor into 25 ms chunks
	# Audio is in shape (1, audio samples)
	# sample rate is num audio samples per second

	audio_length_seconds = audio.shape[1] / sample_rate
	samples_25ms = int(sample_rate * .025)

	num_chunks = np.ceil(audio.shape[1] / samples_25ms)

	audio_chunks = []
	start = 0
	end = 0

	while start + samples_25ms < audio.shape[1]:
		end += samples_25ms
		# print("start end", start, end)

		audio_chunks.append(audio[:,start:end])
		start = end

	# print(audio_chunks[0])

	if len(audio_chunks) < num_chunks:
		last_chunk = audio[:,start:audio.shape[1]]
		if len(last_chunk) < samples_25ms:
			# print("last", last_chunk.shape)
			# print("pad", last_chunk.shape)
			last_chunk = torch.nn.functional.pad(last_chunk, (0, samples_25ms-last_chunk.shape[1]))
			# print("last chunk", last_chunk)
			audio_chunks.append(last_chunk)


	audio_chunks = torch.stack(audio_chunks)

	return audio_chunks

def load_audio(ids, directory="data/s1audio/", mfcc_max_length=7, max_num_mfccs=75):
	print(len(ids), "files")
	audio_features = []

	# Extract MFCC features for each sample
	for filename in ids:

		# Load audio file and split into 25 ms chunks
		filename = directory + filename + ".wav"
		audio, sample_rate = torchaudio.load(filename)
		audio_chunks = split_audio(audio, sample_rate)
		sample_features = []

		# Iterate over chunks and compute mfccs for each 25ms segment of this audio file
		for chunk in audio_chunks:
			m = torchaudio.transforms.MFCC(sample_rate, n_mfcc=13)
			feat_vec = m(chunk)
			# Pad each mfcc feat vec so that all are the same size for each chunk
			feat_vec = torch.nn.functional.pad(feat_vec, (0, mfcc_max_length-feat_vec.shape[2])).squeeze()
			sample_features.append(feat_vec)

		# This tensor contains all mfcc feat vecs for one audio sample
		sample_features = torch.stack(sample_features)
		# Pad this tensor so that each audio sample has the same number of 25ms segments
		sample_features = torch.nn.functional.pad(sample_features, (0, 0, 0, 0, 0, max_num_mfccs-sample_features.shape[0]))
		# print("sample_feat", sample_features.shape)
		audio_features.append(sample_features)

	# This tensor has shape (num_samples, max_num_25ms_segments, 13, max_size_mfcc) -> (1000, 75, 13, 7)
	audio_features = torch.stack(audio_features)
	# print("final features", audio_features.shape)
	return audio_features

def crop_video(ids, directory="data/s1video/", output_dir="data/cropped/", offset=60):

	video_features = []

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	for idx, id in enumerate(ids):
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
def preprocess_frames(video,output_dir=None,num_frames=75,save=False):
    if save:
        if output_dir is  None:
            raise FileNotFoundError('None passed for output_dir')
        try:
            os.mkdir(output_dir)
            print("directory created for ", output_dir)
        except:
            print("directory already created for ", output_dir)

    processed_frames = []
    for i in range(video.shape[0]):
        frame = video[i,:,:,:].numpy()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if save:
            cv2.imwrite(output_dir + str(i) + ".png", gray_frame)
        processed_frames.append(gray_frame)

    black_frame = np.zeros((120,120))
    for i in range(num_frames - video.shape[0]):
        frame_num = i + video.shape[0]
        if save:
            cv2.imwrite(output_dir + str(frame_num) + ".png", black_frame)
        processed_frames.append(black_frame)

    return torch.FloatTensor(np.array(processed_frames))

def get_word_splits(align_file):
	f = open(align_file, "r")
	lines = f.readlines()
	splits = []
	for line in lines:
		words = line.split()
		append = ((words[0],words[1]),words[2])
		splits.append(append)
	return splits

def load_video(ids, input_dir="data/cropped/", out_dir="data/processed/"):
	print("num of vids: ", len(ids))

	video_features = []
	# loop through all ids
	for i,id in enumerate([ids[0]]):
		# create video file and alignment filenames
		infile = input_dir + id + ".mp4"

		#create the output directory if it does not already exist
		out_dir_id = out_dir + id
		try:
			os.mkdir(out_dir_id)
			print("directory created for id: ", id)
		except:
			print("directory already created for id: ", id)

		# load in the video and the audio/frame splits
		vframes,_,_ = torchvision.io.read_video(infile, pts_unit="sec")
		processed_frames = preprocess_frames(vframes, out_dir_id + "/", save=True)
		print(processed_frames.shape)
		video_features.append(processed_frames)

	video_features = torch.stack(video_features)
	print("input", video_features.shape)

	return video_features


if __name__ == "__main__":
	ids = get_ids()
	# load_audio(ids,"data/s1audio/")
	# crop_video(ids, "data/s1video/")
	load_video(ids)
