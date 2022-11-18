import os
import json
import argparse
import random
import csv
from pydub import AudioSegment

def create_json_data(args):

	data = []
	directory = args.file_path.rpartition('/')[0]

	with open(args.file_path, encoding='utf-8') as f:
		lenght = sum(1 for line in f)

	with open(args.file_path, newline='', encoding='utf-8') as csvfile: 
		reader = csv.DictReader(csvfile, delimiter='\t')
		index = 1
		if args.convert:
			print(str(lenght) + " files found")
		for row in reader:  
			file_name = row['path']
			
			filename = file_name + "." + args.file_extension
			text = row['sentence']
			if args.convert:
				print("converting file " + str(index) + "/" + str(lenght) + " to " + args.file_extension, end="\r")
				src = args.audio_source_file_path + "/" + file_name + ".mp3"
				dst = args.audio_destination_file_path + "/"  + filename
				try:
					sound = AudioSegment.from_mp3(src)
					sound.export(dst, format=args.file_extension)
					data.append({
					"key":  "data/wav_clips/" + filename,
					"text": text
					})
				except Exception as e:
					print("could not convert {} to {}".format(filename, args.file_extension))
					print(e)
				index = index + 1
			else:
				data.append({
				"key": directory + "/wav_clips/" + file_name,
				"text": text
				})

	random.shuffle(data)
	print("creating JSON's")
	with open(args.save_json_path, "w") as file:
		json.dump(data , file) 

	print("Done!")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="""
	Utility script to convert commonvoice into wav and create the training and test json files for speechrecognition. """
	)
	parser.add_argument('--file_path', type=str, default=None, required=True,
						help='path to one of the .tsv files found in cv-corpus')
	parser.add_argument('--save_json_path', type=str, default=None, required=True,
						help='path to the dir where the json files are supposed to be saved')
	parser.add_argument('--audio_source_file_path', type=str, default=None, required=True,
						help='path to the dir where the audio files are')
	parser.add_argument('--audio_destination_file_path', type=str, default=None, required=True,
						help='path to the dir where the audio files are supposed to be saved')
	parser.add_argument('--file_extension', default="wav", action='store_true',
						help='saves the audio files in the desired file extension')
	parser.add_argument('--convert', default=True, action='store_true',
						help='says that the script should convert mp3 to wav')

	
	args = parser.parse_args()

	create_json_data(args)