import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from yolo_utils import infer_image, show_image

FLAGS = []

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	array_of_img = [] # this if for store all of the image data
	array_of_height = []
	array_of_width = []
	array_of_filename = []
	array_of_vid = []
	writer = None
	
	parser.add_argument('-m', '--model-path',
		type=str,
		default='./yolov3-coco/',
		help='The directory where the model weights and \
			  configuration files are.')

	parser.add_argument('-w', '--weights',
		type=str,
		default='./yolov3-coco/yolov3.weights',
		help='Path to the file which contains the weights \
			 	for YOLOv3.')

	parser.add_argument('-cfg', '--config',
		type=str,
		default='./yolov3-coco/yolov3.cfg',
		help='Path to the configuration file for the YOLOv3 model.')

	parser.add_argument('-i', '--image-path',
		type=str,
		help='The path to the image file')


	parser.add_argument('-iout', '--image-outputFile-path',
		type=str,
		help='The path to put the image-output file')
		
		
	parser.add_argument('-v', '--video-path',
		type=str,
		help='The path to the video file')


	parser.add_argument('-vo', '--video-output-path',
		type=str,
        default='./output.avi',
		help='The path of the output video file')
		
	parser.add_argument('-vout', '--video-outputFile-path',
		type=str,
		help='The path to put the video-output file')

	parser.add_argument('-l', '--labels',
		type=str,
		default='./yolov3-coco/coco-labels',
		help='Path to the file having the \
					labels in a new-line seperated way.')

	parser.add_argument('-c', '--confidence',
		type=float,
		default=0.5,
		help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

	parser.add_argument('-th', '--threshold',
		type=float,
		default=0.3,
		help='The threshold to use when applying the \
				Non-Max Suppresion')

	parser.add_argument('--download-model',
		type=bool,
		default=False,
		help='Set to True, if the model weights and configurations \
				are not present on your local machine.')

	parser.add_argument('-t', '--show-time',
		type=bool,
		default=False,
		help='Show the time taken to infer each image.')


	FLAGS, unparsed = parser.parse_known_args()
	

	# Download the YOLOv3 models if needed
	if FLAGS.download_model:
		subprocess.call(['./yolov3-coco/get_model.sh'])

	# Get the labels
	labels = open(FLAGS.labels).read().strip().split('\n')

	# Intializing colors to represent each label uniquely
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model
	net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
	# If both image and video files are given then raise error
	if FLAGS.image_path is None and FLAGS.video_path is None:
		print ('Neither path to an image or path to video provided')
		print ('Starting Inference on Webcam')

	# read all image/video into array
	def readDirectory_img(directory_name ):
		for filename in os.listdir(directory_name ):
			img = cv.imread(directory_name  + "/" + filename)
			array_of_img.append(img)
			height,width = img.shape[:2]
			array_of_height.append(height)
			array_of_width.append(width)
			array_of_filename.append(filename)

	# read all video into array
	def readDirectory_video(directory_name ):
		for filename in os.listdir(directory_name ):
			array_of_filename.append(filename)

	# Do inference with given image
	if FLAGS.image_path:
		# Read the image
		try:
			readDirectory_img(FLAGS.image_path)
			num = 0
			#print(FLAGS)
		except:
			raise 'Image cannot be loaded!\n\
                               Please check the path provided!'

		finally:
			num = 0
			if FLAGS.image_outputFile_path:
				output_path = FLAGS.image_outputFile_path
			else:
				output_path = FLAGS.image_path

			while( num < len(array_of_img)):
				height = array_of_height[num]
				width = array_of_width[num]
				img = array_of_img[num]
				img, _, _, _, _ = infer_image(net, layer_names, height, width, img, colors, labels, FLAGS)
				cv.imwrite(output_path + array_of_filename[num]+"_prediction.jpg",img)
				num = num + 1

	elif FLAGS.video_path:
		# Read the video
		try:
			readDirectory_video(FLAGS.video_path)
			#vid = cv.VideoCapture(FLAGS.video_path)
			#height, width = None, None
			#writer = None
		except:
			raise 'Video cannot be loaded!\n\
								Please check the path provided!'

		finally:
			num = 0
			#print(array_of_filename)
			while num < len(array_of_filename) :
				vid = cv.VideoCapture(FLAGS.video_path + "/" + array_of_filename[num])
				height, width = None, None
				writer = None
				FLAGS.video_output_path = array_of_filename[num]+"_prediction.avi"
				while True :
					
					grabbed, frame = vid.read()
					#print(array_of_filename[num],"*",time.time())
					# Checking if the complete video is read
					if not grabbed:
						break

					if width is None or height is None:
						height, width = frame.shape[:2]

					frame, _, _, _, _ = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)

					if writer is None:
						
						# Initialize the video writer
						if FLAGS.video_outputFile_path : 
							output_path = FLAGS.video_outputFile_path + FLAGS.video_output_path
							fourcc = cv.VideoWriter_fourcc(*"MJPG")
							writer = cv.VideoWriter(output_path, fourcc, 30,
										(frame.shape[1], frame.shape[0]), True)
						else:
							output_path = FLAGS.video_path + FLAGS.video_output_path
							fourcc = cv.VideoWriter_fourcc(*"MJPG")
							writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30, 
										(frame.shape[1], frame.shape[0]), True)

					writer.write(frame)
				 
				writer.release()
				vid.release()
				#print(num)
				num = num + 1 #out of while True
				
			print ("[INFO] Cleaning up...") #out of while num < len()
			#writer.release()
			#vid.release()


	else:
		# Infer real-time on webcam
		count = 0

		vid = cv.VideoCapture(0)
		while True:
			_, frame = vid.read()
			height, width = frame.shape[:2]

			if count == 0:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
		    						height, width, frame, colors, labels, FLAGS)
				count += 1
			else:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
		    						height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
				count = (count + 1) % 6

			cv.imshow('webcam', frame)

			if cv.waitKey(1) & 0xFF == ord('q'):
				break
		vid.release()
		cv.destroyAllWindows()
