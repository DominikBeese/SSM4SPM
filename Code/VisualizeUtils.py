''' Author: Dominik Beese
>>> Utility functions for data visualization.
<<<
'''

import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

from matplotlib import pyplot as plt

import numpy as np


### VIDEO ###

class VideoWriter:
	def __init__(self, filename, fps=30.0, **kw):
		""" Create a new Video Writer to write to the file [filename]."""
		self.writer = None
		self.params = dict(filename=filename, fps=fps, **kw)
	
	def add(self, img):
		""" Add the frame [img] to the video. """
		img = np.asarray(img)
		if self.writer is None:
			h, w = img.shape[:2]
			self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
		
		if img.dtype in [np.float32, np.float64]:
			img = np.uint8(img.clip(0, 1) * 255)
		
		if len(img.shape) == 2:
			img = np.repeat(img[..., None], 3, -1)
		
		self.writer.write_frame(img)
	
	def close(self):
		if self.writer:
			self.writer.close()
	
	def __enter__(self):
		return self
	
	def __exit__(self, *kw):
		self.close()

def zoom(img, scale: int = 4):
	img = np.repeat(img, scale, 0)
	img = np.repeat(img, scale, 1)
	return img

def make_video_from_frames(frames, filename, fps:float=30.0, scale:float=1):
	with VideoWriter(filename, fps=fps) as vid:
		for frame in frames:
			vid.add(zoom(frame, scale=scale))


### IMAGE ###

def show_image(img):
	plt.imshow(img)
	plt.show()
