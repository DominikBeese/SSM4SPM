''' Author: Dominik Beese
>>> Process the Videos
	- keeps resolution
	- changes fps to 30
	- extracts individual frames
<<<
'''

from os.path import join
from os import listdir, makedirs
from subprocess import run, DEVNULL
from tqdm.auto import tqdm
import json


### CONFIGURATION ###

VIDEO_FORMAT = 'mp4' # input video format
TARGET_SIZE = None # target width and height, or None
TARGET_FPS = 30
IMAGE_FORMAT = 'png' # output image format for individual frames
EXPORT_VIDEOS = True


### DO IT ###

source_paths = [
	join('..', 'Cataract-1K-raw', 'Phase_recognition_dataset', 'videos'),
	join('..', 'Cataract-1K-raw', 'Segmentation_dataset', 'videos'),
	join('..', 'Cataract-1K-raw', 'Pupil_reaction'),
	join('..', 'Cataract-1K-raw', 'Lens_irregularity'),
]
target_path = join('..')
if TARGET_SIZE is not None: target_path = join(target_path, str(TARGET_SIZE))

makedirs(join(target_path), exist_ok=False)
def cmd(command): run(command, stdout=DEVNULL, stderr=DEVNULL)
for source_path in tqdm(source_paths):
	for case in tqdm([f for f in listdir(source_path) if f.endswith(VIDEO_FORMAT)]):
		caseId = int(case[5:-len(VIDEO_FORMAT)-1])
		# 1-step
		vid_in = join(source_path, 'videos', 'case_%d.%s' % (caseId, VIDEO_FORMAT))
		img_out = join(target_path, '%04d' % caseId, '%%05d.%s' % IMAGE_FORMAT)
		makedirs(join(target_path, '%04d' % caseId), exist_ok=True)
		if TARGET_SIZE is None: cmd(f'ffmpeg -n -i {vid_in} -vf "fps={TARGET_FPS}" "{img_out}"')
		else: cmd(f'ffmpeg -n -i {vid_in} -vf "scale={TARGET_SIZE}:{TARGET_SIZE},fps={TARGET_FPS}" "{img_out}"')
		# export video
		if not EXPORT_VIDEOS: continue
		vid_out = join(target_path, '%04d.%s' % (caseId, VIDEO_FORMAT))
		if TARGET_SIZE is None: cmd(f'ffmpeg -i {vid_in} -filter:v fps={TARGET_FPS} {vid_out}')
		else: cmd(f'ffmpeg -i {vid_in} -s {TARGET_SIZE}x{TARGET_SIZE} -filter:v fps={TARGET_FPS} {vid_out}')

manifest = dict()
if TARGET_SIZE is not None:
	manifest['width'] = TARGET_SIZE
	manifest['height'] = TARGET_SIZE
manifest['fps'] = float(TARGET_FPS)
manifest['video-format'] = VIDEO_FORMAT
manifest['image-format'] = IMAGE_FORMAT

with open(join(target_path, 'manifest.json'), 'w', encoding='UTF-8') as file:
	json.dump(manifest, file, indent=' ', ensure_ascii=False)
