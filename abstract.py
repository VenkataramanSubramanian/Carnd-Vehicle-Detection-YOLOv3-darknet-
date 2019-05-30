from darknet import *
from moviepy.editor import VideoFileClip
import sys

#Video launch code
yellow_output = sys.argv[2]
clip2 = VideoFileClip(sys.argv[1])
yellow_clip = clip2.fl_image(detect)
yellow_clip.write_videofile(yellow_output, audio=False)