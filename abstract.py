from moviepy.editor import VideoFileClip
import darknet


vehicle_det = 'project_video_output.mp4'
clip2 = VideoFileClip('project_video.mp4')
vehicle_clip = clip2.fl_image(darknet.detect)
vehicle_clip.write_videofile(vehicle_det, audio=False)
