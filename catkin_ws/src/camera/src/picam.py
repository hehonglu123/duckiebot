from picamera import PiCamera
import picamera.array
import rospy
from sensor_msgs.msg import Image

class Webcam_impl():
    #Init the camera being passed the camera number and the camera name
    def __init__(self):
        self.camera = PiCamera()
        self.camera.framerate = 30.0
        self.camera.resolution = (640,480) #(1024,768) default
        self.stream = picamera.array.PiRGBArray(self.camera)
        self.camera_capture = self.camera.capture_continuous(self.stream,'bgr',use_video_port=True)
    def CaptureFrame(self):
        self.camera_capture.next()
        self.stream.seek(0)
        image=self.stream.array
        self.stream.truncate(0)
        return image

def loop(c,camera):
	img.data=camera.CaptureFrame()
	pub.publish(velocity)



if __name__ == '__main__':
	camera=Webcam_impl()
	pub = rospy.Publisher('picam', Image, queue_size=0)
	rospy.init_node('picam', anonymous=True)
	# rate = rospy.Rate(10)
	img=Image()
	(img.width,img.height)=camera.resolution
	loop(pub,img,camera)