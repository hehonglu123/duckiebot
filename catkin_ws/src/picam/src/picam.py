from picamera import PiCamera
import picamera.array
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Webcam_impl():
    #Init the camera being passed the camera number and the camera name
    def __init__(self):
        self.camera = PiCamera()
        self.camera.framerate = 30.0
        self.camera.resolution = (640,480)
        self.stream = picamera.array.PiRGBArray(self.camera)
        self.camera_capture = self.camera.capture_continuous(self.stream,'bgr',use_video_port=True)
    def CaptureFrame(self):
        self.camera_capture.next()
        self.stream.seek(0)
        image=self.stream.array
        self.stream.truncate(0)
        return image


if __name__ == '__main__':
	picam=Webcam_impl()
	pub = rospy.Publisher('picam', Image, queue_size=0)
	rospy.init_node('picam', anonymous=True)
    bridge=CvBridge()
	img=Image()
	(img.width,img.height)=picam.camera.resolution
    while True:
        try:
            frame=picam.CaptureFrame()
            pub.publish(bridge.cv2_to_imgmsg(frame,"bgr8"))
        except KeyboardInterrupt:
            print("Shutting down")
