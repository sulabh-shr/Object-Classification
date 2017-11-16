import cv2
from parameters import *
from webcam_video_stream import WebcamVideoStream
from show_frame import show_frame


if __name__ == '__main__':
    camera = WebcamVideoStream(src=VIDEO_SOURCE_INPUT).start()
    while True:
        frame = camera.read()
        show_frame(frame, 'Original', x_size=WIN_X, y_size=WIN_Y, pos_x=None, pos_y=None)

        # Waiting for exit key
        key = cv2.waitKey(1)
        if (key & 0xFF == ord('q')) | (key & 0xFF == 27):
            break
        elif key == ord('c'):
            cv2.imwrite("persp.jpg", frame)

    camera.stop()
    cv2.destroyAllWindows()
