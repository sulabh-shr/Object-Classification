import cv2

def show_frame(frame, name, x_size=712, y_size=400, pos_x=None, pos_y=None):
    """
    It is used to create and display a window for a particular frame.

    :param frame: image to display
    :param name: name of the window
    :param x_size: width of window
    :param y_size: height of the window
    :param pos_x: window starting x pos
    :param pos_y: window starting y pos
    """

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, x_size, y_size)
    if pos_x is not None:
        cv2.moveWindow(winname=name, x=pos_x, y=pos_y)
    cv2.imshow(name, frame)

