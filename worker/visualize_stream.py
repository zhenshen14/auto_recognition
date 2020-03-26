import logging
import time
from threading import Thread

import cv2
from worker.state import State
from worker.video_reader import VideoReader
from worker.video_writer import VideoWriter


class Visualizer:
    def __init__(self, state: State, coord, car_number, color=(0, 0, 255), thick=2, font_scale=1.2, font=cv2.FONT_HERSHEY_SIMPLEX):
        self.state = state
        self.coord_x, self.coord_y = coord
        self.color = color
        self.thickness = thick
        self.font_scale = font_scale
        self.font = font
        self.car_number = car_number
        self.accuracy = 0.0

    def _draw_ocr_text(self, text, frame):
        i = 0
        n = 0
        while (i < len(text) and i < len(self.car_number)):
            if text[i] == self.car_number[i]:
                n += 1
            i += 1
        curr_acc = n/len(self.car_number)
        if curr_acc > self.accuracy:
            self.accuracy = curr_acc
        if text:
            frame = cv2.putText(frame, text,
                        (self.coord_x,self.coord_y),
                        self.font,
                        self.font_scale,
                        self.color,
                        self.thickness)
        return frame

    def __call__(self, text, frame):
        frame = self._draw_ocr_text(text, frame)
        return frame


class VisualizeStream:
    def __init__(self, name,
                 in_video: VideoReader,
                 state: State, video_path, fps, frame_size, coord, car_number):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.state = state
        self.coord = coord
        self.fps = fps
        self.frame_size = tuple(frame_size)


        self.out_video = VideoWriter("VideoWriter", video_path, self.fps, self.frame_size)
        self.sleep_time_vis = 1. / self.fps
        self.in_video = in_video
        self.stopped = True
        self.visualize_thread = None

        self.visualizer = Visualizer(self.state, self.coord, car_number)

        self.logger.info("Create VisualizeStream")

    def _visualize(self):
        try:
            while True:
                if self.stopped:
                    return
                frame = self.state.frame
                if frame is None:
                    continue
                text = self.state.text
                frame = cv2.resize(frame, self.frame_size)
                out_frame = self.visualizer(text,frame)
                self.out_video.write(out_frame)
                #TODO: Read && resize (if needed) then use visualizer to put text on frame
                # then save video with VideoWriter
                time.sleep(self.sleep_time_vis)

        except Exception as e:
            self.logger.exception(e)
            self.state.exit_event.set()

    def start(self):
        self.logger.info("Start VisualizeStream")
        self.stopped = False
        self.visualize_thread = Thread(target=self._visualize, args=())
        self.visualize_thread.start()
        # self.in_video.start()

    def stop(self):
        self.logger.info("Accuracy:{}".format(self.visualizer.accuracy))
        self.logger.info("Stop VisualizeStream")
        self.stopped = True
        self.out_video.stop()
        if self.visualize_thread is not None:
            self.visualize_thread.join()
