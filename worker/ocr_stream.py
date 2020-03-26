import logging
from threading import Thread
from cnd.ocr.predictor import Predictor
from worker.state import State
from worker.video_reader import VideoReader
import time

class OcrStream:
    def __init__(self, name, state: State, video_reader: VideoReader):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.state = state
        self.video_reader = video_reader
        self.ocr_thread = None
        self.predictor = Predictor('C:\\Users\\user\\results\\experiment1\\model-052-1.928074.pth',(32, 80)) #TODO: Your Predictor
        self.logger.info("Create OcrStream")
        self.time = None
        self.frame_quantity = 0

    def _ocr_loop(self):
        try:
            self.start_time = time.time()
            while True:
                frame = self.video_reader.read()
                pred = self.predictor.predict(frame)
                self.state.text = pred
                self.state.frame = frame
                self.frame_quantity += 1
                if self.frame_quantity % 10 == 0:
                    print(self.frame_quantity / (time.time() - self.start_time))

        except Exception as e:
            self.logger.exception(e)
            self.state.exit_event.set()

    def _start_ocr(self):
        self.ocr_thread = Thread(target=self._ocr_loop)
        self.ocr_thread.start()

    def start(self):
        self._start_ocr()
        self.logger.info("Start OcrStream")

    def stop(self):
        if self.ocr_thread is not None:
            self.ocr_thread.join()
        self.logger.info("Stop OcrStream")
