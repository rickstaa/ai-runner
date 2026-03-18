import logging

from .interface import Pipeline, BaseParams
from ..trickle import VideoFrame


class Noop(Pipeline):
    def transform(self, frame: VideoFrame, params: BaseParams):
        return frame.tensor.clone()

    def on_ready(self, **params):
        logging.info(f"Initializing Noop pipeline with params: {params}")
        logging.info("Pipeline initialization complete")

    def on_update(self, **params):
        logging.info(f"Updating params: {params}")

    def on_stop(self):
        logging.info("Stopping pipeline")

    @classmethod
    def prepare_models(cls):
        logging.info("Noop pipeline does not require model preparation")
