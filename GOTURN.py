from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k
import cv2 as cv
import numpy as np

class GOTURN(Tracker):
    """Example on how to define a tracker.

        To define a tracker, simply override ``init`` and ``update`` methods
            from ``Tracker`` with your own pipelines.
    """
    def __init__(self):
        super(GOTURN, self).__init__(
            name='GOTURN', # name of the tracker
            is_deterministic=True   # deterministic (True) or stochastic (False)
        )
    
    def init(self, image, box):
        """Initialize your tracking model in the first frame
        
        Arguments:
            image {PIL.Image} -- Image in the first frame.
            box {np.ndarray} -- Target bounding box (4x1,
                [left, top, width, height]) in the first frame.
        """

        # Convert PIL to numpy for opencv
        image = np.array(image) 
        # Convert RGB to BGR 
        image = image[:, :, ::-1].copy() 

        # to ensure all frames are of the same shape
        self.height, self.width = image[:,:,0].shape

        # convert ndarray to tuple for opencv
        box = tuple(box)

        # Initialize tracker
        self.tracker = cv.TrackerGOTURN_create()
        self.tracker.init(image, box)


    def update(self, image):
        """Locate target in an new frame and return the estimated bounding box.
        
        Arguments:
            image {PIL.Image} -- Image in a new frame.
        
        Returns:
            np.ndarray -- Estimated target bounding box (4x1,
                [left, top, width, height]) in ``image``.
        """

        image = np.array(image) 
        # Convert RGB to BGR 
        image = image[:, :, ::-1].copy() 

        image = cv.resize(image, (self.width, self.height))

        ok, box = self.tracker.update(image)


        if ok:
            self.box = np.array(box)
            
        


        return self.box

if __name__ == '__main__':
    # setup tracker
    tracker = GOTURN()

    # setup experiment (validation subset)
    experiment = ExperimentGOT10k(
        root_dir="data/GOT-10k",          # GOT-10k's root directory
        subset='val',               # 'train' | 'val' | 'test'
        result_dir='results',       # where to store tracking results
        report_dir='reports'        # where to store evaluation reports
    )

    # run experiments on GOT-10k
    experiment.run(tracker, visualize=False)

    # report performance on GOT-10k (validation subset)
    experiment.report([tracker.name])