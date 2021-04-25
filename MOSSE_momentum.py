from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k
import cv2 as cv
import numpy as np
from momentum import Momentum



class MOSSEMomentum(Tracker):
    """Example on how to define a tracker.

        To define a tracker, simply override ``init`` and ``update`` methods
            from ``Tracker`` with your own pipelines.
    """
    def __init__(self):
        super(MOSSEMomentum, self).__init__(
            name='MOSSE_momentum0.2',  # name of the tracker
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

        # needed to ensure all frames are of the same shape
        self.height, self.width = image[:,:,0].shape

        # convert ndarray to tuple for opencv
        self.box = tuple(box)

        # Initialize tracker
        self.tracker = cv.TrackerMOSSE_create()
        self.tracker.init(image, self.box)

        # Initialize motion model
        cx = box[0] + (box[2]/2)
        cy = box[1] + (box[3]/2)
        self.motion_model = Momentum(cx, cy, beta=0.9)


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

        if image[:,:,0].shape[0] != self.height or image[:,:,0].shape[1] != self.width:
            image = cv.resize(image, (self.width, self.height))

        ok, box = self.tracker.update(image)

        if ok and self.motion_model.iou(box, self.box) > 0.2:
            self.box = np.array(box)

            # update motion model
            cx = box[0] + (box[2]/2)
            cy = box[1] + (box[3]/2)
            self.motion_model.update(cx, cy)
        else:
            # if tracker fails, predict using motion model
            cx, cy = self.motion_model.predict()
            x = cx - self.box[2]/2
            y = cy - self.box[3]/2
            self.box = np.array([x, y, self.box[2], self.box[3]])        


        return self.box

if __name__ == '__main__':
    # setup tracker
    tracker = MOSSEMomentum()

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