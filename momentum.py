


class Momentum:
    def __init__(self, cx, cy, beta):
        '''
        cx : center x coordinate of the initial bbox
        cy : center y coordinate of the initial bbox
        beta: coeffient for the exponentially weighted average
        '''
        self.x_t = cx
        self.y_t = cy
        self.beta = beta
        self.mux_t = 0      # initialize the average in the x direction
        self.muy_t = 0      # initialize the average in the y direction

        
        # the cache will store the required averages for the predict method
        self.cache = {
                        "mux_tminus1"   : 0.0,
                        "muy_tminus1"   : 0.0,
                        "mux_tminus2"   : 0.0,
                        "muy_tminus2"   : 0.0
                    }

    def update(self, new_x, new_y):
        '''
        Update the estimate of the averages and modify cache

        Input:
        new_x = the x coordinate of the new bounding box
        new_y = the y coordinate of the new bounding box
        '''
        
        # Set current bbox as bbox on previous time step
        self.x_tminus1 = self.x_t
        self.y_tminus1 = self.y_t

        # Set new bbox as bbox for current time step
        self.x_t = new_x
        self.y_t = new_y

        # Calculate new averages
        self.mux_t = self.beta * self.cache["mux_tminus1"] + (1 - self.beta) * (self.x_t - self.x_tminus1)
        self.muy_t = self.beta * self.cache["muy_tminus1"] + (1 - self.beta) * (self.y_t - self.y_tminus1)

        # update cache for next frame
        self.cache["mux_tminus2"] = self.cache["mux_tminus1"]
        self.cache["muy_tminus2"] = self.cache["muy_tminus1"]
        self.cache["mux_tminus1"] = self.mux_t
        self.cache["muy_tminus1"] = self.muy_t

    def predict(self):
    
    	# Set current bbox as bbox on previous time step
        #self.x_tminus1 = self.x_t
        #self.y_tminus1 = self.y_t
	
	# Predict bbox for next frame
        try:
            self.x_t = self.x_tminus1 + (1.5 * self.cache["mux_tminus1"]) - (0.5 * self.cache["mux_tminus2"])
            self.y_t = self.y_tminus1 + (1.5 * self.cache["muy_tminus1"]) - (0.5 * self.cache["muy_tminus2"])
        except:
            print("Prediction failed. Motion model not properly initialized")

        return self.x_t, self.y_t
    
    @staticmethod
    def iou(bbox1, bbox2):
        bb1 = {
                "x1" : bbox1[0],
                "y1" : bbox1[1],
                "x2" : bbox1[0] + bbox1[2],
                "y2" : bbox1[1] + bbox1[3]
            }    
        
        bb2 = {
                "x1" : bbox2[0],
                "y1" : bbox2[1],
                "x2" : bbox2[0] + bbox2[2],
                "y2" : bbox2[1] + bbox2[3]
            } 

        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

        



            
