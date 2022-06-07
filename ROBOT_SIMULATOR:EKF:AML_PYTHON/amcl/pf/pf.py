class Pose():
    def __init__(self, x , y , theta) -> None:
        self.x = x
        self.y = y
        self.theta = theta

class Particle():
    def __init__(self, pose , weight) -> None:
        self.pose = pose
        self.weight = weight
  
