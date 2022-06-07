class AMCLODOM():
    def __init__(self) -> None:
        pass

    '''
    sets alpha variable for diffrbetial drive following motion model
     updatee check notes motion update odometry page 27 note 6
    '''
    def setModelDiffrentialDriveErrorVariable(self,alpha1, alpha2, alpha3 , alpha4):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4
    def UpdateAction(self,pf, data)->bool:
        
    