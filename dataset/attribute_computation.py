import torch
from scipy.signal import find_peaks

def get_angle(a_vec, b_vec):
    inner_product = (a_vec * b_vec).sum(dim=1)
    a_norm = a_vec.norm(p=2,dim=1)+1e-8
    b_norm = b_vec.norm(p=2,dim=1)+1e-8
    cos = inner_product / ( a_norm * b_norm)
    angle = torch.acos(cos)
    return angle

class Attribute_Computation:
    """
     for the given joint order of humanML3D dataset this class is used to
     extract some deterministic attributes from time windows of skeleton animations
    """
    def __init__(self):
        self.bones=torch.tensor([[2,5],[5,8],[8,11],[1,4],[4,7],[7,10],[15,12],[12,9],[9,6],[6,3],[3,0],[17,19],[19,21],[16,18],[18,20],[10,11]])
        # 0 - "upper_left_leg": [2,5],
        # 1 - "lower_left_leg": [5,8],
        # 2 - "left_foot": [8,11],
        # 3 - "upper_right_leg": [1,4],
        # 4 - "lower_right_leg": [4,7],
        # 5 - "right_foot": [7,10],
        # 6 - "neck":[15,12],
        # 7 - "breast":[12,9],
        # 8 - "body":[9,6],
        # 9 - "navel":[6,3],
        # 10 - "pelvis":[3,0],
        # 11 - "upper_left_arm":[17,19],
        # 12 - "lower_left_arm":[19,21],
        # 13 - "upper_right_arm":[16,18],
        # 14 - "lower_right_arm":[18,20],
        # 15 - "foot_distance":[10,11],}



    def get_attributes(self, motion):
        """
        compute some artificial attributes from the raw motion [T,J,3] data, 
        that can be used for testing the quality of regression networks
        """
        bones=motion[:,self.bones[:,0],:]-motion[:,self.bones[:,1],:]
        
        left_leg_length=torch.mean(torch.norm(bones[:,0,:],p=2,dim=1),0,True)+torch.mean(torch.norm(bones[:,1,:],p=2,dim=1),0,True)
        right_leg_length=torch.mean(torch.norm(bones[:,3,:],p=2,dim=1),0,True)+torch.mean(torch.norm(bones[:,4,:],p=2,dim=1),0,True)
        left_arm_length=torch.mean(torch.norm(bones[:,11,:],p=2,dim=1),0,True)+torch.mean(torch.norm(bones[:,12,:],p=2,dim=1),0,True)
        right_arm_length=torch.mean(torch.norm(bones[:,13,:],p=2,dim=1),0,True)+torch.mean(torch.norm(bones[:,14,:],p=2,dim=1),0,True)
        
        left_knee_angles = torch.max(get_angle(bones[:,0,:],bones[:,1,:]),0,True)[0]
        right_knee_angles = torch.max(get_angle(bones[:,3,:],bones[:,4,:]),0,True)[0]

        body=bones[:,7,:]+bones[:,8,:]+bones[:,9,:]+bones[:,10,:]
        left_arm_angles=get_angle(bones[:,11,:],body)
        left_arm_elongation = torch.max(left_arm_angles,dim=0,keepdim=True)[0]
        left_arm_motion = torch.std(left_arm_angles,dim=0,keepdim=True)
        right_arm_angles=get_angle(bones[:,13,:],body)
        right_arm_elongation = torch.max(right_arm_angles,dim=0,keepdim=True)[0]
        right_arm_motion =torch.std(right_arm_angles,dim=0,keepdim=True)
        #variance of angle between the legs
        left_leg=bones[:,0,:]+bones[:,1,:]
        right_leg=bones[:,3,:]+bones[:,4,:]
        leg_angles=get_angle(left_leg,right_leg)
        stepangle=torch.std(leg_angles,dim=0,keepdim=True)
        stepanglemax=torch.max(leg_angles,dim=0,keepdim=True)[0]

        left_elbow_angles = get_angle(bones[:,11,:],bones[:,12,:])
        left_elbow_motion = torch.std(left_elbow_angles,dim=0,keepdim=True)
        left_elbow_angles = torch.max(left_elbow_angles,0,True)[0]
        right_elbow_angles = get_angle(bones[:,13,:],bones[:,14,:])
        right_elbow_motion = torch.std(right_elbow_angles,dim=0,keepdim=True)
        right_elbow_angles = torch.max(right_elbow_angles,0,True)[0]
        

        # #compute steplength and frequency
        # foot_distance=torch.abs(bones[:,15,2]).cpu()
        # print("foot_distance",foot_distance)
        # peaks,_=find_peaks(-foot_distance,width=2,distance=5,height=-0.15) #find minima
        # num_peaks = torch.tensor([peaks.shape[0]],dtype=torch.float)
        # print("num_peaks",num_peaks,peaks)
        # #average distances of consecutive peaks
        # if peaks.shape[0] >1:
        #     dtime = torch.tensor(peaks[1:]-peaks[0:-1],dtype=torch.long)
        #     steplength=torch.abs(motion[peaks[1:],10,2]-motion[peaks[0:-1],10,2])
        #     print("dtime: ",dtime)
        #     print("steplength:",steplength)
        #     steplength=torch.mean(steplength,0,True)*2.
        #     dtime=torch.mean(dtime.float(),0,True)
        # else:
        #     steplength=torch.zeros((1))
        #     dtime=torch.zeros((1))
        
        # dtime = dtime.to(device=motion.device)
        # steplength = steplength.to(device=motion.device)
        # num_peaks = num_peaks.to(device=motion.device)
        # #sample rate is 20Hz
        # attributes = torch.cat((20./dtime,steplength,num_peaks,stepangle,stepanglemax,left_arm_elongation,right_arm_elongation,left_arm_motion,right_arm_motion,left_knee_angles,right_knee_angles,left_leg_length,right_leg_length,left_arm_length,right_arm_length))
        attributes = torch.cat((stepangle,stepanglemax,left_arm_elongation,right_arm_elongation,left_arm_motion,right_arm_motion,left_knee_angles,right_knee_angles,left_leg_length,right_leg_length,left_arm_length,right_arm_length,
            left_elbow_angles,left_elbow_motion,right_elbow_angles,right_elbow_motion))
        #print("attributes",attributes.shape)

        return attributes
