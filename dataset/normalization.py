import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm  # for colormap
import torch
from torch.nn.functional import normalize

#from quaternion_ops import qnormalize, qbetween, qbetween_np, qrot, qrot_np

# compute a local base system for each timestamp
def get_local_bases(joints_t):
    """
        expects joints to have defined order where hips and shoulders are 1,2 and 15,16
        joints [B,T,J,C] (C - coordinates in global space)

        yields a [B,T,3,3] rotation matrix and the global root position [B,T,3] where y component is 0 (only 2D root_position in xz)
    """
    l_hip = joints_t[:,:,2, :]
    r_hip = joints_t[:,:,1, :]
    l_shoulder = joints_t[:,:,17, :]
    r_shoulder = joints_t[:,:,16, :]

    across1 = r_hip - l_hip
    across2 = r_shoulder - l_shoulder
    
    across = across1 + across2 # to get avg across-body direction
    across[:,:,1]=0.
    #across = joints_t[:,:,21, :]-joints_t[:,:,20, :] #wrists for testing
    
    across = normalize(across,p=2,dim=2) # dividing across-vector by its length

    #print("across", across.shape)
    up=torch.tensor([0, 1, 0], dtype=across.dtype, device=across.device).repeat(across.shape[0],across.shape[1],1)
    #print("up ", up.shape)
    forward_vector = torch.cross( across, up, axis=2) # kreuzprodukt aus vektor in vertical y-richtung und across-vektro gibt forward vektor
    forward_vector = normalize(forward_vector,p=2,dim=2)
    #print("forward in get_local_base ",forward_vector)
    up_vector=torch.tensor([0, 1, 0], dtype=across.dtype, device=across.device).repeat(joints_t.shape[0],joints_t.shape[1],1)
    left_vector=torch.cross(up_vector,forward_vector,dim=2)
    left_vector = normalize(left_vector,p=2,dim=2)
    
    bases=torch.stack((left_vector,up_vector,forward_vector),dim=3)
    #print("bases.shape ",bases.shape)
    #bases=bases.transpose(3,2)

    root_poses=joints_t[:,:,0,:].clone()
    #root_poses[:,:,1] *= 0.
    return bases, root_poses


def normalize_skeletons(batch):
    """
    applies the transformation to the local coordinates and computes the relative joint positions as well as the root joint movement
    batch has shape [B,T,J,3]
    returns a batch with two additional joints (representing the movement of the root and the rotation difference (around y-axis) (as forward pointing vector))
    """
    #compute the transformation matrices and offsets for each timestep
    bases,root_poses=get_local_bases(batch)
   
    #apply root transformation to all joints
    # translation offset
    batch_shifted=batch-root_poses.unsqueeze(2).repeat(1,1,batch.shape[2],1)
    # rotation to local coordinates by matrix multiplication
    mats=bases.unsqueeze(2).repeat(1,1,batch.shape[2],1,1) # copy the matrix once for each joint
    vecs=batch_shifted.unsqueeze(4) # need to add a dim for batched matmul   
    batch_local=torch.matmul(mats.transpose(4,3),vecs).reshape(batch.shape) # get rid of additional matrix dimension, so we have 3d vectors again

    #compute the delta of the root position (loose one time step)
    delta_root=root_poses[:,1:,:]-root_poses[:,0:-1,:]
    #rotate into local coordinates by matrix multiplication 
    delta_root=delta_root.unsqueeze(3) # need additional dim for batched matmul
    delta_root_local=torch.matmul(bases[:,:-1,:,:].transpose(3,2),delta_root).reshape((root_poses.shape[0],root_poses.shape[1]-1,1,3)) #reshaped to batch shape
    
    #clone the last element to have one for each timestamp
    delta_root_local=torch.cat((delta_root_local, delta_root_local[:,-1:,:,:]), dim=1)
    
    #compute the orientations of the following bases t+1 seen from local coordinates at t
    # this represents the rotation of the root in the next frame
    forward_vectors=bases[:,1:,:,2:3] #get only forward vectors from  the rotation matrices since they are sufficient to reconstruct the orthonormal matrix
    forward_local=torch.matmul(bases[:,:-1,:,:].transpose(2,3),forward_vectors).reshape((root_poses.shape[0],root_poses.shape[1]-1,1,3)) #rotated into local coordinates
    forward_local=torch.cat((forward_local, forward_local[:,-1:,:,:]), dim=1) #repeat last element to fill up time steps
    
    #join all together
    batch_normalized=torch.cat((delta_root_local,forward_local,batch_local),dim=2)
    
    return batch_normalized # [B,T,J+2,3]

def get_start_poses(batch):
    """
        extracts the startposition and orientation from a raw batch.
        This is used as necessary information for the reconstruction in denormalize_skeletons
    """
    #uses the get_local_bases method for simplicity but could be more efficient if only the first t would be computed
    bases,roots=get_local_bases(batch)
    
    startpos=roots[:,0,:] #first time step of roots in global coordinates
    startorient=bases[:,0,:,2] # third column is the forward vector
    return startpos,startorient


def denormalize_skeletons(batch, startpos,startorientation):
    """
        input batch [B,T,J,3] (normalized batch where first two joints are delta root and delta orientation)
        startpos [B,3] root positions for the first time stamp
        startorientation [B,3] the forward vector of the first pose in the sample

        reconstructs the joint poses in absolute coordinates from a normalized batch
        returns [B,T,J,3]
    """    
    #compute rotation matrices for all timestamps (this unfortunately is defined recursively)
    # the bases are 3x3 rotation matrices that transform from local to global coordinates
    bases=torch.tensor([0, 1, 0], dtype=batch.dtype, device=batch.device)[None,None,:,None].repeat(batch.shape[0],batch.shape[1],1,3) # matrices with all up vectors
    #put start orientations into the first elements
    bases[:,0,:,2]=startorientation[:,:]
    #compute the orthogonal component
    bases[:,0,:,0]=torch.cross(bases[:,0,:,1],bases[:,0,:,2],dim=1)
    #recursively construct the base matrices from its predecessor and the forward vector stored in the normalized samples
    for t in range(1,batch.shape[1]):
        #get forward vectors from batch and fill it into the matrices
        bases[:,t,:,2]=batch[:,t-1,1,:] # orientation is encoded joint 1 in t-1
        #compute orthogonal vectors for the rot mat
        bases[:,t,:,0]=torch.cross(bases[:,t,:,1],bases[:,t,:,2],dim=1)
        #apply the base from the last timestep to get a transformation into global coordiantes
        bases[:,t,:,:]=torch.matmul(bases[:,t-1,:,:],bases[:,t,:,:])

    #now we can reconstruct the root positions
    root_pos=startpos[:,None,:].repeat(1,batch.shape[1],1) # once for each timestep initialize with teh startpose
    for t in range(1,batch.shape[1]):
        # next root_pos is old rootposin global coordinates plus delta vector rotated also to global coordinates using the bases
        root_pos[:,t,:]= root_pos[:,t-1,:]+ torch.matmul(bases[:,t-1,:,:],batch[:,t-1,0,:,None]).squeeze(2)

    #now transform all joints back to global pose
    joints=torch.matmul(bases[:,:,None,:,:].repeat(1,1,batch.shape[2]-2,1,1),batch[:,:,2:,:,None]).squeeze(4) # rotate all joints from local to global coordinates
    joints=joints+root_pos[:,:,None,:].repeat(1,1,batch.shape[2]-2,1) # add the root position again
    return joints # [B,T,J,3]
