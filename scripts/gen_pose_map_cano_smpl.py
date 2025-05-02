

import numpy  as np
import torch
from os.path import join
import os
import sys
sys.path.append('../')
from submodules  import smplx

from scipy.spatial.transform import Rotation as R
import trimesh
from utils.general_utils import load_masks, load_barycentric_coords, gen_lbs_weight_from_ori
# from arguments import smplx_cpose_param, smpl_cpose_param

import math


# Function to convert Euler angles to rotation matrix
def euler_to_matrix(euler_angles, order='XYZ'):
    # Convert degrees to radians
    euler_angles = np.radians(euler_angles)
    
    # Extract individual angles
    roll, pitch, yaw = euler_angles
    
    # Rotation matrices for each axis
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    # Combine the rotation matrices based on the order
    if order == 'XYZ':
        return np.dot(R_z, np.dot(R_y, R_x))
    elif order == 'ZYX':
        return np.dot(R_x, np.dot(R_y, R_z))
    # Add other orders if needed (e.g., 'XZY', 'YXZ', etc.)
    return None

# Function to convert rotation matrix to axis-angle
def matrix_to_axis_angle(R):
    angle = np.arccos((np.trace(R) - 1) / 2)
    
    # If angle is close to zero, the axis is arbitrary (we return a zero vector)
    if angle < 1e-6:
        axis = np.zeros(3)
    else:
        axis = np.array([(R[2, 1] - R[1, 2]) / (2 * np.sin(angle)),
                         (R[0, 2] - R[2, 0]) / (2 * np.sin(angle)),
                         (R[1, 0] - R[0, 1]) / (2 * np.sin(angle))])
    
    return axis * angle


leg_angle = 30
smplx_cpose_param = torch.zeros(1, 165)
smplx_cpose_param[:, 5] =  leg_angle / 180 * math.pi
smplx_cpose_param[:, 8] = -leg_angle / 180 * math.pi
oula_arm_l = euler_to_matrix(np.array([-90, 0, 0]) / 180 * np.pi, 'XYZ')
axis_arm_l = matrix_to_axis_angle(oula_arm_l)

smpl_cpose_param = torch.zeros(1, 72)
smpl_cpose_param[:, 5] =  leg_angle / 180 * math.pi
smpl_cpose_param[:, 8] = -leg_angle / 180 * math.pi



def render_posmap(v_minimal, faces, uvs, faces_uvs, img_size=32):
    '''
    v_minimal: vertices of the minimally-clothed SMPL body mesh
    faces: faces (triangles) of the minimally-clothed SMPL body mesh
    uvs: the uv coordinate of vertices of the SMPL body model
    faces_uvs: the faces (triangles) on the UV map of the SMPL body model
    '''
    from posmap_generator.lib.renderer.gl.pos_render import PosRender

    # instantiate renderer
    rndr = PosRender(width=img_size, height=img_size)

    # set mesh data on GPU
    rndr.set_mesh(v_minimal, faces, uvs, faces_uvs)

    # render
    rndr.display()

    # retrieve the rendered buffer
    uv_pos = rndr.get_color(0)
    uv_mask = uv_pos[:, :, 3]
    uv_pos = uv_pos[:, :, :3]

    uv_mask = uv_mask.reshape(-1)
    uv_pos = uv_pos.reshape(-1, 3)

    rendered_pos = uv_pos[uv_mask != 0.0]

    uv_pos = uv_pos.reshape(img_size, img_size, 3)

    # get face_id (triangle_id) per pixel
    face_id = uv_mask[uv_mask != 0].astype(np.int32) - 1

    assert len(face_id) == len(rendered_pos)

    return uv_pos, uv_mask, face_id

def save_obj(data_path, name):
    smpl_data = torch.load( data_path + '/smpl_parms.pth')
    smpl_model = smplx.SMPL(model_path ='../assets/smpl_files/smpl',batch_size = 1)
    cano_dir = os.path.join(data_path,)


    cano_smpl = smpl_model.forward(betas=smpl_data['beta'],
                            global_orient=smpl_cpose_param[:, :3],
                            transl = torch.tensor([[0, 0.30, 0]]),
                            # global_orient=cpose_param[:, :3],
                            body_pose=smpl_cpose_param[:, 3:],
                            )

    ori_vertices = cano_smpl.vertices.detach().cpu().numpy().squeeze()
    joint_mat = cano_smpl.A
    print(joint_mat.shape)
    torch.save(joint_mat ,join(cano_dir, 'smpl_cano_joint_mat.pth'))


    mesh = trimesh.Trimesh(ori_vertices, smpl_model.faces, process=False)
    mesh.export('%s/%s.obj' % (cano_dir, 'cano_smpl'))



def save_npz(data_path, res=128):
    from posmap_generator.lib.renderer.mesh import load_obj_mesh
    verts, faces, uvs, faces_uvs = load_obj_mesh(uv_template_fn, with_texture=True)
    start_obj_num = 0
    result = {}
    body_mesh = trimesh.load('%s/%s.obj'%(data_path, 'cano_smpl'), process=False)

    if res==128:
        posmap128, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=128)
        result['posmap128'] = posmap128   
    elif res == 256:
    
        posmap256, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=256)
        result['posmap256'] = posmap256

    else:
        posmap512, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=512)
        result['posmap512'] = posmap512

    save_fn = join(data_path, 'query_posemap_%s_%s.npz'% (str(res), 'cano_smpl'))
    np.savez(save_fn, **result)



if __name__ == '__main__':
    smplx_parm_path = '../assets/test_pose' # path to the folder that include smpl params
    parms_name = 'smpl_parms.pth'
    uv_template_fn = '../assets/template_mesh_smpl_uv.obj'
    assets_path = '../'    # path to the folder that include 'assets'

    print('saving obj...')
    save_obj(smplx_parm_path, parms_name)

    print('saving pose_map 512 ...')
    save_npz(smplx_parm_path, 512)