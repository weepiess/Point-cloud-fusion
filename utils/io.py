import sys,os
import numpy as np
import cv2 as cv

#########################################################
# meshObj: Data structure to parse the obj file format
# contains: vertices & faces of the mesh 
#          + 51 3D landmarks indicies on the mesh
#####################################################
class meshObj(object):
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.uv = []
        self.landmarks_3D = []

    def load_from_obj(self, path, MeshNum):
        print(path)
        #load the vertices 
        regexp = r"v\s([+-]?[0-9]*[.]?[0-9]+[e]?[+-]?[0-9]*)\s([+-]?[0-9]*[.]?[0-9]+[e]?[+-]?[0-9]*)\s([+-]?[0-9]*[.]?[0-9]+[e]?[+-]?[0-9]*)*"

        # Error handle and check for corrupt formating of file
        try:
                self.vertices = np.fromregex(path + MeshNum+".obj", regexp, ('f'))
        except IOError:
                print ("I/O error: "+path + MeshNum+".obj" +" not found!")
                quit()
        if self.vertices.shape[0] == 0:
                print ("File Format Error: vertices cannot be found in file, or the file is corrupted. Please check that "+\
                                        path + MeshNum+".obj"+ " is in valid wavefront OBJ format.")
                quit()
        #load faces
        regexp = r"f\s(\d+)\/*\d*\/*\d*\s(\d+)\/*\d*\/*\d*\s(\d+)\/*\d*\/*\d*"

        # Error handle and check for corrupt formating of file
        try:
                self.faces = np.fromregex(path + MeshNum+".obj", regexp, ('i')) -1
        except IOError:
                print ("I/O error: "+path + MeshNum+".obj" +" not found!")
                quit()
        if self.faces.shape[0] == 0:
                print ("File Format Error: faces cannot be found in file, or the file is corrupted. Please check that "+\
                                        path + MeshNum+".obj"+ " is in valid wavefront OBJ format.")
                quit()

        #load the 3D landmark indices
        self.landmarks_3D  = np.loadtxt("/".join(path.split("/")[:-1])+"/"+ 'VertexLandmarks'+str(MeshNum)+'.txt', dtype='int32').flatten()

        #check for the number of  landmarks to be 51
        # if self.landmarks_3D.shape[0] != 51:
        #         print ("File Format Error: "+'VertexLandmarks'+str(MeshNum)+'.txt' + ' file does not have exactly 51 landmarks.')
        #         quit()

        # check to make sure free vertices are not refered to in the landmark indices file, otherwise remove all free vertices
        uniq = np.unique(self.faces)
        print(uniq)
        landmarks = []
        for idx, i in enumerate(self.landmarks_3D):
                #print('idx: ',idx)
                #print('i: ',i)
                if i == -1: 
                        landmarks.append(-1)
                        if idx == 13:
                                print ("File Value Error: 3D Landmark indices cannot have an unknown value at index 13, nose tip!")
                                quit()
                        continue
                try:
                        landmarks.append(np.where(i == uniq)[0][0])
                except:
                        print ("File Format Error: 3D Landmark indices allude to free vertices which will be removed.\
                        \nPlease fix VertexLandmarks correspondences to be not use free vertices!")
                        quit()
        landmarks = np.array(landmarks)
        self.landmarks_3D = landmarks

        if self.vertices.shape[0] < uniq.shape[0]:
                print ("File Format Error: File contains references to non-existant vertices!")
                quit()

        ## correct the faces of the meshes ##
        if self.vertices.shape[0] > uniq.shape[0]:
                new_idx = range(0,len(uniq))
                idx_map = dict(zip(uniq,new_idx))
                new_faces = []
                for tri in self.faces:
                        new_tri = [idx_map[tri[0]], idx_map[tri[1]],idx_map[tri[2]]]
                        new_faces.append(new_tri)
                new_faces = np.array(new_faces)
                self.faces = new_faces
        #####

        #filter to only include vertices that are part of a face. NO free vertices
        self.vertices = self.vertices[uniq]
        return 

class meshwrl(object):
    def __init__(self):
        self.vertices = []
        self.faces = None
        self.uv3d = None
        self.uv2d = None
        self.landmarks_3D = []
        self.image = None
        self.h = 0
        self.w = 0
    
    def load_wrl(self,path):
        regexp = r"\s([+-]?[0-9]*[.]?[0-9]+[e]?[+-]?[0-9]*)\s([+-]?[0-9]*[.]?[0-9]+[e]?[+-]?[0-9]*)\s([+-]?[0-9]*[.]?[0-9]+[e]?[+-]?[0-9]*),"
        regexp2 = r"\s([+-]?[0]*[.]?[0-9]+[e]?[+-]?[0-9]*)\s([+-]?[0]*[.]?[0-9]+[e]?[+-]?[0-9]*),"
        regexp3 = r"\s([+-]?[0-9]*),\s([+-]?[0-9]*),\s([+-]?[0-9]*), -1,"
        self.vertices = np.fromregex(path+".wrl", regexp, ('f'))
        self.uv3d = np.fromregex(path+".wrl",regexp2,('f'))
        self.faces = np.fromregex(path+".wrl",regexp3,('i'))
        self.image = cv.imread(path+'.bmp')
        [self.h, self.w, _] = self.image.shape


#writing obj
def write_obj_with_colors(obj_name, vertices, triangles):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
    '''
    triangles = triangles.copy()
    triangles += 1 # meshlab start with 1
    
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
        
    # write obj
    with open(obj_name, 'w') as f:
        
        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2],
                vertices[i, 5],vertices[i, 4],vertices[i,3])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            # s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            f.write(s)

def write_obj_without_tri(obj_name, vertices):

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
        
    # write obj
    with open(obj_name, 'w') as f:
        
        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2])
            f.write(s)

def write_obj_with_ccolors(obj_name, vertices, triangles, colors):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
    '''
    triangles = triangles.copy()
    triangles += 1 # meshlab start with 1
    
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
        
    # write obj
    with open(obj_name, 'w') as f:
        
        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2],colors[i,2],colors[i,1],colors[i,0])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            # s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            f.write(s)