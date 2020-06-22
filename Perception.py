import sys
sys.path.append('../../build')
import libry as ry
import time 
import cv2
import numpy as np
from sklearn.cluster import KMeans


class Perceptor(object):
    def __init__(self, no_obj):

        self.RealWorld = ry.Config()
        self.RealWorld.addFile("../../scenarios/challenge.g")
        self.Model = ry.Config()
        self.Model.addFile('../../scenarios/pandasTable.g')
        self.Model_Viewer = ry.ConfigurationViewer()
        self.Model_Viewer.setConfiguration(self.Model)
        self.camera_Frame = self.Model.frame('camera')
        self.no_obj = no_obj
        self.reorder_objects(self.no_obj)
        self.Simulation = self.RealWorld.simulation(ry.SimulatorEngine.physx, True)
        self.Simulation.addSensor('camera')
        self.set_focal_length(0.895)
        self.tau = 0.01
        [self.background_rgb, self.background_depth] = self.Simulation.getImageAndDepth()
        self.background_gray = cv2.cvtColor(self.background_rgb, cv2.COLOR_BGR2GRAY)
        print('Init successful!')


    def reorder_objects(self, no_obj):
        for i in range(0, 30):
            name = "obj%i" % i
            self.RealWorld.delFrame(name)
        for i in range(1, no_obj+1):
            obj_name = 'Sphere_{}'.format(i)
            spawn_object = self.RealWorld.addFrame(obj_name)
            spawn_object.setShape(ry.ST.sphere, [0.03])
            spawn_object.setColor([1,0,0])
            spawn_object.setPosition([0., .1*i, 2.])
            spawn_object.setMass(0.1)
            spawn_object.setContact(1)

        self.Model_Viewer.recopyMeshes(self.Model)
        self.Model_Viewer.setConfiguration(self.Model)
        # print(self.RealWorld.getFrameNames())


    def set_focal_length(self, f):
        self.fxfypxpy = [f*360.0, f*360.0, 320., 180.]

    def update_binary_mask(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        background_diff = abs(self.background_gray - gray)
        _, self.bin_mask = cv2.threshold(background_diff, 0, 255, cv2.THRESH_BINARY)

    def update_point_cloud(self, rgb, depth):
        depth_mask = cv2.bitwise_and(depth, depth, mask=self.bin_mask)
        # depth_indices = np.where(depth_mask>0)
        # values = np.empty((depth_indices[0].shape[0], 3))
        # values[:,0] = depth_indices[1]
        # values[:,1] = depth_indices[0]
        # values[:,2] = depth[depth_indices[0], depth_indices[1]]
        # points = self.Simulation.depthData2pointCloud(values, self.fxfypxpy)
        points = self.Simulation.depthData2pointCloud(depth_mask, self.fxfypxpy)
        # self.current_object_points = [j for i in points for j in i if j[2]!=0]
        self.current_object_points = [j for i in points for j in i if j[0]!=0 and j[1]!=0 and j[2]!=0]
        self.camera_Frame.setPointCloud(points, rgb)
        self.Model_Viewer.recopyMeshes(self.Model)
        self.Model_Viewer.setConfiguration(self.Model)

    def step(self):
        self.Simulation.step([], self.tau, ry.ControlMode.none)

    def get_centers(self, no_objects):
        kmeans = KMeans(n_clusters=no_objects).fit(self.current_object_points)
        return kmeans.cluster_centers_
        # return np.mean(self.current_object_points, axis=0)

    def create_shapes(self, centers):
        for i, center in enumerate(centers):
            obj_name = 'Approximation_object_{}'.format(i+1)
            appr_object = self.Model.addFrame(obj_name)
            appr_object.setShape(ry.ST.sphere, [0.03])
            appr_object.setColor([0,1,0])
            self.Model.attach('camera', obj_name)
            appr_object.setRelativePosition(center)
            print('Spawning Model object {}!'.format(i+1))
        self.Model_Viewer.recopyMeshes(self.Model)
        self.Model_Viewer.setConfiguration(self.Model)
            # time.sleep(0.01)
            # print(self.Model.getFrameNames())

    def move_pregrasp_unoccluded(self, target_object):
        komo = self.RealWorld.komo_path(1, 20, 5., True)
        komo.addObjective([1.], ry.FS.scalarProductXY, ['R_gripper', 'world'], ry.OT.eq, [1.], target=[1.])
        z_target = [1/np.sqrt(2), 0, 1/np.sqrt(2)]
        komo.addObjective([1.], ry.FS.vectorZ, ['R_gripper'], ry.OT.eq, [1.], target=z_target)
        komo.addObjective([1.], ry.FS.positionDiff, ['R_gripper', target_object], ry.OT.eq, [1.], target=[0.2,0,0])
        komo.optimize()
        komo_viewer = komo.view()
        time.sleep(10)
        # komo_viewer.playVideo(delay=0.1)



if __name__ == "__main__":
    detector = Perceptor(5)

    for t in range(2000):
        # time.sleep(0.01)

        if t%10 == 0:
            [rgb, depth] = detector.Simulation.getImageAndDepth() 
            detector.update_binary_mask(rgb)
            detector.update_point_cloud(rgb, depth)

        if t==250:
            centers = detector.get_centers(5).tolist()
            print(len(centers))
            detector.create_shapes(centers)
   

        detector.step()

    # detector.move_pregrasp_unoccluded('Sphere_1')

    time.sleep(5)

    # cv2.imshow('asldf', detector.background_gray)
    # cv2.imshow('rgb', detector.background_rgb)
    # cv2.waitKey(0)
    # # time.sleep(6)
    # cv2.destroyAllWindows()    