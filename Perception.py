import sys
sys.path.append('../../build')
import libry as ry
import time 
import cv2
import numpy as np
from sklearn.cluster import KMeans
from termcolor import colored
import random


class Perceptor(object):
    def __init__(self, no_obj):

        # self._shape_dic = {'sphere': [ry.ST.sphere, 1], 'box': [ry.ST.box, 3]}
        self._shape_dic = {'sphere': [ry.ST.sphere, 1], 'box': [ry.ST.box, 3], 'capsule': [ry.ST.capsule, 3], 'cylinder': [ry.ST.cylinder, 3]}
        self.steps_taken = 0
        self.sizes = np.arange(0.01, 0.05, 0.002)
        self.positions = np.arange(-0.5, 0.55, 0.05)
        self.objects_spawned = 0
        self.RealWorld = ry.Config()
        self.RealWorld.addFile("../../scenarios/challenge.g")
        self.Model = ry.Config()
        self.Model.addFile('../../scenarios/pandasTable.g')
        self.Model_Viewer = ry.ConfigurationViewer()
        self.Model_Viewer.setConfiguration(self.Model)
        self.camera_Frame = self.Model.frame('camera')
        self.no_obj = no_obj
        self._reorder_objects()
        # for _ in range(self.no_obj):
            # self.spawn_random_object()
        self.Simulation = self.RealWorld.simulation(ry.SimulatorEngine.physx, True)
        self.Simulation.addSensor('camera')
        self._set_focal_length(0.895)
        self.tau = 0.01
        [self.background_rgb, self.background_depth] = self.Simulation.getImageAndDepth()
        self.background_gray = cv2.cvtColor(self.background_rgb, cv2.COLOR_BGR2GRAY)
      
        self.open_gripper()
        self.start_JV = self.Simulation.get_q()
        
        print('Init successful!')


    def _reorder_objects(self):
        """
        Method for deleting the objects in 'challenge.g' and respawning desired objects
        """

        for i in range(0, 30):
            name = "obj%i" % i
            self.RealWorld.delFrame(name)
        for i in range(1, self.no_obj+1):
            obj_name = 'Sphere_{}'.format(i)
            spawn_object = self.RealWorld.addFrame(obj_name)
            spawn_object.setShape(ry.ST.sphere, [0.04])
        #     spawn_object.setShape(ry.ST.box, [0.04, 0.04, 0.04])
            spawn_object.setColor([1,0,0])
            spawn_object.setPosition([0., .3*i, 2.])
            spawn_object.setMass(1.1)
            spawn_object.setContact(1)

        self.Model_Viewer.recopyMeshes(self.Model)
        self.Model_Viewer.setConfiguration(self.Model)
        # print(self.RealWorld.getFrameNames())


    def _set_focal_length(self, f):
        self.fxfypxpy = [f*360.0, f*360.0, 320., 180.]

    def update_binary_mask(self, image):
        """
        Checks which image pixels are different than the background image recorded at the start.
        Writes resulting mask into a class attribute, so it may more easily be used by other methods.

        Args:
            image: Image to analyse.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        background_diff = abs(self.background_gray - gray)
        _, self.bin_mask = cv2.threshold(background_diff, 0, 255, cv2.THRESH_BINARY)

    def update_point_cloud(self, rgb, depth):
        """
        Creates a point cloud in the model configuration based on received rgb and depth values.
        Filters the points for non-zero coordinates and saves them to a class attribute, so it
        may more easily be used by other methods.

        Args:
            rgb: An rgb image.
            depth: The corresponding pixel-wise depth values.
        """

        depth_mask = cv2.bitwise_and(depth, depth, mask=self.bin_mask)
        points = self.Simulation.depthData2pointCloud(depth_mask, self.fxfypxpy)
        self.current_object_points = [j for i in points for q,j in enumerate(i) if j[0]!=0 and j[1]!=0 and j[2]!=0]
        # self.current_object_points = [j for i in points for q,j in enumerate(i) if j[0]!=0 and j[1]!=0 and j[2]!=0 and q%2==0]
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


    def spawn_cloud_objects(self):
        self.no_point_objects = 0
        for i, position in enumerate(self.current_object_points):
                obj_name = 'Point_object_{}'.format(i+1)
                point_object = self.Model.addFrame(obj_name)
                point_object.setShape(ry.ST.sphere, [.0005])
                # if i == 20:
                    # point_object.setColor([0, 0, 1])
                # point_object.setShape(ry.ST.sphere, [.002])
                self.Model.attach('camera', obj_name)
                point_object.setRelativePosition(position)
                self.no_point_objects += 1
        self.Model_Viewer.recopyMeshes(self.Model)
        self.Model_Viewer.setConfiguration(self.Model)


    def delete_cloud_objects(self):
        for i in range(self.no_point_objects):
            obj_name = 'Point_object_{}'.format(i+1)
            self.Model.delFrame(obj_name)


    def optimize(self, obj_name):
        optimizer = self.Model.komo_path(1.,1,self.tau,True)
        optimizer.clearObjectives()
        optimizer.add_qControlObjective(order=1, scale=1e3)
        optimizer.addSquaredQuaternionNorms(0., 1., 1e2)

        # optimizer.addObjective([], ry.FS.vectorY, [obj_name], ry.OT.eq, [1e2], order=1)

        # optimizer.addObjective([1.], ry.FS.distance, [obj_name, "table"], ry.OT.sos, [1e2])
        optimizer.addObjective([1.], ry.FS.distance, [obj_name, "table"], ry.OT.eq, [1e2], target=[0.0001])
        optimizer.addObjective([1.], ry.FS.quaternionDiff, [obj_name, 'world'], ry.OT.eq, [1e2])
        # optimizer.addObjective([1.], ry.FS.distance, [obj_name, "Point_object_21"], ry.OT.eq, [1e4])

        # for i in range(99):
        for i in range(self.no_point_objects):
            point_name = 'Point_object_{}'.format(i+1)
            # optimizer.addObjective([1.], ry.FS.distance, [obj_name, point_name], ry.OT.sos, [1e1 - (i*10/self.no_point_objects)])
            optimizer.addObjective([1.], ry.FS.distance, [obj_name, point_name], ry.OT.sos, [1e0], target=[-0.0005])
            # optimizer.addObjective([1.], ry.FS.distance, [obj_name, point_name], ry.OT.sos, [1e1 - i])

        optimizer.optimize()

        self.Model.setFrameState(optimizer.getConfiguration(0))
        self.Model_Viewer.setConfiguration(self.Model)

        return optimizer


    def spawn_object(self, shape, size, position=[0,0.2,1.5], color=[1,0,0]):
        object_name = shape + str(size)
        self.new_object = self.Model.addFrame(object_name) 
        self.Model.makeObjectsFree([object_name])
        if type(size) is not list:
            size_list = []
            size_list.append(size)
        else:
            size_list = size
        self.new_object.setShape(self._shape_dic[shape][0], size_list)
        self.new_object.setColor(color)
        self.new_object.setPosition(position)
        self.new_object.setContact(1)
        self.Model_Viewer.recopyMeshes(self.Model)
        self.Model_Viewer.setConfiguration(self.Model)

        return object_name
        # center = self.get_centers(1).tolist()
        # self.test_object.setPosition(center[0])

                           
    def find_best_fit(self):
        smallest_error = None
        
        print(f'Finding best match among {len(self._shape_dic)} shapes, trying {len(self.sizes)} sizes.')
        for shape in self._shape_dic.keys():
            previous_error = 0
            for size in self.sizes:
                if shape in ['box', 'capsule', 'cylinder']:
                    size = [size, size, size]
                name = self.spawn_object(shape, size)
                opt = self.optimize(name)
                # time.sleep(0.1)  

                current_error = opt.getCosts()

                # First condition: Error starts increasing for the current shape.
                # That means the current size is the best fit for the given shape.
                if current_error > previous_error and previous_error != 0:
                    # Second condition: the smallest error of the current shape is less than the smallest
                    # error of any shape that came before.
                    if not smallest_error or previous_error < smallest_error:
                        smallest_error = previous_error
                        best_shape = previous_shape
                        best_size = previous_size
                        best_position = previous_position
                        self.Model.delFrame(name)
                        break

                previous_error = current_error
                previous_shape = shape
                previous_size = size
                previous_position = self.Model.frame(name).getPosition()

                self.Model.delFrame(name)

        print(colored('################## OPTIMIZATION RESULTS ##################', color='green', attrs=['bold']))
        print(colored('Best shape match: {}'.format(best_shape), color='white', attrs=['bold']))
        print(colored('Best size match: {}'.format(best_size), color='white', attrs=['bold']))
        print(colored('Best position match: {}'.format(best_position), color='white', attrs=['bold']))
        print(colored('Error for best match: {}'.format(smallest_error), color='white', attrs=['bold']))
        print(colored('##########################################################', color='green', attrs=['bold']))
        

        name = self.spawn_object(best_shape, best_size, best_position, color=[0,1,0])

        return name, best_shape
        # return best_shape, best_size, best_position


    def open_gripper(self):
        self.Simulation.openGripper("R_gripper")
        # self.Model.attach("world", "object")
        for i in range(50):
            time.sleep(self.tau)
            self.Simulation.step([], self.tau, ry.ControlMode.none)
            self.Model.setJointState(self.Simulation.get_q())
            self.Model_Viewer.setConfiguration(self.Model)
            self.gripper_open = True
            self.steps_taken += 1

    def grasp(self, shape):
        self.Simulation.closeGripper("R_gripper")
        print('Attempting to grasp object...')
        while True:
            time.sleep(self.tau)
            self.Simulation.step([], self.tau, ry.ControlMode.none)
            self.Model.setJointState(self.Simulation.get_q())
            self.Model_Viewer.setConfiguration(self.Model)
            self.steps_taken += 1
            if self.Simulation.getGripperIsGrasping("R_gripper"): 
                # self.Model.attach("R_gripper", "object")
                print(colored(f'Grasped a {shape}!', color='yellow', attrs=['blink', 'bold']))
                return True
            if self.Simulation.getGripperWidth("R_gripper") < -0.05 and self.gripper_open: 
                print('Gripper closed, nothing grasped.')
                self.gripper_open = False
                self.open_gripper()
                return False

    def move_to_target_pre_grasp(self, target_name, **kwargs):
        n_steps = 50
        self.Model.attach("world", target_name)

        self.Model.setJointState(self.Simulation.get_q())
        planer = self.Model.komo_path(1., n_steps, n_steps*self.tau, True)
        planer.addObjective([1.], ry.FS.positionDiff, ["R_gripperCenter", target_name], ry.OT.eq, [2e1], target=[0,0,0.1])
        planer.addObjective([1.], ry.FS.vectorZ, ["R_gripperCenter"], ry.OT.eq, scale=[1e1], target=[0,0,1]);
        planer.addObjective([1.], ry.FS.scalarProductYX, ["R_gripperCenter", target_name], ry.OT.eq);
        planer.addObjective([1.], ry.FS.scalarProductYY, ["R_gripperCenter", target_name], ry.OT.eq);
        planer.addObjective([], ry.FS.accumulatedCollisions, type=ry.OT.ineq, scale=[1e1])
        planer.addObjective([], ry.FS.qItself, ["R_finger1"], ry.OT.eq, [1e1], order=1)
        planer.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e1], order=1)
        planer.optimize()
        for t in range(n_steps):
            self.Model.setFrameState(planer.getConfiguration(t))
            q = self.Model.getJointState()
            time.sleep(self.tau)
            self.Simulation.step(q, self.tau, ry.ControlMode.position)
            self.Model_Viewer.setConfiguration(self.Model)
            self.steps_taken += 1


    def move_down(self):
        n_steps = 20
        self.Model.setJointState(self.Simulation.get_q())
        planer = self.Model.komo_path(1., n_steps, n_steps*self.tau, True)
        target = self.Model.getFrame("R_gripperCenter").getPosition()
        target[-1] -= 0.1
        planer.addObjective([1.], ry.FS.position, ["R_gripperCenter"], ry.OT.eq, [2e1], target=target)
        planer.addObjective([], ry.FS.quaternion, ["R_gripperCenter"], ry.OT.eq, scale=[1e1], order=1)
        planer.addObjective([], ry.FS.accumulatedCollisions, type=ry.OT.ineq, scale=[1e1])
        planer.addObjective([], ry.FS.qItself, ["R_finger1"], ry.OT.eq, [1e1], order=1)
        planer.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e1], order=1)
        planer.optimize()
        for t in range(n_steps):
            self.Model.setFrameState(planer.getConfiguration(t))
            q = self.Model.getJointState()
            time.sleep(self.tau)
            self.Simulation.step(q, self.tau, ry.ControlMode.position)
            self.Model_Viewer.setConfiguration(self.Model)
            self.steps_taken += 1


    def move_to_start(self):
        n_steps = 30
        self.Model.setJointState(self.Simulation.get_q())
        planer = self.Model.komo_path(1., n_steps, n_steps*self.tau, True)
        planer.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [2e1], target=self.start_JV)
        planer.addObjective([], ry.FS.accumulatedCollisions, type=ry.OT.ineq, scale=[1e1])
        planer.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e1], order=1)
        planer.optimize()
        for t in range(n_steps):
            self.Model.setFrameState(planer.getConfiguration(t))
            q = self.Model.getJointState()
            self.Simulation.step(q, self.tau, ry.ControlMode.position)
            self.Model_Viewer.setConfiguration(self.Model)
            self.steps_taken += 1
            time.sleep(self.tau)

    def place_object(self, target_shape):
        if target_shape == 'sphere':
            target = [0.5, -1.3, 0.8]
        else:
            target = [0.6, 0.1, 0.8]
        n_steps = 30
        self.Model.setJointState(self.Simulation.get_q())
        planer = self.Model.komo_path(1., n_steps, n_steps*self.tau, True)
        planer.addObjective([1.], ry.FS.position, ['R_gripperCenter'], type=ry.OT.eq, scale=[2e1], target=target)
        planer.addObjective([], ry.FS.accumulatedCollisions, type=ry.OT.ineq, scale=[1e1])
        planer.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e1], order=1)
        planer.optimize()
        for t in range(n_steps):
            self.Model.setFrameState(planer.getConfiguration(t))
            q = self.Model.getJointState()
            self.Simulation.step(q, self.tau, ry.ControlMode.position)
            self.Model_Viewer.setConfiguration(self.Model)
            self.steps_taken += 1
            time.sleep(self.tau)


    def pick_and_place_object(self, target_name, target_shape):
        self.move_to_target_pre_grasp(target_name=target_name)
        self.move_down()
        self.grasp(shape=target_shape)
        self.move_to_start()
        time.sleep(0.5)
        self.place_object(target_shape=target_shape)
        self.open_gripper()
        self.move_to_start()


    def spawn_random_object(self):
        shapes = list(self._shape_dic.keys())
        shape = random.choice(shapes)
        size = []
        for i in range(3):
        # for i in range(self._shape_dic[shape][1]):
            size.append(random.choice(self.sizes))
        position = []
        position.append(random.choice(self.positions))
        position.append(random.choice(self.positions))
        position.append(1.5)

        self.objects_spawned += 1

        obj_name = 'Object_{}'.format(self.objects_spawned)
        spawn_object = self.RealWorld.addFrame(obj_name)
        spawn_object.setShape(ry.ST.box, size)
        # spawn_object.setShape(self._shape_dic[shape][0], size)
        spawn_object.setColor([1,0,0])
        spawn_object.setPosition(position)
        spawn_object.setMass(1.1)
        spawn_object.setContact(1)
        self.Model_Viewer.recopyMeshes(self.Model)
        self.Model_Viewer.setConfiguration(self.Model)

if __name__ == "__main__":
    detector = Perceptor(1)

    for t in range(2000):

        if t%10 == 0:
            [rgb, depth] = detector.Simulation.getImageAndDepth() 
            detector.update_binary_mask(rgb)
            detector.update_point_cloud(rgb, depth)

        if t==150:
            detector.spawn_cloud_objects()

        if t==480:
            name, shape = detector.find_best_fit()

        # if t==200:
            # detector.delete_cloud_objects()

        # if t==220:
            # detector.pick_and_place_object(target_name=name, target_shape=shape)

        detector.step()

    