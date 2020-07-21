import sys
sys.path.append('../../build')
import libry as ry
import time 
import cv2
import copy
import numpy as np
from termcolor import colored
import random
from decorators import *


class Perceptor(object):

    def __init__(self, no_obj):
        self.no_obj = no_obj
        self._shape_dic = {'sphere': [ry.ST.sphere, 1], 'cylinder': [ry.ST.cylinder, 2], 'box': [ry.ST.box, 3]}
        self.steps_taken = 0
        self.sizes = np.arange(0.025, 0.1, 0.002)
        # self.sizes = np.arange(0.025, 0.06, 0.001)
        self.positions = np.arange(-0.3, 0.35, 0.05)
        self.locations = {'trash': [0.0, -1.3, 0.8], 'good': [0.55, 0.05, 0.65], 'maybe': [-0.55, 0.05, 0.65]}
        self.object_indices = np.arange(1, self.no_obj+1)
        self.objects_spawned = 0
        self.RealWorld = ry.Config()
        self.RealWorld.addFile("../../scenarios/challenge.g")
        self.Model = ry.Config()
        self.Model.addFile('../../scenarios/pandasTable_2.g')
        self.Model_Viewer = ry.ConfigurationViewer()
        self.Model_Viewer.setConfiguration(self.Model)
        self.camera_Frame = self.Model.frame('camera')
        self._reorder_objects()
        for _ in range(self.no_obj):
            # self.spawn_random_object()
            self.spawn_nice_object()
        self.Simulation = self.RealWorld.simulation(ry.SimulatorEngine.physx, True)
        self.Simulation.addSensor('camera')
        self._set_focal_length(0.895)
        self.tau = 0.01
        [self.background_rgb, self.background_depth] = self.Simulation.getImageAndDepth()
        self.background_gray = cv2.cvtColor(self.background_rgb, cv2.COLOR_BGR2GRAY)
        self.open_gripper()
        self.start_JV = self.Simulation.get_q()

        print('Init successful!')


    def __repr__(self):
        return f'Perceptor({self.no_obj} Objects)'


    def _reorder_objects(self):
        """
        Method for deleting the objects in 'challenge.g' and respawning desired objects
        """

        for i in range(0, 30):
            name = "obj%i" % i
            self.RealWorld.delFrame(name)
    
        # self.Model_Viewer.recopyMeshes(self.Model)
        # self.Model_Viewer.setConfiguration(self.Model)
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
        # print(f'min x value: {np.min(points[:,:,0])}')
        # print(f'max x value: {np.max(points[:,:,0])}')
        # print(f'min y value: {np.min(points[:,:,1])}')
        # print(f'max x value: {np.max(points[:,:,1])}')
        self.current_object_points = [j for i in points for q,j in enumerate(i) if j[0]!=0 and j[1]!=0 and j[2]!=0 \
                                       and j[2]>-2.0 and j[0] > -0.35 and j[0] < 0.35 and j[1] > -0.45 and j[1] < 0.1]
        self.camera_Frame.setPointCloud(points, rgb)
        self.Model_Viewer.recopyMeshes(self.Model)
        self.Model_Viewer.setConfiguration(self.Model)


    def step(self, no_steps=1):
        for _ in range(no_steps):
            self.Simulation.step([], self.tau, ry.ControlMode.none)
            time.sleep(self.tau)
            self.steps_taken += 1


    # @debug
    def spawn_cloud_objects(self):
        self.no_point_objects = 0
        self.point_object_list = []
        if len(self.current_object_points) > 150:
            # spawn_every = 1
            spawn_every = 2
        else:
            spawn_every = 1

        for i, position in enumerate(self.current_object_points):

            if i%spawn_every==0:
                obj_name = 'Point_object_{}'.format(i+1)
                point_object = self.Model.addFrame(obj_name)
                point_object.setShape(ry.ST.sphere, [.0005])
                # if i == 20:
                    # point_object.setColor([0, 0, 1])
                # point_object.setShape(ry.ST.sphere, [.002])
                self.Model.attach('camera', obj_name)
                point_object.setRelativePosition(position)
                point_object.setColor([0,0,1])
                self.no_point_objects += 1
                self.point_object_list.append(obj_name)
        self.Model_Viewer.recopyMeshes(self.Model)
        self.Model_Viewer.setConfiguration(self.Model)

        return {'Number of cloud objects spawned': self.no_point_objects}


    def delete_cloud_objects(self):
        for name in self.point_object_list :
            self.Model.delFrame(name)


    def optimize(self, obj_name):
        optimizer = self.Model.komo_path(1.,1,self.tau,True)
        optimizer.clearObjectives()
        optimizer.add_qControlObjective(order=1, scale=1e3)#
        optimizer.addSquaredQuaternionNorms(0., 1., 1e2)#


        optimizer.addObjective([1.], ry.FS.distance, [obj_name, "table"], ry.OT.eq, [1e4], target=[0.0001]);
        optimizer.addObjective([1.], ry.FS.quaternionDiff, [obj_name, 'world'], ry.OT.eq, [1e2]);

        for name in self.point_object_list:
            optimizer.addObjective([1.], ry.FS.distance, [obj_name, name], ry.OT.sos, [3e0], target=[-0.0005]);

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

    @timer               
    def find_best_fit(self, error_threshold=0.01):
        smallest_error = None
        
        # print(f'Finding best match among {len(self._shape_dic)} shapes, trying max. of {len(self.sizes)} sizes per shape dimension.')
        print('Analyzing point cloud data ...')
        # Shapes
        for shape in self._shape_dic.keys():
            # print(shape)

            size_combi = [0.01 for _ in range(self._shape_dic[shape][1])]
            # Dimensions
            for dim in range(self._shape_dic[shape][1]):
                # print('###############################################')
                # print(f'Starting to change dimension {dim}.')
                # print('###############################################')

                previous_error = 0
                # Sizes
                for size in self.sizes:
                    size = np.round(size, decimals=5)
                    if shape=='box':
                        size_combi[-(dim+1)] = size
                    else:
                        size_combi[dim] = size

                    # print(size_combi)

                    name = self.spawn_object(shape, size_combi)
                    opt = self.optimize(name)

                    current_error = opt.getCosts()

                    # print(f'Error difference: (current - previous_error): {current_error - previous_error:.4f}')
                    # print(f'Absolute error: {current_error:.6f}')


                    # First condition: Error starts increasing for the current shape and dimension.
                    # That means the current size is the best fit for the given shape and dimension. 
                    if current_error > previous_error + 0.01 and previous_error != 0:
                        # Second condition: the smallest error of the current shape is less than the smallest
                        # error of any shape that came before.
                        if not smallest_error or previous_error < smallest_error:
                            smallest_error = previous_error
                            print(f'New smallest error: {smallest_error}')
                            best_shape = previous_shape
                            best_size = previous_size
                            if shape=='box':
                                size_combi[-(dim+1)] = best_size[-(dim+1)]
                            else:
                                size_combi[dim] = best_size[dim]
                            # print(previous_size)
                            # print(f'New best size: {best_size}')
                            best_position = previous_position
                            # if dim==0:
                            self.Model.delFrame(name)
                            break

                        else:
                            if shape=='box':
                                size_combi[-(dim+1)] = previous_size[-(dim+1)]
                            else:
                                size_combi[dim] = previous_size[dim]
                            # size_combi[dim] = previous_size[dim]
                            self.Model.delFrame(name)
                            break

                    # print(f'Changing previous_error from {previous_error} to {current_error}.')
                    previous_error = copy.deepcopy(current_error)
                    previous_shape = copy.deepcopy(shape)
                    previous_size = copy.deepcopy(size_combi)
                    previous_position = self.Model.frame(name).getPosition()

                    self.Model.delFrame(name)


            if smallest_error is not None and smallest_error < error_threshold:
                print(f'Got error ({smallest_error}) below threshold ({error_threshold}), skipping remaining iterations.')
                break

        print(colored('################## OPTIMIZATION RESULTS ##################', color='green', attrs=['bold']))
        print(colored('Best shape match: {}'.format(best_shape), color='white', attrs=['bold']))
        print(colored('Best size match: {}'.format(best_size), color='white', attrs=['bold']))
        print(colored('Best position match: {}'.format(best_position), color='white', attrs=['bold']))
        print(colored('Error for best match: {}'.format(smallest_error), color='white', attrs=['bold']))
        print(colored('##########################################################', color='green', attrs=['bold']))
        

        name = self.spawn_object(best_shape, best_size, best_position, color=[0,1,0])

        return name, best_shape, best_size, smallest_error


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


    # @debug
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
        planer.addObjective([1.], ry.FS.positionDiff, ["R_gripperCenter", target_name], ry.OT.eq, [2e1], target=[0,0,0.1]);
        planer.addObjective([1.], ry.FS.vectorZ, ["R_gripperCenter"], ry.OT.eq, scale=[1e1], target=[0,0,1]);
        planer.addObjective([1.], ry.FS.vectorY, ["R_gripperCenter"], ry.OT.eq, scale=[1e1], target=[0,1,0]);
        planer.addObjective([1.], ry.FS.vectorX, ["R_gripperCenter"], ry.OT.eq, scale=[1e1], target=[1,0,0]);
        planer.addObjective([1.], ry.FS.scalarProductYX, ["R_gripperCenter", target_name], ry.OT.eq);
        planer.addObjective([1.], ry.FS.scalarProductYY, ["R_gripperCenter", target_name], ry.OT.eq);
        planer.addObjective([], ry.FS.accumulatedCollisions, type=ry.OT.ineq, scale=[1e1]);
        planer.addObjective([], ry.FS.qItself, ["R_finger1"], ry.OT.eq, [1e1], order=1);
        planer.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e1], order=1);
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
        planer.addObjective([1.], ry.FS.position, ["R_gripperCenter"], ry.OT.eq, [2e1], target=target);
        planer.addObjective([], ry.FS.quaternion, ["R_gripperCenter"], ry.OT.eq, scale=[1e1], order=1);
        planer.addObjective([], ry.FS.accumulatedCollisions, type=ry.OT.ineq, scale=[1e1]);
        planer.addObjective([], ry.FS.qItself, ["R_finger1"], ry.OT.eq, [1e1], order=1);
        planer.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e1], order=1);
        planer.optimize()
        for t in range(n_steps):
            self.Model.setFrameState(planer.getConfiguration(t))
            q = self.Model.getJointState()
            time.sleep(self.tau)
            self.Simulation.step(q, self.tau, ry.ControlMode.position)
            self.Model_Viewer.setConfiguration(self.Model)
            self.steps_taken += 1


    def move_up(self):
        n_steps = 20
        self.Model.setJointState(self.Simulation.get_q())
        planer = self.Model.komo_path(1., n_steps, n_steps*self.tau, True)
        target = self.Model.getFrame("R_gripperCenter").getPosition()
        target[-1] += 0.1
        planer.addObjective([1.], ry.FS.position, ["R_gripperCenter"], ry.OT.eq, [2e1], target=target);
        planer.addObjective([], ry.FS.quaternion, ["R_gripperCenter"], ry.OT.eq, scale=[1e1], order=1);
        planer.addObjective([], ry.FS.accumulatedCollisions, type=ry.OT.ineq, scale=[1e1]);
        planer.addObjective([], ry.FS.qItself, ["R_finger1"], ry.OT.eq, [1e1], order=1);
        planer.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e1], order=1);
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
        planer.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [2e1], target=self.start_JV);
        planer.addObjective([], ry.FS.accumulatedCollisions, type=ry.OT.ineq, scale=[1e1]);
        planer.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e1], order=1);
        planer.optimize()
        for t in range(n_steps):
            self.Model.setFrameState(planer.getConfiguration(t))
            q = self.Model.getJointState()
            self.Simulation.step(q, self.tau, ry.ControlMode.position)
            self.Model_Viewer.setConfiguration(self.Model)
            self.steps_taken += 1
            time.sleep(self.tau)


    def place_object(self, target):
        
        n_steps = 30
        self.Model.setJointState(self.Simulation.get_q())
        planer = self.Model.komo_path(1., n_steps, n_steps*self.tau, True)
        planer.addObjective([1.], ry.FS.position, ['R_gripperCenter'], type=ry.OT.eq, scale=[2e1], target=target);
        planer.addObjective([], ry.FS.accumulatedCollisions, type=ry.OT.ineq, scale=[1e1]);
        planer.addObjective([1.], ry.FS.vectorZ, ["R_gripperCenter"], ry.OT.eq, scale=[1e2], target=[0,0,1]);
        planer.addObjective([1.], ry.FS.vectorY, ["R_gripperCenter"], ry.OT.eq, scale=[1e2], target=[0,1,0]);
        planer.addObjective([1.], ry.FS.vectorX, ["R_gripperCenter"], ry.OT.eq, scale=[1e2], target=[1,0,0]);
        planer.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e1], order=1);
        # planer.addObjective([1.], ry.FS.scalarProductYX, ["R_gripperCenter", target_name], ry.OT.eq);
        # planer.addObjective([1.], ry.FS.scalarProductYY, ["R_gripperCenter", target_name], ry.OT.eq);
        planer.optimize()
        for t in range(n_steps):
            self.Model.setFrameState(planer.getConfiguration(t))
            q = self.Model.getJointState()
            self.Simulation.step(q, self.tau, ry.ControlMode.position)
            self.Model_Viewer.setConfiguration(self.Model)
            self.steps_taken += 1
            time.sleep(self.tau)


    def pick_and_place_object(self, target_name, target_shape, target_size, error):
        self.move_to_target_pre_grasp(target_name=target_name)
        self.move_down()
        grasp_success = self.grasp(shape=target_shape)
        self.move_to_start()
        target = self.analyse_object(target_shape=target_shape, target_size=target_size, error=error)
        if grasp_success:
            time.sleep(0.2)
            self.place_object(target=target)
            self.open_gripper()
            self.move_up()
            self.move_to_start()
            return True
        else: 
            return False

    # @debug
    def analyse_object(self, target_shape, target_size, error):
        result = []
        print(self.locations['good'][2])
        if self.locations['maybe'][2] > 1.7:
            self.locations['maybe'][1] -= 0.15
            self.locations['maybe'][2] = 0.65
        if self.locations['good'][2] > 0.85:
            self.locations['good'][1] -= 0.35
            self.locations['good'][2] = 0.65
        if error > 1:
            print('Not sure about this shape, throwing away to be safe.')
            result = copy.deepcopy(self.locations['trash'])
        else:
            if target_shape == 'sphere':
                print('Shape not suitable for stacking, throwing away.')
                result = copy.deepcopy(self.locations['trash'])
            else:
                if (target_shape=='box' and target_size[0] > 0.05 and target_size[1] > 0.05):
                    print('Nice object for stacking, putting it on the good stack.')
                    result = copy.deepcopy(self.locations['good']) + (target_size[2]/2) + 0.01
                    self.locations['good'][2] += target_size[2]
                elif (target_shape=='cylinder' and target_size[1] > 0.04):
                    print('Nice object for stacking, putting it on the good stack.')
                    result = copy.deepcopy(self.locations['good'])  + (target_size[0]/2) + 0.01
                    self.locations['good'][2] += target_size[0]
                elif (any([True if s<0.02 else False for s in target_size])):
                    print('Object too small for stacking, throwing away.')
                    result = copy.deepcopy(self.locations['trash'])
                else:
                    print('Medium quality object, putting it on the risky stack.')

                    if target_shape=='box':
                        result = copy.deepcopy(self.locations['maybe']) + (target_size[2]/2) + 0.01
                        self.locations['maybe'][2] += target_size[2]
                    else:
                        result = copy.deepcopy(self.locations['maybe']) + (target_size[0]/2) + 0.01
                        self.locations['maybe'][2] += target_size[0]

        return result


    # @debug
    def detect(self):
        # self.step(100)
        [rgb, depth] = self.Simulation.getImageAndDepth() 
        self.update_binary_mask(rgb)
        self.update_point_cloud(rgb, depth)
        self.step(10)
        try:
            assert len(self.current_object_points)>0, 'No valid point cloud objects detected!'
            self.spawn_cloud_objects()
            return True

        except Exception as e:
            print(e)
            return False

        finally:
            self.step(10)

    # @debug
    def detect_and_sort(self):
        detected = self.detect()
        if detected:
            name, shape, size, error = self.find_best_fit()
            self.step(10)
            self.delete_cloud_objects()
            success = self.pick_and_place_object(target_name=name, target_shape=shape, target_size=size, error=error)
            self.Model.delFrame(name)
            if success:
                return True
            else:
                return False
        else:
            return True


    def spawn_random_object(self):
        shapes = list(self._shape_dic.keys())
        shape = random.choice(shapes)
        size = []
        for i in range(self._shape_dic[shape][1]):
            size.append(random.choice(self.sizes))
        position = []
        position.append(3 + random.choice(self.positions))
        position.append(3 + random.choice(self.positions))
        position.append(1.5)

        self.objects_spawned += 1

        obj_name = 'Object_{}'.format(self.objects_spawned)
        spawn_object = self.RealWorld.addFrame(obj_name)
        # spawn_object.setShape(ry.ST.box, size)
        spawn_object.setShape(self._shape_dic[shape][0], size)
        spawn_object.setColor([1,0,0])
        # self.RealWorld.attach("world", obj_name)
        spawn_object.setPosition(position)
        spawn_object.setMass(1.1)
        spawn_object.setContact(1)
        self.Model_Viewer.recopyMeshes(self.Model)
        self.Model_Viewer.setConfiguration(self.Model)


    def spawn_nice_object(self):
        self.objects_spawned += 1
        obj_name = 'Object_{}'.format(self.objects_spawned)
        spawn_object = self.RealWorld.addFrame(obj_name)
        spawn_object.setShape(ry.ST.box, [0.09,0.09,0.03])
        # spawn_object.setShape(ry.ST.cylinder, [0.03,0.05])
        position = []
        position.append(3 + random.choice(self.positions))
        position.append(3 + random.choice(self.positions))
        position.append(1.5)
        spawn_object.setPosition(position)
        spawn_object.setColor([1,0,0])
        spawn_object.setMass(0.4)
        spawn_object.setContact(1)
        self.Model_Viewer.recopyMeshes(self.Model)
        self.Model_Viewer.setConfiguration(self.Model)



    def object_to_workspace(self):
        object_index = random.choice(self.object_indices)
        self.object_indices = np.delete(self.object_indices, np.where(self.object_indices == object_index))
        obj_name = 'Object_{}'.format(object_index)
        pos = []
        pos.append(np.random.uniform(low=-0.3, high=0.3))
        pos.append(np.random.uniform(low=0.0, high=0.3))
        pos.append(1.5)

        f = self.RealWorld.getFrame(obj_name)
        f.setPosition(pos)
        f.setQuaternion([1.0, 0.0, 0.0, 0.0])
        self.Simulation.setState(self.RealWorld.getFrameState())
        self.Simulation.step([], self.tau, ry.ControlMode.none)

        self.step(100)


if __name__ == "__main__":

    detector = Perceptor(60)

    while True:

        detector.object_to_workspace()
        # detector.detect()
        success = False
        while not success:
            success = detector.detect_and_sort()
