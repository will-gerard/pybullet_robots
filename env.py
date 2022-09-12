import pybullet as p
import pybullet_data as pd
import numpy as np
import time

from typing import List

class Robo_env():
    def __init__(self):
        self.RoboID = None
        self.sleep_t = 1. / 240  # decrease the value if it is too slow.
        self.maxVelocity = 10
        self.force = 100
        self.n_sim_steps = 1
        self.startPos = [0, 0, 0.5]
        self.startOri = [np.pi / 2, 0, 0]
        self.initial_action = []
        self.jointIds = []
        self.reset()

    def step(self, action):
        for j in range(12):
            targetPos = float(action[j])
            #print(targetPos)
            targetPos = self.jointDirections[j] * targetPos + self.jointOffsets[j]
            p.setJointMotorControl2(bodyIndex=self.RoboID,
                                    jointIndex=self.jointIds[j],
                                    targetPosition=targetPos,
                                    controlMode=p.POSITION_CONTROL,
                                    force=self.force,
                                    maxVelocity=self.maxVelocity)
        for _ in range(self.n_sim_steps):
            p.stepSimulation()
            time.sleep(1. / 500)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        planeId = p.loadURDF("plane.urdf")  # URDF Id = 0
        self.RoboID = p.loadURDF("laikago/laikago.urdf", self.startPos,
                                 p.getQuaternionFromEuler(self.startOri))  # URDF Id = 1

        self.jointOffsets = []
        self.jointDirections = [-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]

        for i in range(4):
            self.jointOffsets.append(0)
            self.jointOffsets.append(-0.7)
            self.jointOffsets.append(0.7)


        for j in range(p.getNumJoints(self.RoboID)):
            p.changeDynamics(self.RoboID, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.RoboID, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                self.jointIds.append(j)

        with open(pd.getDataPath() + "/laikago/data-test.txt", "r") as filestream:
            for line in filestream:
                currentline = line.split(",")
                joints = currentline[2:14]
                self.step(joints)
                self.initial_action = joints
    
    def push_off_ground_with_leg(self, leg_index: int, curr_joint_positions: List[float]) -> List[float]:
        print(f"push with leg {leg_index}")
        shoulder_joint_index = leg_index*3 + 1
        next_joint_positions = [x for x in curr_joint_positions]
        while float(next_joint_positions[shoulder_joint_index]) < 0.6:
            next_joint_positions[shoulder_joint_index] = str(float(next_joint_positions[shoulder_joint_index]) + 0.1)
            self.step(action=next_joint_positions)
        return next_joint_positions 

    def lift_leg(self, leg_index: int, curr_joint_positions: List[float]) -> List[float]:
        print(f"lift leg {leg_index}")
        elbow_joint_index = leg_index*3 + 2
        next_joint_positions = [x for x in curr_joint_positions]
        while float(next_joint_positions[elbow_joint_index]) > -2:
            next_joint_positions[elbow_joint_index] = str(float(next_joint_positions[elbow_joint_index]) - 0.1)
            self.step(action=next_joint_positions)

        return next_joint_positions 

    def move_leg_forward(self, leg_index: int, curr_joint_positions: List[float]) -> List[float]:
        print(f"move forward leg {leg_index}")
        shoulder_joint_index = leg_index*3 + 1
        next_joint_positions = [x for x in curr_joint_positions]
        while float(next_joint_positions[shoulder_joint_index]) > 0.2:
            next_joint_positions[shoulder_joint_index] = str(float(next_joint_positions[shoulder_joint_index]) - 0.1)
            self.step(action=next_joint_positions)
        return next_joint_positions

    def drop_leg(self, leg_index: int, curr_joint_positions: List[float]) -> List[float]:
        print(f"drop leg {leg_index}")
        elbow_joint_index = leg_index*3 + 2
        next_joint_positions = [x for x in curr_joint_positions]
        while float(next_joint_positions[elbow_joint_index]) < -1.2:
            next_joint_positions[elbow_joint_index] = str(float(next_joint_positions[elbow_joint_index]) + 0.1)
            self.step(action=next_joint_positions)
        return next_joint_positions

def step_right(env: Robo_env, starting_joint_positions: List[float]):
    """
    Step with the front right and back left legs.

    The step occurs in three separate movements:
        1. lift leg
        2. move leg forward
        3. drop leg

    Each of the three steps happens first with the front right leg, then with the back left leg.
    In between each of these six total movements, we push off the ground with the other two legs,
    propelling the robot forward.
    """
    desired_joint_positions = env.lift_leg(leg_index = 0, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 1, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 2, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)

    desired_joint_positions = env.lift_leg(leg_index = 3, curr_joint_positions=desired_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 1, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 2, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)

    desired_joint_positions = env.move_leg_forward(leg_index=0, curr_joint_positions=desired_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 1, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 2, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)

    desired_joint_positions = env.move_leg_forward(leg_index=3, curr_joint_positions=desired_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 1, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 2, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)


    desired_joint_positions = env.drop_leg(leg_index=0, curr_joint_positions=desired_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 1, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 2, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)

    desired_joint_positions = env.drop_leg(leg_index=3, curr_joint_positions=desired_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 1, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 2, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)

    return desired_joint_positions

def step_left(env: Robo_env, starting_joint_positions: List[float]):
    """
    Step with the front left and back right legs.
    Same as the step_right function, but with the opposite legs.
    TODO: Consolidate these two functions into a single one, and create helper functions to remove some of the repeated code.
    """
    desired_joint_positions = env.lift_leg(leg_index = 1, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 0, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 3, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)

    desired_joint_positions = env.lift_leg(leg_index = 2, curr_joint_positions=desired_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 0, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 3, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)

    desired_joint_positions = env.move_leg_forward(leg_index=1, curr_joint_positions=desired_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 0, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 3, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)

    desired_joint_positions = env.move_leg_forward(leg_index=2, curr_joint_positions=desired_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 0, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 3, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)

    desired_joint_positions = env.drop_leg(leg_index=1, curr_joint_positions=desired_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 0, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 3, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)

    desired_joint_positions = env.drop_leg(leg_index=2, curr_joint_positions=desired_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 0, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)
    desired_joint_positions = env.push_off_ground_with_leg(leg_index = 3, curr_joint_positions=starting_joint_positions)
    env.step(desired_joint_positions)

    return desired_joint_positions

if __name__ == '__main__':
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pd.getDataPath())  # optionally
    env1 = Robo_env()
    c = 0
    action_list = env1.initial_action
    while 1:
        # action_list = [str(float(x)+.1) if ind == 11 else x for ind, x in enumerate(action_list)]
        # env1.step(action_list)
        c += 1
        action_list = step_right(env=env1, starting_joint_positions=action_list)
        action_list = step_left(env=env1, starting_joint_positions=action_list)

        # action_list = env1.lift_leg(leg_index=3, curr_joint_positions=action_list)
        # env1.step(action_list)
        # action_list = env1.move_leg_forward(leg_index=3, curr_joint_positions=action_list)
        # env1.step(action_list)
        # action_list = env1.drop_leg(leg_index=3, curr_joint_positions=action_list)
        # env1.step(action_list)
        # action_list = env1.push_off_ground_with_leg(leg_index=3, curr_joint_positions=action_list)
        # env1.step(action_list)
        # while float(action_list[11]) > -2.5:
        #     action_list = env1.lift_leg(leg_index=3, curr_joint_positions=action_list)
        #     env1.step(action_list)
        # while float(action_list[11]) < -1.0:
        #     action_list = env1.drop_leg(leg_index=3, curr_joint_positions=action_list)
        #     env1.step(action_list)
