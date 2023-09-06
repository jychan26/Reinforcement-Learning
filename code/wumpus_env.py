# version 1.1

import os
import itertools
from typing import List, Dict

import gym
from gym import spaces
import numpy as np
from PIL import Image, ImageDraw

ACTION_TURN_RIGHT = 0
ACTION_TURN_LEFT = 1
ACTION_FORWARD = 2
ACTION_GRAB = 3
ACTION_CLIMB = 4
ACTION_SHOOT = 5
ACTIONS = [ACTION_TURN_RIGHT, ACTION_TURN_LEFT, ACTION_FORWARD, ACTION_GRAB, ACTION_CLIMB, ACTION_SHOOT]

DIRECTION_RIGHT = 'right'
DIRECTION_LEFT = 'left'

HEADING_NORTH, HEADING_EAST, HEADING_SOUTH, HEADING_WEST = [0, 1, 2, 3]

IMG_DIR = 'img/'
IMG_FILES = {'A': 'agent.PNG', 'W': 'wumpus.PNG', 'P': 'pit.PNG', 'G': 'gold.PNG', 'B': 'breeze.PNG', 'S': 'stench.PNG',
             'E': 'entrance.PNG'}

class WumpusWorld(gym.Env):
    '''
    This class simulates the Wumpus World environment from Russell and Norvig (as seen in pages 210-213, 404-407 in 4th
    edition). The class adheres to the OpenAI gym interface for reinforcement learning environments. The location of
    entities in the environment is an (x, y) tuple, where the bottom left corner is (1, 1) and the top right corner
    is (self.width, self.height).

    The state is an 8-tuple consisting of:
      - x location  {1, ..., width}
      - y location  {1, ..., height}
      - heading     {0: North, 1: West, 2: South, 3: East}
      - stench:     {0, 1}
      - breeze:     {0, 1}
      - glitter:    {0, 1}
      - bump:       {0, 1}
      - scream:     {0, 1}

    The possible actions are:
      - 'TurnRight': Rotate 90 degrees to the right with probability 1 - kappa
      - 'TurnLeft': Rotate 90 degrees to the left with probability 1 - kappa
      - 'Forward': Move forward one space with probability 1 - kappa
      - 'Grab': Grabs the gold if there is gold in the current cell. Does nothing if the gold is not present.
      - 'Climb': Climb out of the cave if the agent is at the entrance. Does nothing if agent isn't at the entrance.
      - 'Shoot': Shoot the arrow if it hasn't been used yet. Does nothing if the agent already shot their arrow.

    The reward is the sum of:
      - (-1) for each action taken
      - (1000) if the agent escapes the cave with the gold
      - (-1000) if the agent dies (i.e. being eaten by wumpus or falling into a pit)
      - (-10) if the agent shoots its arrow
    '''
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, width: int = 4, height: int = 4, entrance: List[int] = [1, 1], heading: int = HEADING_EAST,
                 wumpus: List[int] = [1, 3], pits: List[List[int]]=[[3, 3], [3, 1], [4, 4]],
                 gold: List[int] = [2, 3], max_steps: int=5000, kappa: float = 0.0):
        '''
        Initializes the Wumpus World environment and resets the state. The default arguments produce a Wumpus World that
        is consistent with the illustration in Figure 7.2 on page 211 of Russell & Norvig 4e.
        :param width: Number of cells in the east-west direction
        :param height: Number of cells in the north-south direction
        :param entrance: [x, y] location of the entrance of the cave
        :param heading: Initial direction the agent is facing, in {0, 1, 2, 3}
        :param wumpus: [x, y] location of the wumpus
        :param pits: [x, y] locations of the pits
        :param gold: [x, y] location of the gold
        :param max_steps: Maximum number of environment steps before episode times out
        :param kappa: error probability for the 'Forward', 'TurnLeft' and 'TurnRight' actions (in [0, 1])
        '''

        # Validate inputs
        assert width > 0 and height > 0, "Width and height must be positive integers."
        assert entrance[0] in [1, width] or entrance[1] in [1, height], "Entrance must be on the periphery of the grid."
        assert entrance not in [wumpus] + pits, "Entrance cannot be in the same location as the wumpus or a pit."
        assert heading in [HEADING_NORTH, HEADING_EAST, HEADING_SOUTH, HEADING_WEST], "Heading must be in [0, 1, 2, 3]."
        assert all([1 <= l[0] <= width and 1 <= l[1] <= height for l in pits]), "Pits must be on the grid."
        assert 1 <= gold[0] <= width and 1 <= gold[1] <= height, "Gold must be on the grid."
        assert gold not in pits and gold != wumpus, "Gold cannot be in the same location as a pit or the wumpus."
        assert 0. <= kappa <= 1., "Kappa must be in [0, 1]."

        self.width = width
        self.height = height
        self.entrance_location = entrance
        self.init_heading = heading
        self.wumpus_location = wumpus
        self.pit_locations = pits
        self.gold_location = gold
        self.max_steps = max_steps
        self.kappa = kappa

        self.has_gold = False
        self.has_arrow = True
        self.wumpus_alive = True
        self.agent_alive = True

        self.stench_locations = [l for l in self._get_neighbour_locs(self.wumpus_location) if l not in self.pit_locations]
        breeze_locations = []
        for pit in self.pit_locations:
            breeze_locations += [l for l in self._get_neighbour_locs(pit) if l not in [self.wumpus_location] + self.pit_locations]
        self.breeze_locations = [list(t) for t in {tuple(l) for l in breeze_locations}]

        self.actions = ACTIONS
        self.n_actions = len(self.actions)
        self.n_states = self.width * self.height * 4 * 2 * 2 * 2 * 2 * 2
        self.state_arr_to_id = {}

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.MultiDiscrete([self.width, self.height, 4, 2, 2, 2, 2, 2], dtype=np.int32)

        # Get a bijection for state IDs and their array representations
        state_combinations = list(itertools.product(*[list(range(k)) for k in self.observation_space.nvec]))
        for i in range(len(state_combinations)):
            state_combinations[i] = (state_combinations[i][0] + 1, state_combinations[i][1] + 1) + \
                state_combinations[i][2:8]  # (x, y) coordinates start at 1
        self.state_arr_to_id = {''.join(list(map(str, state_combinations[i]))): i for i in range(len(state_combinations))}
        self.state_id_to_arr = {i: state_combinations[i] for i in range(len(state_combinations))}

        self.img_dict = {k: Image.open(os.path.join(IMG_DIR, v)) for (k, v) in IMG_FILES.items()}
        self._reset()   # Reset the environment

    def reset(self) -> np.array:
        '''
        Resets the environment and returns the initial state.
        :return: the initial state
        '''
        self._reset()
        return self._get_state_id(self._state)

    def step(self, action: int) -> (int, float, bool, Dict):
        '''
        Applies the agent's action, causing the environment to transition to the next state. Calculates the reward and
        determines if the next state is terminal (i.e. the end of an episode). Also returns an empty dictionary, to
        fulfill return type requirement of a gym environment.
        :return: Next state ID, reward, whether the next state is terminal, and a dictionary containing the array
                 representation of the state
        '''

        # Check for invalid actions
        assert self.action_space.contains(action), "Invalid action."

        execute_as_intended = np.random.random() >= self.kappa
        bump = escaped = scream = arrow_shot = False
        self._t += 1

        # Execute action
        if action == ACTION_FORWARD and execute_as_intended:
            bump = self._move_agent(1)              # Move forward by 1 in current direction
        elif action == ACTION_TURN_RIGHT and execute_as_intended:
            self._rotate_agent(DIRECTION_RIGHT)     # Rotate right
        elif action == ACTION_TURN_LEFT and execute_as_intended:
            self._rotate_agent(DIRECTION_LEFT)      # Rotate left
        elif action == ACTION_GRAB:
            if execute_as_intended and self.location == self.gold_location:
                self.has_gold = True                # Pick up the gold
        elif action == ACTION_SHOOT:
            if execute_as_intended and self.has_arrow:
                scream = self._shoot_arrow()        # Shoot the arrow
                arrow_shot = True
        elif action == ACTION_CLIMB:
            if execute_as_intended and self.location == self.entrance_location:
                escaped = True                      # Agent safely climbs out of the cave

        # Determine next state
        next_state = self._determine_state(bump, scream)

        # Determine if state is terminal
        agent_killed = self._check_if_agent_killed()

        # Calculate reward
        reward = -1 + 1000 * (escaped and self.has_gold) - 1000 * agent_killed - 10 * arrow_shot

        terminal = escaped or not self.agent_alive or self._t == self.max_steps
        next_state_id = self._get_state_id(next_state)
        self._state = next_state    # Update current state of the environment
        return next_state_id, reward, terminal, {'arr': self._state}

    def render(self, mode='rgb_array'):
        '''
        Creates a visual representation of the environment's current state.
        :param mode: If set to "human", prints to the console. If set to "rgb_array", returns an image representation
                     of the state as a numpy array.
        '''
        grid_str = [['' for _ in range(self.width)] for _ in range(self.height)]
        grid_str[self.height - self.location[1]][self.location[0] - 1] += "A "
        for l in self.pit_locations:
            grid_str[self.height - l[1]][l[0] - 1] += "P "
        for l in self.breeze_locations:
            grid_str[self.height - l[1]][l[0] - 1] += "B "
        if self.wumpus_alive:
            grid_str[self.height - self.wumpus_location[1]][self.wumpus_location[0] - 1] += "W "
            for l in self.stench_locations:
                grid_str[self.height - l[1]][l[0] - 1] += "S "
        if not self.has_gold:
            grid_str[self.height - self.gold_location[1]][self.gold_location[0] - 1] += "G "
        grid_str[self.height - self.entrance_location[1]][self.entrance_location[0] - 1] += "E "

        if mode == 'human':
            self._print_env(grid_str)
        elif mode == 'rgb_array':
            rgb_img = self._get_env_img(grid_str)
            return rgb_img

    def seed(self, seed: int=None):
        '''
        Sets the random seed for this environment's random number generator.
        :param seed: Random seed for numpy
        '''
        if seed is not None:
            np.random.seed(seed)

    # HELPER FUNCTIONS

    def _reset(self) -> np.array:
        '''
        Resets the environment to its initial state.
        :return: the initial state
        '''

        self.location = self.entrance_location.copy()
        self.heading = self.init_heading
        self.has_gold = False
        self.has_arrow = True
        self.wumpus_alive = True
        self.agent_alive = True
        self._state = self._determine_state(False, False)
        self._t = 0

    def _get_state_id(self, state_arr: np.array):
        '''
        Return the integer state ID for state_arr
        :param state_arr: The numpy representation of a state
        :return: The state ID corresponding to state_arr
        '''
        return self.state_arr_to_id[''.join(list(map(str, state_arr.tolist())))]

    def _get_state_arr(self, state_id: int) -> np.array:
        '''
        Return the numpy array representation for state state_id
        :param state_id: The state ID
        :return: The state array corresponding to state_id
        '''
        return np.array(self.state_id_to_arr[state_id])

    def _move_agent(self, delta) -> bool:
        '''
        Moves the agent forward or backward by 1 square.
        :param delta (str): Amount of squares by which to move the agent forward (if positive) or backward (if negative)
        :return (bool): True if the agent walks into a wall
        '''

        bump = False
        if self.heading == HEADING_NORTH:
            self.location[1] += delta
        elif self.heading == HEADING_EAST:
            self.location[0] += delta
        elif self.heading == HEADING_SOUTH:
            self.location[1] -= delta
        else:
            self.location[0] -= delta

        if self.location[0] < 1:
            self.location[0] = 1
            bump = True
        elif self.location[0] > self.width:
            self.location[0] = self.width
            bump = True
        if self.location[1] < 1:
            self.location[1] = 1
            bump = True
        elif self.location[1] > self.height:
            self.location[1] = self.height
            bump = True
        return bump

    def _rotate_agent(self, direction: str):
        '''
        Rotates the agent left or right by 90 degrees.
        :param direction (str): Direction of rotation. Either "left" or "right".
        '''
        self.heading = (self.heading + (-1) ** (direction == "left")) % 4

    def _shoot_arrow(self):
        '''
        Shoots the agent's arrow if it hasn't been used yet. If the wumpus is in the arrow's path, the wumpus dies.
        :return: True if the wumpus was killed by the arrow
        '''
        if self.wumpus_alive:
            if (self.heading == HEADING_NORTH and self.location[0] == self.wumpus_location[0] and self.location[1] < self.wumpus_location[1]) or \
                    (self.heading == HEADING_EAST and self.location[1] == self.wumpus_location[1] and self.location[0] < self.wumpus_location[0]) or \
                    (self.heading == HEADING_SOUTH and self.location[0] == self.wumpus_location[0] and self.location[1] > self.wumpus_location[1]) or \
                    (self.heading == HEADING_WEST and self.location[1] == self.wumpus_location[1] and self.location[0] > self.wumpus_location[0]):
                self.wumpus_alive = False
                return True
        return False

    def _determine_state(self, bump: bool, scream: bool) -> np.array:
        '''
        Creates a numpy array representation of the current state.
        :param bump: True if the agent bumped into a wall this time step
        :param scream: True if the wumpus screamed this time step
        :return: The state
        '''
        breeze = self.location in self.breeze_locations
        stench = self.location in self.stench_locations and self.wumpus_alive
        glitter = self.location == self.gold_location and not self.has_gold
        state = list(map(int, [self.location[0], self.location[1], self.heading, stench, breeze, glitter, bump, scream]))
        return np.array(state)

    def _check_if_agent_killed(self) -> bool:
        '''
        Determine if the agent was killed this time step either by the wumpus or by falling into a pit.
        :return: True if the agent was killed this time step
        '''
        agent_killed = (self.wumpus_alive and self.location == self.wumpus_location) or (self.location in self.pit_locations)
        self.agent_alive = not agent_killed
        return agent_killed

    def _get_neighbour_locs(self, location) -> List[List[int]]:
        '''
        Given a location in the environment, determines a list of all neighbouring locations.
        :return: A list of all of location's neighbours
        '''
        neighbours = []
        if location[0] > 1:
            neighbours.append([location[0] - 1, location[1]])
        if location[1] > 1:
            neighbours.append([location[0], location[1] - 1])
        if location[0] < self.width:
            neighbours.append([location[0] + 1, location[1]])
        if location[1] < self.height:
            neighbours.append([location[0], location[1] + 1])
        return neighbours

    def _print_env(self, grid_str: List[List[str]]):
        '''
        Prints a string representation of the Wumpus environment's state
        :param grid_str: A list of lists representing the contents of each grid cell
        '''
        print("LEGEND: 'A' = agent, 'W' = wumpus, 'P' = pit, 'B' = breeze, 'S' = stench, 'G' = gold, 'E = entrance")
        print('  |' + '|'.join('---------' for _ in range(len(grid_str[0]))) + '|')
        for i in range(self.height):
            print('{} | '.format(self.height - i) + ' | '.join(x.ljust(7) for x in grid_str[i]) + ' |')
            print('  |' + '|'.join('---------' for _ in grid_str[i]) + '|')
        print('   ' + ' '.join('    {}    '.format(i + 1) for i in range(self.width)))

    def _get_env_img(self, grid_str: List[List[str]]) -> np.array:
        '''
        Creates a RGB image representing the Wumpus environment's state, similar to Figure 7.2 (page 211) in Russell &
        Norvig 4e.
        :param grid_str: A list of lists representing the contents of each grid cell
        '''
        cell_w = 60
        img_w = self.width * cell_w
        img_h = self.height * cell_w
        img = Image.new('RGBA', (img_w + 1, img_h + 1), '#f8e4dc')
        drawer = ImageDraw.Draw(img)

        # Draw icons for each entity in the environment on the grid
        for i in range(self.width):
            for j in range(self.height):
                if len(grid_str[j][i]) > 0:
                    x = i * cell_w
                    y = j * cell_w
                    icons = grid_str[j][i].strip().split(' ')
                    if 'A' in icons:
                        icons.remove('A')
                        draw_agent = True
                    else:
                        draw_agent = False
                    for k in range(len(icons)):
                        icon = self.img_dict[icons[k]]
                        icon_h = int(icon.size[1] / icon.size[0] * cell_w)
                        icon = icon.resize((cell_w, icon_h))
                        img.paste(icon, (x, y + int(k * cell_w / len(icons))))
                    if draw_agent and self.agent_alive:
                        if self.heading == HEADING_NORTH:
                            agent_img = self.img_dict['A'].rotate(90)
                        elif self.heading == HEADING_EAST:
                            agent_img = self.img_dict['A']
                        elif self.heading == HEADING_SOUTH:
                            agent_img = self.img_dict['A'].rotate(-90)
                        else:
                            agent_img = self.img_dict['A'].transpose(Image.FLIP_LEFT_RIGHT)
                        img.paste(agent_img, (x, y), agent_img)

        # Draw lines on the grid
        for i in range(self.width + 1):
            x = i * cell_w
            drawer.line([(x, 0), (x, img_h - 1)], fill='black', width=1)
        for i in range(self.height + 1):
            y = i * cell_w
            drawer.line([(0, y), (img_w - 1, y)], fill='black', width=1)
        return np.array(img.convert('RGB'))
