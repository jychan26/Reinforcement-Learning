# Reinforcement-Learning
Explored the wumpus world using Q-Learning and SARSA algorithms. A comparison was done to understand which learning algorithm and exploration strategy is the best choice for this problem.

Note the following description is adapted from Artificial Intelligence: A Modern Approach by Russell and Norvig (section 7.2 on page 210 in the 4th edition), with a minor difference being that the three movement actions (Forward, TurnLeft, TurnRight) has no effect with a small probability κ.

## Wumpus World
The wumpus world is a cave consisting of rooms connected by passageways. Lurking somewhere in the cave is the terrible wumpus, a beast that eats anyone who enters its room. The agent can shoot the wumpus, but the agent has only one arrow. Some rooms contain bottomless pits that will trap the agent if the agent wonders into these rooms. The only redeeming feature of this bleak environment is the possibility of finding a heap of gold.

### Actions
The agent takes one of the following actions during each time step:
- Forward: This action is noisy with an error probability of κ ∈ [0, 1]. With a probability
of 1−κ, the agent moves to the next square in the direction the agent is facing. If the
agent bumps into a wall, it remains in the same square. With a probability of κ, the
action has no effect and the agent remains in the same square.
- TurnLeft: This action is noisy with an error probability of κ ∈ [0, 1]. With a probability
of 1−κ, the agent rotates 90 degrees to the left (counterclockwise) and remains
in the same square. With a probability of κ, the action has no effect and the agent
faces the same direction as before.
- TurnRight: This action is noisy with an error probability of κ ∈ [0, 1]. With a probability
of 1 − κ, the agent rotates 90 degrees to the right (clockwise) and remains in
the same square. With a probability of κ, the action has no effect and the agent faces
the same direction as before.
- Grab: The agent obtains the gold if the gold is in the same square as the agent.
Otherwise, nothing happens.
- Shoot: The agent fires an arrow in a straight line in the direction the agent is facing.
The arrow continues until it hits and kills a wumpus or hits a wall. The agent has one
arrow. If the agent has already used the arrow, this action has no effect.
- Climb: The agent climbs out of the cave if it is at the entrance. Otherwise, this action
has no effect. Once the agent climbs out of the cave, this episode of the wumpus world
journey ends.

### Rewards
The agent may obtain the following rewards.
- +1000 for climbing out of the cave with the gold.
- −1000 for falling into a pit.
- −1000 for being eaten by the wumpus.
- −1 for each action taken (even if the action has no effect).
- −10 for using up the arrow.

### Observations
In addition to knowing its current location and orientation, the agent may receive the following
observations:
- In the squares directly (not diagonally) adjacent to the wumpus, the agent will perceive
a stench.
- In the squares directly (not diagonally) adjacent to a pit, the agent will perceive a
breeze.
- In the square where the gold is, the agent will perceive a glitter.
- When the agent bumps into a wall, it will perceive a bump.
- When the wumpus is killed, it emits a woeful scream that can be perceived anywhere
in the cave.
