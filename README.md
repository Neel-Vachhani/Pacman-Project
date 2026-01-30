# Pacman AI Project

Implementation of artificial intelligence algorithms applied to the classic Pacman game, demonstrating search strategies, multi-agent adversarial systems, probabilistic inference, and reinforcement learning techniques.

## Overview

This project implements various AI techniques through the UC Berkeley Pacman Project framework. The work is organized into four major components: search algorithms, adversarial multi-agent systems, probabilistic tracking, and reinforcement learning. Each component demonstrates different aspects of artificial intelligence and their practical applications in game-playing agents.

**Attribution**: This project is based on the [UC Berkeley CS188 Intro to AI Pacman Projects](http://ai.berkeley.edu), developed by John DeNero and Dan Klein.

## Tech Stack

- **Language**: Python 3
- **Data Structures**: Queues, Stacks, Priority Queues, Counters
- **AI Techniques**: 
  - Search Algorithms (DFS, BFS, UCS, A*)
  - Adversarial Search (Minimax, Alpha-Beta Pruning, Expectimax)
  - Probabilistic Inference (Exact Inference, Particle Filtering)
  - Reinforcement Learning (Q-Learning, Value Iteration, Approximate Q-Learning)

## Project Structure

The project is organized into four main directories, each representing a distinct AI component:

### 1. Search (`/search`)

Implementation of foundational search algorithms for pathfinding and problem-solving.

**Algorithms Implemented:**
- **Depth-First Search (DFS)**: Explores deepest nodes first using a stack-based frontier
- **Breadth-First Search (BFS)**: Explores shallowest nodes first using a queue-based frontier
- **Uniform Cost Search (UCS)**: Cost-optimal search using a priority queue ordered by path cost
- **A* Search**: Heuristic-guided search combining path cost and heuristic estimates

**Key Features:**
- Generic graph search implementation with state tracking
- Multiple search problems:
  - Position Search: Navigate to specific locations
  - Corners Problem: Visit all four corners of the maze
  - Food Search: Collect all food pellets efficiently
- Custom heuristics:
  - Manhattan Distance heuristic for position problems
  - Corners heuristic for multi-goal navigation
  - Food heuristic for efficient pellet collection

**Core Files:**
- `search.py`: Search algorithm implementations
- `searchAgents.py`: Problem formulations and heuristics

### 2. Multi-Agent Search (`/multiagent`)

Adversarial search algorithms for handling multiple agents with competing objectives.

**Algorithms Implemented:**
- **Minimax**: Optimal decision-making assuming perfect adversarial play
  - Recursive tree exploration
  - Alternating maximizer (Pacman) and minimizer (Ghosts) layers
  - Depth-limited search with evaluation functions
  
- **Alpha-Beta Pruning**: Optimized minimax with branch pruning
  - Maintains alpha (best MAX value) and beta (best MIN value)
  - Prunes subtrees that cannot affect final decision
  - Significantly reduces search space
  
- **Expectimax**: Handles probabilistic/suboptimal opponents
  - Replaces MIN nodes with expected value calculations
  - Models random ghost behavior
  - Better suited for non-adversarial scenarios

**Evaluation Functions:**
- Custom evaluation function considering:
  - Current game score
  - Food proximity (inverse distance to closest food)
  - Ghost distances (threat assessment)
  - Scared ghost opportunities
  - Capsule availability

**Core Files:**
- `multiAgents.py`: Multi-agent search implementations
- Reflexed agent for baseline comparison

### 3. Ghostbusters - Probabilistic Tracking (`/tracking`)

Bayesian inference for tracking invisible ghosts using noisy distance sensors.

**Inference Methods:**
- **Exact Inference**: 
  - Maintains exact belief distribution over all possible ghost positions
  - Time elapse: Propagates beliefs through transition model
  - Observation update: Uses Bayes' rule with sensor model
  - Handles discrete probability distributions
  
- **Particle Filtering**:
  - Approximate inference using weighted samples
  - Particle initialization across legal positions
  - Importance sampling for observation updates
  - Resampling when all weights collapse to zero
  - Scalable to larger state spaces

**Probabilistic Model:**
- **Sensor Model**: Noisy Manhattan distance measurements
- **Transition Model**: Ghost movement probabilities based on legal actions
- **Observation Updates**: Bayesian belief propagation
- **Joint Particle Filtering**: Tracks multiple ghosts simultaneously

**Key Features:**
- Discrete distribution normalization
- Belief visualization overlay
- Joint inference for multi-ghost scenarios
- Greedy hunting strategy based on belief distributions

**Core Files:**
- `inference.py`: Exact and particle filtering implementations
- `bustersAgents.py`: Greedy agents using inference
- `busters.py`: Ghostbusters game rules

### 4. Reinforcement Learning (`/reinforcement`)

Model-free learning agents that learn optimal policies through experience.

**Algorithms Implemented:**

**Value Iteration** (Model-Based):
- Computes optimal value function through dynamic programming
- Bellman update: V(s) = max_a Σ T(s,a,s')[R(s,a,s') + γV(s')]
- Iteratively updates values until convergence
- Extracts optimal policy from converged values

**Q-Learning** (Model-Free):
- Learns state-action values (Q-values) from experience
- Update rule: Q(s,a) ← (1-α)Q(s,a) + α[R + γ max_a' Q(s',a')]
- Epsilon-greedy exploration strategy
- Temporal difference learning

**Approximate Q-Learning**:
- Uses feature-based representation for generalization
- Linear function approximation: Q(s,a) = Σ f_i(s,a) * w_i
- Feature extractors for Pacman domain
- Weight updates instead of explicit Q-value storage
- Scales to large state spaces

**Domains:**
- **Gridworld**: Simple MDP for testing value iteration
- **Crawler Robot**: Continuous control problem
- **Pacman**: Full game with learned policies

**Key Concepts:**
- Exploration vs. exploitation trade-off
- Learning rate (α) and discount factor (γ) tuning
- Feature engineering for function approximation
- Policy extraction from Q-values

**Core Files:**
- `valueIterationAgents.py`: Value iteration implementation
- `qlearningAgents.py`: Q-learning and approximate Q-learning
- `analysis.py`: Parameter analysis questions
- `featureExtractors.py`: Feature functions for approximation

## Installation and Usage

### Prerequisites
```bash
python 3.x
tkinter (for graphics)
```

### Running Search Algorithms

```bash
# Navigate to search directory
cd search

# Run DFS on medium maze
python pacman.py -l mediumMaze -p SearchAgent -a fn=dfs

# Run BFS on big maze
python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z 0.5

# Run A* with manhattan heuristic
python pacman.py -l bigMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic -z 0.5

# Solve corners problem with BFS
python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem

# Food search with A* and custom heuristic
python pacman.py -l trickySearch -p AStarFoodSearchAgent
```

### Running Multi-Agent Search

```bash
# Navigate to multiagent directory
cd multiagent

# Run Minimax agent
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4

# Run Alpha-Beta agent
python pacman.py -p AlphaBetaAgent -l smallClassic -a depth=3

# Run Expectimax agent
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3

# Test evaluation function
python pacman.py -p ExpectimaxAgent -l smallClassic -a evalFn=better -q -n 10
```

### Running Ghostbusters (Tracking)

```bash
# Navigate to tracking directory
cd tracking

# Exact inference with greedy agent
python busters.py -p GreedyBustersAgent -l bigHunt

# Particle filtering
python busters.py -p GreedyBustersAgent -l bigHunt -a inference=ParticleFilter

# Joint particle filter (multiple ghosts)
python busters.py -p GreedyBustersAgent -a inference=MarginalInference -l bigHunt
```

### Running Reinforcement Learning

```bash
# Navigate to reinforcement directory
cd reinforcement

# Value iteration on gridworld
python gridworld.py -a value -i 100 -g BridgeGrid

# Q-learning on gridworld
python gridworld.py -a q -k 100

# Pacman with Q-learning
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid

# Approximate Q-learning
python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid

# Crawler robot
python crawler.py
```

## Implementation Highlights

### Search Algorithms

**Graph Search Template**:
```python
def graphSearch(problem):
    frontier = DataStructure()  # Stack/Queue/PriorityQueue
    visited = set()
    frontier.push((start_state, []))
    
    while not frontier.isEmpty():
        state, actions = frontier.pop()
        
        if problem.isGoalState(state):
            return actions
            
        if state not in visited:
            visited.add(state)
            for successor, action, cost in problem.getSuccessors(state):
                frontier.push((successor, actions + [action]))
```

**A* Search with Heuristic**:
- Priority = g(n) + h(n) where:
  - g(n) = cost from start to current node
  - h(n) = heuristic estimate to goal
- Guarantees optimality if heuristic is admissible (h(n) ≤ true cost)
- Consistency (h(n) ≤ cost(n,n') + h(n')) improves efficiency

### Minimax with Alpha-Beta Pruning

**Pruning Conditions**:
- At MAX node: if value ≥ β, prune remaining children
- At MIN node: if value ≤ α, prune remaining children
- Maintains optimality while reducing search space by ~50%

**Implementation**:
```python
def minimax(state, depth, agentIndex, alpha, beta):
    if terminal or depth == 0:
        return evaluationFunction(state)
    
    if agentIndex == 0:  # MAX (Pacman)
        value = -infinity
        for action in legalActions:
            successor = generateSuccessor(state, action)
            value = max(value, minimax(successor, depth, 1, alpha, beta))
            alpha = max(alpha, value)
            if alpha > beta:  # Beta cutoff
                break
        return value
    else:  # MIN (Ghosts)
        value = +infinity
        for action in legalActions:
            successor = generateSuccessor(state, action)
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth - 1 if nextAgent == 0 else depth
            value = min(value, minimax(successor, nextDepth, nextAgent, alpha, beta))
            beta = min(beta, value)
            if alpha > beta:  # Alpha cutoff
                break
        return value
```

### Exact Inference

**Belief Propagation**:
```python
def observeUpdate(observation, gameState):
    # Bayesian update: P(X|obs) ∝ P(obs|X) * P(X)
    for position in allPositions:
        beliefs[position] *= observationProbability(observation, position)
    beliefs.normalize()

def elapseTime(gameState):
    # Predict next beliefs through transition model
    newBeliefs = Counter()
    for position in allPositions:
        nextPositionDist = getPositionDistribution(position)
        for nextPos, prob in nextPositionDist.items():
            newBeliefs[nextPos] += prob * beliefs[position]
    beliefs = newBeliefs
```

### Q-Learning Update

**Temporal Difference Learning**:
```python
def update(state, action, nextState, reward):
    # Q-learning update rule
    sample = reward + gamma * max(Q(nextState, a) for a in actions)
    Q[state][action] = (1 - alpha) * Q[state][action] + alpha * sample
```

**Feature-Based Approximation**:
```python
def update(state, action, nextState, reward):
    # Compute TD error
    prediction = sum(weights[i] * features[i](state, action))
    target = reward + gamma * max(getValue(nextState, a) for a in actions)
    difference = target - prediction
    
    # Update weights
    for i, feature_value in enumerate(features(state, action)):
        weights[i] += alpha * difference * feature_value
```

## Key Concepts Demonstrated

### Search
- **Completeness**: BFS, UCS, and A* are complete (find solution if exists)
- **Optimality**: UCS and A* (with admissible heuristic) find optimal solutions
- **Time/Space Complexity**: Trade-offs between different search strategies
- **Heuristic Design**: Admissibility and consistency properties

### Multi-Agent
- **Game Theory**: Zero-sum games, optimal play assumptions
- **Minimax Theorem**: Existence of optimal mixed strategies
- **Pruning**: Alpha-beta reduces search without affecting result
- **Expectimax**: Modeling probabilistic opponents

### Probabilistic Inference
- **Bayes' Rule**: Updating beliefs with evidence
- **Hidden Markov Models**: State estimation in partially observable environments
- **Particle Filtering**: Monte Carlo approximation methods
- **Belief State**: Probability distribution over possible states

### Reinforcement Learning
- **Markov Decision Processes**: States, actions, transitions, rewards
- **Bellman Equations**: Optimal value function characterization
- **Exploration-Exploitation**: Epsilon-greedy strategies
- **Function Approximation**: Generalization with features
- **Credit Assignment**: Temporal difference learning

## Performance Metrics

### Search
- **Path Cost**: Total cost of solution path
- **Nodes Expanded**: Number of states explored
- **Search Time**: Computational efficiency
- **Solution Quality**: Optimality of found path

### Multi-Agent
- **Win Rate**: Percentage of games won
- **Average Score**: Expected game performance
- **Search Depth**: Lookahead capability
- **Nodes Evaluated**: Computational cost

### Tracking
- **Tracking Accuracy**: Belief concentration near true position
- **Capture Time**: Steps to catch all ghosts
- **Inference Speed**: Computational efficiency

### Reinforcement Learning
- **Convergence Rate**: Episodes to near-optimal policy
- **Average Reward**: Learned policy performance
- **Generalization**: Performance on unseen states

## Challenges and Solutions

### Search Challenges
- **Large State Spaces**: Solved with informed search (A*) and good heuristics
- **Heuristic Design**: Balancing admissibility with informativeness
- **Multiple Goals**: Handled with problem decomposition and MST heuristics

### Multi-Agent Challenges
- **Exponential Branching**: Mitigated with alpha-beta pruning
- **Depth Limitation**: Addressed with evaluation functions
- **Multiple Ghosts**: Simplified with one-ghost-at-a-time assumptions

### Tracking Challenges
- **Exact Inference Scaling**: Solved with particle filtering approximation
- **Particle Degeneracy**: Handled with reinitialization strategies
- **Joint Distributions**: Approximated with factored representations

### Reinforcement Learning Challenges
- **Large State Spaces**: Addressed with function approximation
- **Exploration**: Balanced with epsilon-greedy and learning rate decay
- **Sample Efficiency**: Improved with experience replay and feature engineering

---

## Project Attribution

**Original Project**: UC Berkeley CS188 Intro to AI - Pacman Projects  
**Developers**: John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu)  
**Course Website**: http://ai.berkeley.edu  

---

**Note**: This project demonstrates fundamental AI concepts through the engaging medium of the Pacman game. Each component builds upon core computer science principles including graph theory, probability theory, optimization, and machine learning.
