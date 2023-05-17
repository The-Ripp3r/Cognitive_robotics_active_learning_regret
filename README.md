# Cognitive_robotics_active_learning_regret
This is the repository for an implementation of an active learning human-in-the-loop algorithm designed for data-sparse environments. The goal of this repo was to characterize the bias, if any, of the trajectories taken by said algorithm towards the training data of the reward model and the oracle queries selected by a unique extension of the “regret query selection method”, and benchmark how the algorithm performs in spite of said bias. 

Quick Start - Will generate the path taken by the robot to find 5 interesting measurements in a simple grid world of 20 x 20 with 1 dimensional uniformly distributed topics:

  - python -m pip install numpy scikit-learn matplotlib

  - python final_proj_cog_rob.py

Note: there is a small error that can occur very rarely (I discuss it in my report), but if it does not run on first try, just rerun it

Documentation:

class State 

	- a class that represents a state in a Markov Decision Process (MDP) in this context

	- attributes: x (int), y (int), topics (np.array), visited (bool), label (bool; 1 if interesting)

	- methods: set_topics, get_topics, set_label

class Oracle

	- a class to represent the oracle/human-in-the-loop

	- methods: label_states, interesting_or_not

	- label_states(states: List[State]): given a set of states, assigns each one their true label as determined by the fixed interest profile (y=x -> 1)

	- interesting_or_not(state: State) -> bool: determines whether state is interesting or not according to fixed interest profile; this is used for evaluation purposes as a stopping condition for active_learning_exploration

class LogisticRewardModel
	
	- a class to represent the reward model mapping topics to interest/reward

	- attributes: classifier (LogisticRegression scikit model), train_data (tuple(np.ndarray,np.ndarray)), 

	- methods: train, predict, update

	- train(data: tuple(np.ndarray, np.ndarray): given data of topics, labels as 2d arrays, the regression model is fit to said data

	- predict(topics: np.ndarray): given an array of topics attributed to some state, return the probability that the state is interesting

	- update(data: tuple(np.ndarray, np.ndarray): given data of topic, label (should only be one data point), add it to the train_data and retrain the model

class MDP 

	- a class to represent the Markov Decision Process (MDP) in this context

	- atrributes: width (int) , height (int) , num_topics (int) , grid (List[List[State]]), all_states (List[state]), oracle (Oracle), train_states (List[states]),
			  reward_model (LogisticRewardModel), estimated_model (LogisiticRewardModel), current_state(State), horizon (int), 
			  reference_trajectory (List[(str,State)], regret_states(List[State])

	- methods: set_uniform_topics, get_state_with_x_y_value, get_initial_state, _valid_action, _apply_action, generate_trajectories, update_state,
		     make_estimated_model, get_topics

	- set_uniform_topics: creates a uniform topic distribution by assigning a topic value to every state according to the size of the grid world and its x,y position
	
	- get_state_with_x_y_value(x: int, y: int) -> State: return state with specified x,y position on grid

	- get_initial_state -> State: return initial state of MDP

	- _valid_action(state: State, action:str) -> bool: returns true if the action direction specified from given state is within bounds of the grid

	- _apply_action(state: State, action:str) -> State: return the next state or the state you would be at if you took the given action direction (str) from the given state
	
	- update_state(state: State, trajectory: List[(str)]): given a list of actions, apply each action to a given initial state, and mark each state as visited

	- get_topics(stat: State): return the topics initially assigned by set_uniform_topics if the state is visited, otherwise linear interpolate states from nearby states with a radius of 1

	- generate_trajectories(state: State, h: int, visited: List[State]) -> List[List[str, State]] : returns the set of all possible trajectories in the form of a list of actions,state from an initial state up to a horizon h with each state visited a maximum of one time.  

	- make_estimated_model(state:State, true_label: bool): add the state.get_topics, true_label to the training data of the current reward_model and make a new estimated_model for use in the regret query selection; need to train a new model without replacing the old one



OTHER IMPORTANT FUNCTIONS:

	- get_optimal_trajectory(mdp:MDP, rewardModel: LogisticRewardModel, trajectories: List[List[str, State]])-> List[str, State], float: returns the optimal trajectory and its predicted reward value by accessing the predicted reward value at each state along a potential trajectory and choosing the trajectory with maximum sum

	- calc_regret(mdp:MDP, state:State, true_label:int) -> float: returns the regret of a given state s as defined by reward(optimal_traj|true_label)-reward(relative_traj|true_label) where the true_label is actually a guess for the true_label of s and the reward model is retrained using this guess; the relative trajectory is the most recent trajectory

	- regret_based_query_selection(mdp:MDP, k:int) -> List[State]: returns the k vistied states with unlabeled measurements and with most expected regret:  pred_state * r_1 + (1 - pred_state) * r_0 where pred_state is the current estimate of the reward of a state and r0 is the regret of a state with a guess of its true label = 0.

	- active_learning_exploration(mdp:MDP, n:int, bandwidth:int, oracle:Oracle, uniform: bool): returns the trajectory taken by the agent to find n interesting data points in a grid world defined by a given mdp and a given bandwidth that constrains the number of queries, the uniform bool is default set to false to choose regret query selection as opposed to uniform query selection


