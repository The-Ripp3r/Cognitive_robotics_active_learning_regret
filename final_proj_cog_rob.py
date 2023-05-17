import random
import numpy as np
from typing import List, Tuple
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class State:
    def __init__(self, x: int, y: int, num_topics: int):
        self.x = x
        self.y = y
        self.topics = np.zeros(num_topics)
        self.visited = False
        self.label = None

    def set_topics(self, topics: np.ndarray):
        self.topics = topics

    def get_topics(self) -> np.ndarray: 
        return self.topic
    def set_label(self, label: int):
        self.label = label


class Oracle:
    def label_states(self, states: List[State]) -> None:
        for state in states:
            if state.x == state.y:
                state.set_label(1)
            else:
                state.set_label(0)
    
    def interesting_or_not(self, state: State) -> bool:
        #this is used just for counting the number of interesting states visited
        if state.x==state.y:
            return 1
        else: return 0
class LogisticRewardModel:
    def __init__(self, train_data: tuple) -> None:
        self.classifier=None
        self.train_data=train_data
        self.train(self.train_data)

    def train(self, data: tuple): #maybe make hidden
        self.classifier = LogisticRegression(random_state = 0).fit(data[0], data[1])

    def predict(self, topics: np.ndarray):
        #topics should be np.array([[],[],[]])
        #classes are ordered 0,1 use probability of being 1 or interesting as reward
        return self.classifier.predict_proba(topics)[0][1] #make a new model for regret based query selection

    def update(self, data: tuple) -> None: # this is only called when oracle returns query
        topics_current=self.train_data[0]
        labels_current=self.train_data[1]
        self.train_data=np.insert(topics_current, len(topics_current), data[0], axis=0), np.insert(labels_current, len(labels_current), data[1]) #remove axis to flatten
        self.train(self.train_data)

class MDP:
    def __init__(self, width: int, height: int, num_topics: int) -> None:
        # For simplicity, have width and height be the same
        self.width = width
        self.height = height
        self.num_topics=num_topics
        self.grid = [[State(x, y, num_topics) for y in range(height)] for x in range(width)]
        self.all_states = [self.grid[x][y] for x in range(width) for y in range(height)]
        self.set_uniform_topics()
        self.oracle=Oracle()
        #10% of total grid; assuming square
        train_labels=[]
        self.train_states=[]
        while sum(train_labels)==0:
            print("training")
            train_states= [self.grid[i][np.random.randint(len(self.grid[1]), size=1)[0]] for i in range(len(self.grid[0]))]
            train_topics=np.array([state.get_topics() for state in train_states])
            self.oracle.label_states(train_states)
            self.train_states=train_states #used for graphing
            train_labels=np.array([state.label for state in train_states]) #dim 1
        train_data = train_topics, train_labels
        self.reward_model = LogisticRewardModel(train_data)
        self.estimated_model=None
        self.current_state=self.get_initial_state()
        self.horizon=self.width//5 
        self.reference_trajectory=None
        self.regret_states=[]

    def set_uniform_topics(self):
        width = self.width
        height = self.height
        for s in self.all_states:
            x = s.x
            y = s.y
            topics = np.array([(x + y) / (height + width)])
            s.set_topics(topics)

    def get_state_with_x_y_value(self, x: int, y: int) -> State:
        # note - x and y start with 0 indexing
        for s in self.all_states:
            if s.x == x and s.y == y:
                return s
        return None

    def get_initial_state(self) -> State:
        initial_state = self.get_state_with_x_y_value(0, 0)
        initial_state.visited = True
        return initial_state


    def _valid_action(self, state: State, action: str) -> bool:
        x, y = state.x, state.y
        if action == "up":
            y += 1
        elif action == "down":
            y -= 1
        elif action == "left":
            x -= 1
        elif action == "right":
            x += 1
        return (0 <= x < self.width) and (0 <= y < self.height)

    def _apply_action(self, state: State, action: str) -> State:
        if self._valid_action(state, action):
            if action == "up":
                return self.grid[state.x][state.y + 1]
            elif action == "down":
                return self.grid[state.x][state.y - 1]
            elif action == "left":
                return self.grid[state.x - 1][state.y]
            elif action == "right":
                return self.grid[state.x + 1][state.y]
        return state
    

    def update_state(self, state: State, trajectory: List[str], visit_once: bool = False) -> None:
        #self.reference_trajectory=trajectory #saves most recent trajectory as reference for regret query selection
        for next_action in trajectory:
            next_state = self._apply_action(state, next_action)
            if visit_once and next_state.visited:
                continue
            state.visited = True
            next_state.visited = True
            state = next_state
        
        self.current_state=state
        return state
    
    def get_topics(self, state:State):
        if state.visited:
            return state.get_topics()
        else:
            #interpolation
            actions = ["up", "down", "left", "right"]
            sum=np.zeros(self.num_topics)
            count=0
            for action in actions:
                if not self._valid_action(state, action):
                    continue
                next_state = self._apply_action(state, action)
                if next_state.visited:
                    sum+=next_state.get_topics()
                    count+=1
            if sum>0:
                return sum/count
            else: return sum

    def generate_trajectories(self, state: State, h: int, visited: List[State] = []) -> List[List[str]]:
        #visited in header is artifact of recursion
        if h == 0:
            return [[]]
        trajectories = [] # actions
        actions = ["up", "down", "left", "right"]
        visited = [s for s in self.all_states if s.visited] + visited
        for action in actions:
            if not self._valid_action(state, action):
                continue
            next_state = self._apply_action(state, action)
            if next_state not in visited:
                new_visited = visited + [state]
                for trajectory in self.generate_trajectories(next_state, h - 1, new_visited):
                    trajectories.append([(action, next_state)] + trajectory)
        return trajectories
    
    def make_estimated_model(self, state:State, true_label):
        #used for regret query selection
        topics_current=self.reward_model.train_data[0]
        labels_current=self.reward_model.train_data[1]
        data = np.insert(topics_current,len(topics_current),state.get_topics(), axis=0), np.insert(labels_current, len(labels_current), [true_label])
        self.estimated_model=LogisticRewardModel(data)


            
####MAIN HELPERS########

def get_optimal_trajectory(mdp: MDP, rewardModel, trajectories: List[List[Tuple[str, State]]]) -> Tuple[List[Tuple[str, State]],float]:
    best_trajectory = []
    max_reward = -np.inf
    for trajectory in trajectories:
        reward = sum(rewardModel.predict(state.get_topics().reshape(1,mdp.num_topics)) for _, state in trajectory)
        if reward > max_reward:
            max_reward = reward
            best_trajectory = trajectory
    return best_trajectory, max_reward

def regret_based_query_selection(mdp: MDP, k: int = 1) -> List[State]:
    # Return k states with the most regret (must also calculate regret per state)
    # Make sure to only measure regret for visited states that don't currently have a label
    regrets = []
    for state in mdp.all_states:
        if state.visited and state.label is None:
            y_pred = mdp.reward_model.predict(state.get_topics().reshape(1,mdp.num_topics))
            r_1 = calc_regret(mdp, state, 1)
            r_0 = calc_regret(mdp, state, 0)
            regret = y_pred * r_1 + (1 - y_pred) * r_0
            regrets.append((state, regret))

    regrets.sort(key=lambda x: (x[1], x[0].x, x[0].y), reverse=True)
    answer = [state for state, _ in regrets[:k]]
    answer = answer[::-1]
    return answer

def calc_regret(mdp:MDP, state: State, true_label:int) -> float:
    #create new temp reward model
    mdp.make_estimated_model(state, true_label)
    #generate optimal trajectory using new reward model
    a, s_prime = get_optimal_trajectory(mdp, mdp.estimated_model, mdp.generate_trajectories(mdp.current_state, mdp.horizon))
    s_0 = sum(mdp.estimated_model.predict(mdp.get_topics(state).reshape(1,mdp.num_topics)) for _, state in mdp.reference_trajectory)
    return s_prime - s_0

def active_learning_exploration(mdp: MDP, n: int, bandwidth: int, oracle: Oracle, uniform: bool = False) -> None:
    current_state = mdp.get_initial_state()
    visited_interesting_states=0
    steps=0
    total_traj=[]

    while visited_interesting_states<n:
        # Generate trajectories
        trajectories = mdp.generate_trajectories(mdp.current_state, mdp.horizon)

        # Get the optimal trajectory
        optimal_trajectory, val = get_optimal_trajectory(mdp, mdp.reward_model, trajectories)

        #bookkeeping
        total_traj+=optimal_trajectory

        # Update the MDP state based on the optimal trajectory
        mdp.update_state(mdp.current_state, [action for action, _ in optimal_trajectory], visit_once=True)
        mdp.reference_trajectory=optimal_trajectory #bc we just did the optimal traj it is our most recent traj
        steps+=mdp.horizon

        #update # of visited interesting states, not known to robot just using omnipotent view, makes experimentation easier
        #less on convergence of reward model
        visited_interesting_states+=sum(oracle.interesting_or_not(state) for _, state in mdp.reference_trajectory)

        if steps>bandwidth: 
            #print("querying")
            #reset # of steps
            steps=0
            if not uniform:
                # Get k states with the most regret where k is how many queries we would have made over the trajectory given our bandwidth
                states_with_most_regret = regret_based_query_selection(mdp)#=steps//bandwidth)
                mdp.regret_states+=states_with_most_regret
            else: states_with_most_regret=[mdp.current_state] #most recent state
            # Ask the oracle to label the states with the most regret
            oracle.label_states(states_with_most_regret)
            # Update the reward model with newly labeled data
            d1 = np.array([state.get_topics() for state in states_with_most_regret])
            #print(d1)
            d2 = np.array([state.label for state in states_with_most_regret])
            #print(d2)
            if len(d2)==0:
                #generate trajectory
                print("error")
            data = d1, d2
            mdp.reward_model.update(data)
    
    print("graphing time")
    for action,state in total_traj:
        tup= state.x, state.y
        print(tup)
    return total_traj

def show_board_w_policy(mdp:MDP, board, policy):
  fig, ax = plt.subplots()

  # plot squares
  height, width = len(board), len(board[1])
  for i in range(height):
    for j in range(width):
      colors = {0:'c',1:'k',2:'y',3:'g'}
      color = colors[board[i][j]]
      if (color != 'y' and color != 'g') and (i, j) in policy: color = 'r'
      rect = patches.Rectangle((j,height-1-i), 1, 1, linewidth=1, edgecolor=color, facecolor=color)
      ax.add_patch(rect)

  # resize
  plt.xlim(0, mdp.width)
  plt.ylim(0, mdp.height)
  fig.set_size_inches(5,5)
  plt.show()


if __name__ == "__main__":
    # n=5 #of required interesting states
    # bandwidth=5
    # mdp=MDP(20,20,1)
    # oracle=Oracle()
    # traj=active_learning_exploration(mdp, n, bandwidth, oracle)
    # x=[state.x for action, state in traj]
    # x.insert(0,0)
    # y=[state.y for action, state in traj]
    # y.insert(0,0)
    # plt.plot(x, y)
    # training_x = [state.x for state in mdp.train_states]
    # training_y = [state.y for state in mdp.train_states]
    # for i in range(len(training_x)):
    #     plt.plot(training_x[i], training_y[i], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
    # regret_x = [state.x for state in mdp.regret_states]
    # regret_y = [state.y for state in mdp.regret_states]
    # for i in range(len(regret_x)):
    #     plt.plot(regret_x[i], regret_y[i], marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green")
    # plt.grid(color = 'yellow', linestyle = '--', linewidth = 1.5)
    # plt.xlim(0, max(x))
    # plt.ylim(0, max(y))
    # plt.show()

    # num_explored_states=[]
    # num_interesting_data=[]
    # for n in range(1,10):
    #     num_interesting_data.append(n)
    #     bandwidth=5
    #     mdp=MDP(20,20,1)
    #     oracle=Oracle()
    #     traj=active_learning_exploration(mdp, n, bandwidth, oracle)
    #     num_explored_states.append(len(traj))
    # plt.plot(num_interesting_data, num_explored_states)
    # plt.xlabel('num_interesting_data_required')
    # plt.ylabel('num_explored_states')
    # plt.title('Explored States vs Required # of interesting data')
    # plt.show()

    num_explored_states=[]
    num_interesting_data=[]
    for n in range(1,10):
        num_interesting_data.append(n)
        bandwidth=5
        mdp=MDP(20,20,1)
        oracle=Oracle()
        traj=active_learning_exploration(mdp, n, bandwidth, oracle, uniform=True)
        num_explored_states.append(len(traj))
    plt.plot(num_interesting_data, num_explored_states)
    plt.xlabel('num_interesting_data_required')
    plt.ylabel('num_explored_states')
    plt.title('(UNIFORM) Explored States vs Required # of interesting data')
    plt.show()

##########MISC###########

# def generate_test_mdp(width: int, height: int):
#     mdp = MDP(width, height, 1) #change 1 to 3 later
#     return mdp

# def set_all_states_to_being_visited(mdp: MDP):
#     for s in mdp.all_states:
#         s.visited = True

# def update_reward_function(mdp: MDP, data: tuple) -> None:
#     mdp.reward_model.update(data)

