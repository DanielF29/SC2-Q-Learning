"""

@author: DanielF29
20181031_Q-table_SC2_1

"""
import os.path
import random
import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

DATA_FILE = 'file_181029_2140_RSR_simplified' 

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_Zergling = 'buildZerling'
ACTION_ASSAULT = 'assault'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_Zergling,
    ACTION_ASSAULT,
]

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if ((mm_x + 1) % 32 == 0 ) and ((mm_y + 1) % 32 == 0):
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy 
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) 
        self.disallowed_actions = {}
        
    def choose_action(self, observation, excluded_actions=[]):
 
        self.check_state_exist(observation)
        self.disallowed_actions[observation] = excluded_actions
        state_action = self.q_table.ix[observation, :]
        for excluded_action in excluded_actions:
            del state_action[excluded_action]

        if np.random.uniform() < self.epsilon:
            # choose best action
            #state_action = self.q_table.ix[observation, :]
           
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
           
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(state_action.index)

        return action
    
    def learn(self, s, a, r, s_):
        if s == s_:
            return

        self.check_state_exist(s_)
        self.check_state_exist(s)
       
        q_predict = self.q_table.ix[s, a]
        
        s_rewards = self.q_table.ix[s_, :]
        
        if s_ in self.disallowed_actions:
            for excluded_action in self.disallowed_actions[s_]:
                del s_rewards[excluded_action]
        
        if s_ != 'terminal':
            q_target = r + self.gamma * s_rewards.max() 
             
        else:
            q_target = r # next state is terminal

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table        
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)) 
                        
class ZergAgent(base_agent.BaseAgent):
  def __init__(self):
    super(ZergAgent, self).__init__()
   
    self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
    self.previous_action = None
    self.previous_state = None
    self.move_number = 2
    if os.path.isfile(DATA_FILE + '.gz'):
        self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
    self.a=0
    
  def unit_type_is_selected(self, obs, unit_type):
    if (len(obs.observation.single_select) > 0 and
        obs.observation.single_select[0].unit_type == unit_type):
      return True
    if (len(obs.observation.multi_select) > 0 and
        obs.observation.multi_select[0].unit_type == unit_type):
      return True
    return False

  def get_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]
 
  def can_do(self, obs, action):
    return action in obs.observation.available_actions

  def transformDistance(self, x, x_distance, y, y_distance):
      if not self.base_top_left:
          return [x - x_distance, y - y_distance]
      return [x + x_distance, y + y_distance]
    
  def transformLocation(self, x, y):
      if not self.base_top_left:
          return [64 - x, 64 - y]
      return [x, y]

  def splitAction(self, action_id):
      smart_action = smart_actions[action_id]       
      x = 0
      y = 0
      if '_' in smart_action:
          smart_action, x, y = smart_action.split('_')
      return (smart_action, x, y)
  #----------------------------------------------------------------------------
  #----------------------------------------------------------------------------
  def step(self, obs):
    super(ZergAgent, self).step(obs)
    
    if obs.last():
        reward = obs.reward 
        self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
        self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
        self.qlearn.q_table.to_csv(DATA_FILE + '.csv')
        self.previous_action = None
        self.previous_state = None
        self.move_number = 2
        return actions.FUNCTIONS.no_op()
    if obs.first():
      player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
      xmean = player_x.mean()
      ymean = player_y.mean()
      if xmean <= 31 and ymean <= 31:
        self.assault_coordinates = (49, 49)
        self.base_top_left = 1
      else:
        self.assault_coordinates = (12, 16)
        self.base_top_left = 0
    #--------------------------------------------------------------------------   
    spawning_pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)
    spawning_pools_count = spawning_pools_count = 1 if len(spawning_pools) >= 1 else 0
    supply_used  = obs.observation.player.food_used
    supply_limit = obs.observation.player.food_cap
    army_count = obs.observation.player.army_count
    free_supply = (supply_limit - supply_used)      
    drones = self.get_units_by_type(obs, units.Zerg.Drone)
    if spawning_pools_count == 0 and len(drones) > 0 and not (self.unit_type_is_selected(obs, units.Zerg.Drone)): 
            drone = random.choice(drones)
            while (drone.x >= 84 or drone.y >= 84 or drone.x <0 or drone.y <0):
                    drones = self.get_units_by_type(obs, units.Zerg.Drone)
                    self.a+=1
                    if len(drones) == 0 or self.a>10:
                        self.a=0
                        return actions.FUNCTIONS.no_op()
                    drone = random.choice(drones)
                    print ('Drone(x,y): (',drone.x, ',',drone.y, ')')
            return actions.FUNCTIONS.select_point("select_all_type", (drone.x, drone.y))
    #####-------------------------------------------------------------------------- 
    if spawning_pools_count == 0 and self.unit_type_is_selected(obs, units.Zerg.Drone):
        if self.can_do(obs, actions.FUNCTIONS.Build_SpawningPool_screen.id): 
            x = random.randint(0, 83)
            y = random.randint(0, 83)
            return actions.FUNCTIONS.Build_SpawningPool_screen("now", (x, y))
    if spawning_pools_count == 0:
        self.move_number = 2
    if spawning_pools_count == 1 and self.move_number == 2:
        self.move_number = 0
    #####--------------------------------------------------------------------------  
    larvae = self.get_units_by_type(obs, units.Zerg.Larva)
    if free_supply < 10 and supply_limit < 197 and len(larvae) > 0 and not (self.unit_type_is_selected(obs, units.Zerg.Larva)): 
        larva = random.choice(larvae)
        while (larva.x >= 84 or larva.y >= 84 or larva.x <0 or larva.y <0):
           self.a +=1           
           larvae = self.get_units_by_type(obs, units.Zerg.Larva)
           larva = random.choice(larvae)
           print ('Overlord_Larva(x,y): (',larva.x, ',',larva.y,')')
           if len(larvae) == 0 or self.a>10:
               self.a=0
               return actions.FUNCTIONS.no_op()
        return actions.FUNCTIONS.select_point("select_all_type", (larva.x, larva.y))
    if free_supply < 10 and supply_limit < 197 and self.unit_type_is_selected(obs, units.Zerg.Larva): 
        if self.can_do(obs, actions.FUNCTIONS.Train_Overlord_quick.id):
            return actions.FUNCTIONS.Train_Overlord_quick("now")
#<<>><<>><<>><<>><<>><<>><<>><<>><<>><<<>><<>><<>><<>><<>><<>><<>><<>><<>><<<>><<>><                  
    if self.move_number == 0:
        self.move_number = 1
        current_state = np.zeros(1)
        current_state[0] = army_count
    #--------------------------------------------------------------------------
        if self.previous_action is not None:
            reward = 0
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))
    #--------------------------------------------------------------------------
        excluded_actions = []
        if spawning_pools_count == 0 or free_supply == 0 and len(larvae) == 0 :
            excluded_actions.append(1)
        if army_count == 0:
            excluded_actions.append(2)
            excluded_actions.append(3)
            excluded_actions.append(4)
            excluded_actions.append(5)
    #--------------------------------------------------------------------------    
        rl_action = self.qlearn.choose_action(str(current_state), excluded_actions)
        self.previous_state = current_state
        self.previous_action = rl_action
        smart_action, attack_x, attack_y = self.splitAction(rl_action)
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if smart_action == ACTION_DO_NOTHING:
           return actions.FUNCTIONS.no_op()
    #####-------------------------------------------------------------------------- 
        elif  smart_action == ACTION_BUILD_Zergling:
            larvae = self.get_units_by_type(obs, units.Zerg.Larva)
            if len(larvae) > 0:
              larva = random.choice(larvae)
              while (larva.x >= 84 or larva.y >= 84 or larva.x <0 or larva.y <0):
                   larvae = self.get_units_by_type(obs, units.Zerg.Larva)
                   self.a +=1
                   if len(larvae) == 0 or self.a>10:
                       self.a=0
                       return actions.FUNCTIONS.no_op()
                   larva = random.choice(larvae)
                   print ('Larva(x,y): (',larva.x, ',',larva.y,')')
              return actions.FUNCTIONS.select_point("select_all_type", (larva.x, larva.y))
    #####--------------------------------------------------------------------------
        elif smart_action == ACTION_ATTACK or smart_action == ACTION_ASSAULT:
            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")
    #####--------------------------------------------------------------------------     
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    elif self.move_number == 1:
        self.move_number = 0
        smart_action, attack_x, attack_y = self.splitAction(self.previous_action)
    #####--------------------------------------------------------------------------
        if smart_action == ACTION_DO_NOTHING:
           return actions.FUNCTIONS.no_op()
    #####-------------------------------------------------------------------------- 
        elif smart_action == ACTION_BUILD_Zergling:
            if self.unit_type_is_selected(obs, units.Zerg.Larva):
                if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
                    return actions.FUNCTIONS.Train_Zergling_quick("now")   #queued      
    #####--------------------------------------------------------------------------
        elif smart_action == ACTION_ATTACK or smart_action == ACTION_ASSAULT:
            if self.unit_type_is_selected(obs, units.Zerg.Zergling):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    if smart_action == ACTION_ATTACK: 
                        x_offset = random.randint(-1, 1)
                        y_offset = random.randint(-1, 1)    
                        return actions.FUNCTIONS.Attack_minimap("now", self.transformLocation(int(attack_x)+ (x_offset * 8), int(attack_y)+(y_offset * 8)))
                    if smart_action == ACTION_ASSAULT:
                        return actions.FUNCTIONS.Attack_minimap("now",self.assault_coordinates)
    #####--------------------------------------------------------------------------
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    return actions.FUNCTIONS.no_op()
'''
_______________________________________________________________________________
'''
def main(unused_argv):
  agent = ZergAgent()
  try:
    while True:
      with sc2_env.SC2Env(
          map_name="AbyssalReef",
          players=[sc2_env.Agent(sc2_env.Race.zerg),
                   sc2_env.Bot(sc2_env.Race.random,
                               sc2_env.Difficulty.very_easy)],
          agent_interface_format=features.AgentInterfaceFormat(
              feature_dimensions=features.Dimensions(screen=84, minimap=64),
              use_feature_units=True),
          step_mul=16,
          game_steps_per_episode=0,
          visualize=False) as env:
         
        agent.setup(env.observation_spec(), env.action_spec())
       
        timesteps = env.reset()
        agent.reset()
       
        while True:
          step_actions = [agent.step(timesteps[0])]
          if timesteps[0].last():
            break
          timesteps = env.step(step_actions)
     
  except KeyboardInterrupt:
    pass
 
if __name__ == "__main__":
  app.run(main)
