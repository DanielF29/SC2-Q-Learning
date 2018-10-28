# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 21:19:43 2018

@author: dfloresa
20181022_RefinedSparceReward_1

"""
import os.path
import random
import math
import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
#from sklearn.cluster import KMeans

#_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
#_NEUTRAL_MINERAL_FIELD = 341 

DATA_FILE_181025_0840 = 'RefinedSparse_agent_data_181025_0840' 

ACTION_DO_NOTHING = 'donothing'

#ACTION_SELECT_Larva = 'selectLarva'
#ACTION_SELECT_Drone = 'selectDrone'
#ACTION_SELECT_SpawningPool = 'selectSpawningPool'
#ACTION_SELECT_ARMY = 'selectarmy'

ACTION_BUILD_Overlord = 'buildOverlord'
ACTION_BUILD_SpawningPool = 'buildSpawningPool'
ACTION_BUILD_Zergling = 'buildZerling'

ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SpawningPool,    
    ACTION_BUILD_Overlord,
    ACTION_BUILD_Zergling,
]

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if ((mm_x + 1) % 32 == 0 ) and ((mm_y + 1) % 32 == 0):
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))
            
KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        #print("XXXXXXXXXX_columns:", self.actions)
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        #print("-x-x-x-x-x", self.q_table.columns)
        self.disallowed_actions = {}
        


    def choose_action(self, observation, excluded_actions=[]):
        #print("XPre-Choosed-action:", self.q_table.columns)
        self.check_state_exist(observation)
        self.disallowed_actions[observation] = excluded_actions
        state_action = self.q_table.ix[observation, :]
        for excluded_action in excluded_actions:
            del state_action[excluded_action]
            #a=0

        if np.random.uniform() < self.epsilon:
            # choose best action
            #state_action = self.q_table.ix[observation, :]
           
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
           
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(state_action.index)
        
        #a+=0
        #print("XXXXXChoosed-action:", self.q_table.columns)
        return action
    
    def learn(self, s, a, r, s_):
        if s == s_:
            return
        #print("XXXXXLearn-S_:", self.q_table.columns)
        self.check_state_exist(s_)
        self.check_state_exist(s)
       
        q_predict = self.q_table.ix[s, a]
        
        s_rewards = self.q_table.ix[s_, :]
        
        if s_ in self.disallowed_actions:
            for excluded_action in self.disallowed_actions[s_]:
                del s_rewards[excluded_action]
        
        if s_ != 'terminal':
            q_target = r + self.gamma * s_rewards.max() 
                                        #self.q_table.ix[s_, :].max()
        else:
            q_target = r # next state is terminal

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table        
            #print("XXXXXXXXXX_val:", [0] * len(self.actions))
            #print("XXXXXXXXXX_index:", self.q_table.columns)
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)) 
                        
class ZergAgent(base_agent.BaseAgent):
  def __init__(self):
    super(ZergAgent, self).__init__()
   
    #self.attack_coordinates = None
    #print("%%%%%%%%%%%%%%%", len(smart_actions))
    self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
    #print("%%%%%%%%%%%%%%%_QlearningTable")
    #print("%%%%%%:", self.qlearn.q_table.columns)
    #self.previous_killed_unit_score = 0
    #self.previous_killed_building_score = 0
    self.previous_action = None
    self.previous_state = None
    self.cc_y = None
    self.cc_x = None   
    self.move_number = 0
    if os.path.isfile(DATA_FILE_181025_0840 + '.gz'):
        self.qlearn.q_table = pd.read_pickle(DATA_FILE_181025_0840 + '.gz', compression='gzip')
    #print("%%%%%%2:", self.qlearn.q_table.columns)     
    
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
    
    #print("%%%Step%%%:", self.qlearn.q_table.columns)
    if obs.last():
        reward = obs.reward 
        self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
        self.qlearn.q_table.to_pickle(DATA_FILE_181025_0840 + '.gz', 'gzip')
        self.qlearn.q_table.to_csv(DATA_FILE_181025_0840 + '.csv')
        self.previous_action = None
        self.previous_state = None
        self.move_number = 0
        return actions.FUNCTIONS.no_op()
    #unit_type = obs.observation['screen'][features.SCREEN_FEATURES.unit_type.index]
    if obs.first():
      player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
      xmean = player_x.mean()
      ymean = player_y.mean()
      if xmean <= 31 and ymean <= 31:
        self.attack_coordinates = (49, 49)
        self.base_top_left = 1
      else:
        self.attack_coordinates = (12, 16)
        self.base_top_left = 0
    #--------------------------------------------------------------------------
    ccs = self.get_units_by_type(obs, units.Zerg.Hatchery)
    cc_count = cc_count = 1 if len(ccs) >= 1 else 0    
    spawning_pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)
    spawning_pools_count = spawning_pools_count = 1 if len(spawning_pools) >= 1 else 0
    supply_used  = obs.observation.player.food_used
    supply_limit = obs.observation.player.food_cap
                   #obs.observation['player'][4]
    #larvas_count = obs.observation.player.larva_count
    drones_count = obs.observation.player.food_workers
    army_supply = obs.observation.player.food_army
                  #obs.observation['player'][5]
    free_supply = (supply_limit - supply_used)      
    idle_workers = obs.observation.player.idle_worker_count 
    if idle_workers > 0:
        idle_workers = 0
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<                  
    if self.move_number == 0:
        self.move_number = 1
        #print("%%%move-0%%%:", self.qlearn.q_table.columns)
        States_count = 3
        hot_squares_len = 4
        green_squares_len = 4
        current_state = np.zeros(States_count + hot_squares_len + green_squares_len)
        current_state[0] = army_supply
        current_state[1] = cc_count             
        current_state[2] = spawning_pools_count
        #current_state[3] = free_supply
        #current_state[4] = drones_count
        hot_squares = np.zeros(hot_squares_len)        
        enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.ENEMY).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 32))
            x = int(math.ceil((enemy_x[i] + 1) / 32))
            hot_squares[((y - 1) * 2) + (x - 1)] = 1
        if not self.base_top_left:
            hot_squares = hot_squares[::-1]
        for i in range(0, hot_squares_len):
            current_state[i + States_count] = hot_squares[i]
        #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
        green_squares = np.zeros(green_squares_len)        
        friendly_y, friendly_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero() 
        for i in range(0, len(friendly_y)):
            y = int(math.ceil((friendly_y[i] + 1) / 32))
            x = int(math.ceil((friendly_x[i] + 1) / 32))
            green_squares[((y - 1) * 2) + (x - 1)] = 1
        if not self.base_top_left:
            green_squares = green_squares[::-1]
        for i in range(0, green_squares_len):
            current_state[i + States_count + hot_squares_len] = green_squares[i]
    #--------------------------------------------------------------------------
        #killed_unit_score = obs.observation.score_cumulative.killed_value_units
                            #obs.observation['score_cumulative'][5]
        #killed_building_score = obs.observation.score_cumulative.killed_value_structures
                            #obs.observation['score_cumulative'][6]        
    #--------------------------------------------------------------------------
        if self.previous_action is not None:
        #reward = 0
        #if killed_unit_score > self.previous_killed_unit_score:
        #    reward += KILL_UNIT_REWARD
        #if killed_building_score > self.previous_killed_building_score:
        #    reward += KILL_BUILDING_REWARD
            self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))
        #--------------------------------------------------------------------------
        excluded_actions = []
        if drones_count == 0 or cc_count == 0:
            excluded_actions.append(1)
        #if larvas_count == 0:
        #    excluded_actions.append(2)
        #if larvas_count == 0 or spawning_pools_count == 0 or free_supply == 0:
        if spawning_pools_count == 0 or free_supply == 0:
            excluded_actions.append(3)
        if army_supply == 0:
            excluded_actions.append(4)
            excluded_actions.append(5)
            excluded_actions.append(6)
            excluded_actions.append(7)
    #--------------------------------------------------------------------------    
        #print("%%%Pre-choose%%%:", self.qlearn.q_table.columns)
        rl_action = self.qlearn.choose_action(str(current_state), excluded_actions)
        self.previous_state = current_state
        self.previous_action = rl_action
        smart_action, attack_x, attack_y = self.splitAction(rl_action)
        #smart_action = smart_actions[rl_action]
        #print ("smart_action %", smart_action)
        #self.previous_killed_unit_score = killed_unit_score
        #self.previous_killed_building_score = killed_building_score
        #print("%%%after-choose%%%:", self.qlearn.q_table.columns)
    #####--------------------------------------------------------------------------
        if smart_action == ACTION_DO_NOTHING:
           return actions.FUNCTIONS.no_op()
    #####--------------------------------------------------------------------------
    #####--------------------------------------------------------------------------
        if smart_action == ACTION_BUILD_SpawningPool:
            drones = self.get_units_by_type(obs, units.Zerg.Drone)
            if len(drones) > 0:
                drone = random.choice(drones)
                return actions.FUNCTIONS.select_point("select_all_type", (drone.x, drone.y))
    #####-------------------------------------------------------------------------- 
        elif smart_action == ACTION_BUILD_Overlord or smart_action == ACTION_BUILD_Zergling:
            larvae = self.get_units_by_type(obs, units.Zerg.Larva)
            if len(larvae) > 0:
              larva = random.choice(larvae)
              return actions.FUNCTIONS.select_point("select_all_type", (larva.x, larva.y))
    #####--------------------------------------------------------------------------
        elif smart_action == ACTION_ATTACK:
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
    #####-------------------------------------------------------------------------- 
        if smart_action == ACTION_BUILD_SpawningPool:
            spawning_pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)
            if len(spawning_pools) == 0:
              if self.unit_type_is_selected(obs, units.Zerg.Drone):
                if self.can_do(obs, actions.FUNCTIONS.Build_SpawningPool_screen.id):
                  x = random.randint(0, 83)
                  y = random.randint(0, 83)
                  return actions.FUNCTIONS.Build_SpawningPool_screen("now", (x, y))
    #####-------------------------------------------------------------------------- 
        elif smart_action == ACTION_BUILD_Overlord or smart_action == ACTION_BUILD_Zergling:
            if smart_action == ACTION_BUILD_Overlord:
                if self.unit_type_is_selected(obs, units.Zerg.Larva):
                    free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
                    if free_supply < 3 and supply_limit < 197:
                        if self.can_do(obs, actions.FUNCTIONS.Train_Overlord_quick.id):
                            return actions.FUNCTIONS.Train_Overlord_quick("now")
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
            if smart_action == ACTION_BUILD_Zergling:
                if self.unit_type_is_selected(obs, units.Zerg.Larva):
                    if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
                        return actions.FUNCTIONS.Train_Zergling_quick("now")         
    #####--------------------------------------------------------------------------
        elif smart_action == ACTION_ATTACK:
            if self.unit_type_is_selected(obs, units.Zerg.Zergling):
            #if obs.observation.single_select[0][0] != units.Zerg.Drone and actions.FUNCTIONS.Attack_minimap.id in obs.observation.available_actions:
                x_offset = random.randint(-1, 1)
                y_offset = random.randint(-1, 1)    
                return actions.FUNCTIONS.Attack_minimap("now", self.transformLocation(int(attack_x)+ (x_offset * 8), int(attack_y)+(y_offset * 8)))
    #####--------------------------------------------------------------------------
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    return actions.FUNCTIONS.no_op()

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
