"""
@author: DanielF29
20181111_SC2_Q-table_RSR_abs-act

Este codigo esta basado en los codigos y tutoriales de "Steven Brown", 
para mayor información y como referencia estos se pueden encontrar en:
    
    https://github.com/skjb/pysc2-tutorial

This code is based on the codes and tutorials by "Steven Brown",
for more info. and as reference some of his works can be found here:
    
    https://github.com/skjb/pysc2-tutorial
    
"""

import os.path
import random
import numpy as np
import pandas as pd
import math

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
#el nombre del archivo que guardara nuestra tabla Q se coloca entre comillas
DATA_FILE = 'file_181111_0000_RSR_obs-act' 

#Acciones que puede tomar nuestro Bot, definidas como strings
ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_Zergling = 'buildZerling' 
ACTION_Scout = 'Scout' #Se explorar al asar entre 5 posiciones diferentes
ACTION_ATTACK = 'attack'

#lista de acciones
smart_actions = [ 
    ACTION_DO_NOTHING,
    ACTION_BUILD_Zergling,
    ACTION_Scout,
]
#Se agregan 16 acciones de ataque, quedando de la forma: 'attack_x_y'
#dividiendo el mapa en 16 cuadros iguales, al centro de cada uno de estos 
#es donde se dirigirian los ataques.
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if ((mm_x + 1) % 16 == 0 ) and ((mm_y + 1) % 16 == 0):
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        #Hiperparametros del aprendizaje
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy 
        #Definimos la tabla
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) 
        self.disallowed_actions = {} #Lista de acciones desabilitadas
                                     #Debido a que no se puedan tomar dadas 
                                     #las condiciones del estado en el que estemos
        
    def choose_action(self, observation, excluded_actions=[]): #se escogera una acción
        #Primero revisamos si existe el estado, si no se agrega a la tabla-Q
        self.check_state_exist(observation)
        #Se actualiza la variable/arreglo 'disallowed_actions' para el paso de "Learn" (actualización de la tabla Q)
        self.disallowed_actions[observation] = excluded_actions
                               
        #Borramos acciones invalidas/ delete invalid actions
        state_action = self.q_table.ix[observation, :]
        for excluded_action in excluded_actions:
            del state_action[excluded_action]

        if np.random.uniform() < self.epsilon:
            #Selección de la mejor acción, como varias acciones pueden tener 
            #el mismos mejor valor, seleccionaremos al azar de entre las mejores
            #1.-se revuelve 'state_action'
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            #2.-Ya revuelto el state_action se selecciona el primer valor más alto
            action = state_action.idxmax()
        else:
            #Selección de una acción al azar
            action = np.random.choice(state_action.index)
        return action
    
    def learn(self, s, a, r, s_): #Bloque de aprendisaje (actualización del valor de una casilla en la tabla-Q)
        #Si no hemos cambiado de estado evitaremos aprender de este mismo
        #de estaforma el valor del en la tabla no se minimiza por intentos repetidos
        #de aprendizaje en el estado actual (multiplicaciones innecesarias de gamma en la actualización del valor actual)
        if s == s_: 
            return
        
        #Revisamos si ambos estados existen en la tabla Q
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        #Asignamos los valores requeridos para las partes de nuestra regla de 
        #actualización del valor en la tabla Q
        q_predict = self.q_table.ix[s, a]
        s_rewards = self.q_table.ix[s_, :] 
        
        #desabilitaremos/borraremos las acciones que no son posibles en el estado actual
        #Así solo aprenderemos en base a acciones validas y se acelerara el aprendizaje en general
        if s_ in self.disallowed_actions: 
            for excluded_action in self.disallowed_actions[s_]: 
                del s_rewards[excluded_action]
        
        if s_ != 'terminal': #si el siguiente estado no es terminal nuestro valor "q_target"
                             #se estima en base a la regla completa de actualización
            q_target = r + self.gamma * s_rewards.max() 
        else:
            q_target = r #llegamos aquí en caso de ser terminal el siguiente estado
            
        # update / Se actualiza el valor en la tabla-Q
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state): #Checaremos si existe el estado actual en la tabla-Q
        #Si el estado en el que estamos no existia en la tabla Q se agrega a esta
        if state not in self.q_table.index:    
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)) 
                        
class ZergAgent(base_agent.BaseAgent):
  def __init__(self):
    super(ZergAgent, self).__init__()
    
    #Definimos e inicializamaos algunas variables
    self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
    self.previous_action = None 
    self.previous_state = None 
    self.move_number = 2 #Contador de la fase en la que esta el programa de la acción tomada
    if os.path.isfile(DATA_FILE + '.gz'):
        self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
    self.a=0 #Contador para evitar dejar el programa atrado en bucles
    
  def unit_type_is_selected(self, obs, unit_type):
      #Checamos si la selección actual corresponde a la unidad de nuestro interes
      #en selección unica
    if (len(obs.observation.single_select) > 0 and
        obs.observation.single_select[0].unit_type == unit_type):
      return True
      #O en selección multiple
    if (len(obs.observation.multi_select) > 0 and
        obs.observation.multi_select[0].unit_type == unit_type):
      return True
    return False

  def get_units_by_type(self, obs, unit_type):#obtendremos (como arreglos numpy) las caracteristicas del
                                              #tipo de unidades a seleccionar en pantalla
                                              #Las coordenadas X y Y son las caracteristicas que nos estaran interesando. 
    return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]
 
  def can_do(self, obs, action):
      #se revisa si la acción 'action' se puede realizar
    return action in obs.observation.available_actions 

  def transformLocation(self, x, y):
      #Ajustamos las cordenadas de la base (Hatvery), si es que no inicia en la posición 
      #izquierda superior
      if not self.base_top_left:
          return [64 - x, 64 - y]
      return [x, y]

  def splitAction(self, action_id):
      #Aquí se divide las intrucciones de ataque en sus 3 partes:
      #nombre y coordenadas X's y Y's
      smart_action = smart_actions[action_id]       
      x = 0
      y = 0
      if '_' in smart_action:
          smart_action, x, y = smart_action.split('_')
      return (smart_action, x, y)
  #----------------------------------------------------------------------------
  #----------------------------------------------------------------------------
  def step(self, obs): #Esta es la sección principal de nuestro Bot, 
                       #a cada paso del juego aquí se especifica que hace el Bot
    super(ZergAgent, self).step(obs)
    
    #acciones en la ultima observación de cada juego
    if obs.last():
        reward = obs.reward #se devuelve el reward que provee el juego
        #[1]  - Victoria
        #[0]  - Empate
        #[-1] - Derrota
        #Se manda a llamar el aprendizaje en el ultimo paso del juego
        self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
        #Se guarda la tabla Q
        self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
        self.qlearn.q_table.to_csv(DATA_FILE + '.csv')
        #Para el siguiente juego se resetean las variable de 
        #la acción previa, el estado previo 
        #y la variable que indica en que parte de las acciones a tomar vamos
        self.previous_action = None
        self.previous_state = None
        self.move_number = 2 #Se asigna a valor de 2 para permitir construir 
        #un spawningPool al inicio del siguiente juego, antes de la toma de deciciones 
        #desde la tabla Q
        return actions.FUNCTIONS.no_op()
    
    #acciones en la primer observación de cada juego
    if obs.first():
        #Se obtiene el conjunto de coordenadas de nuestras unidades
      player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
      #se promedian para darnos una idea de nuestra ubicación incial en el minimapa
      xmean = player_x.mean()
      ymean = player_y.mean()
      #Se ajustan las coordenadas de ataque a la base enemiga
      #dependiendo si empezamos en la esquina izquiera superior (promedio de coordenadas <31) o no
      #y ajustamos la bariable "base_top_left" acorde a esto.
      if xmean <= 31 and ymean <= 31:
        self.assault_coordinates = (49, 49)
        self.base_top_left = 1
      else:
        self.assault_coordinates = (12, 16)
        self.base_top_left = 0
    #-------------------------------------------------------------------------- 
    #Obtenemos todas las coordenadas de nuestro/s SpawningPools
    spawning_pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)
    #en caso de tener al menos un SpawningPool tendremos más de una coordenada
    #por ello asignamos "spawning_pools_count" a 1 en caso de tener al menos una coordenada
    #con esto sabremos que ya podemos crear tropas (Zerglings)
    spawning_pools_count = spawning_pools_count = 1 if len(spawning_pools) >= 1 else 0
    #obtenemos que tan grande es nuestra población de unidades
    supply_used  = obs.observation.player.food_used
    #obtenemos nuestro limite de población actual (el limite maximo en el juego es de 200)
    supply_limit = obs.observation.player.food_cap
    #obtenemos la cantidad de tropas que tenemos
    army_count = obs.observation.player.army_count
    #Determinamos el espacio disponible que puede crecer la población actualmente
    free_supply = (supply_limit - supply_used) 
    
    #Seleccionaremos un Drone (unidad recolectora/constructora)  
    #Obtención de las coordenadas de los Drones
    drones = self.get_units_by_type(obs, units.Zerg.Drone) 
    #Revisión de que no haya SpawningPools, si haya Drones y no este seleccionado ya un Drone
    if spawning_pools_count == 0 and len(drones) > 0 and not (self.unit_type_is_selected(obs, units.Zerg.Drone)): 
            drone = random.choice(drones) #se escoge un par de coordenadas a azar
            #si las coordenadas resultan estar fuera del rango de la pantalla se seleccionan otras
            while (drone.x >= 84 or drone.y >= 84 or drone.x <0 or drone.y <0): 
                    drones = self.get_units_by_type(obs, units.Zerg.Drone)
                    self.a+=1 
                    #En caso de que ya no haya coordenadas disponibles o 
                    #se realice la selección inexitosamente 10 veces 
                    #se interrumpe el intento de selección y 
                    #se indica que el bot no hace nada en este paso
                    if len(drones) == 0 or self.a>10:
                        self.a=0
                        return actions.FUNCTIONS.no_op()
                    drone = random.choice(drones)
                    print ('Drone(x,y): (',drone.x, ',',drone.y, ')')
            #La acción indicada al juego para seleccionar la coordenada escogida de nuestro Drone
            return actions.FUNCTIONS.select_point("select_all_type", (drone.x, drone.y))
    #####-------------------------------------------------------------------------- 
    #ahora con el Drone seleccionado se procede a indicarle que cree el SpawningPool
    #en una coordenada al azar dentro del campo de vision de la pantalla actual
    if spawning_pools_count == 0 and self.unit_type_is_selected(obs, units.Zerg.Drone):
        if self.can_do(obs, actions.FUNCTIONS.Build_SpawningPool_screen.id): 
            x = random.randint(0, 83)
            y = random.randint(0, 83)
            return actions.FUNCTIONS.Build_SpawningPool_screen("now", (x, y))
    #Mientras aun no se haya creado el SpawningPool no se permite la toma de deciciones del Bot
    #Esto se repite en caso que sea destruido el SpawningPool
    if spawning_pools_count == 0:
        self.move_number = 2
    #En cuanto tenemos 1 SpawningPool procedemos a que el Bot decida las acciones a tomar
    #mediante la variable "self.move_number"
    if spawning_pools_count == 1 and self.move_number == 2:
        self.move_number = 0
    #####--------------------------------------------------------------------------  
    #Bloque para crear un "Overlord" (unidad que aumenta el limite de población)
    #Este bloque se ejecuta automaticamente en cuanto se detecta que 
    #la población esta a menos de 10 unidades del limite actual y aun no se llega
    #al limite maximo de 200 unidades
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
#primer sección para la ejecución de las acciones tomadas por el Bot                
    if self.move_number == 0:
        self.move_number = 1
        #nuestro estado dependera del 3 partes:
        #1.-numero de unidades de combate disponibles al momento
        States_count = 1
        hot_squares_len = 16 #arreglo representando un mapa de 4x4 de donde hay tropas enemigas
        green_squares_len = 16 #arreglo representando un mapa de 4x4 de donde estan nuestras tropas
        current_state = np.zeros(States_count + hot_squares_len + green_squares_len)
        current_state[0] = army_count
        #2.- Divición del minimapa en 16 partes iguales con valor de 1 si contienen tropas enemigas 
        #    si no su valor es cero
        hot_squares = np.zeros(hot_squares_len) 
        enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.ENEMY).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16))
            x = int(math.ceil((enemy_x[i] + 1) / 16))
            hot_squares[((y - 1) * 4) + (x - 1)] = 1
        if not self.base_top_left:  #en casao de que la base no inicie en la parte superior izquierda 
                                    #se voltea el arreglo para que siempre se perciva de la misma forma
                                    #el mapa
            hot_squares = hot_squares[::-1]
        for i in range(0, hot_squares_len): #Se agrega el arreglo de donde hay tropas enemigas a nuestro estado
            current_state[i + States_count] = hot_squares[i]
        #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
        #3.- Divición del minimapa en 16 partes iguales con valor de 1 si contienen nuestras tropas 
        #    si no su valor es cero
        green_squares = np.zeros(green_squares_len)        
        friendly_y, friendly_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero() 
        for i in range(0, len(friendly_y)):
            y = int(math.ceil((friendly_y[i] + 1) / 16))
            x = int(math.ceil((friendly_x[i] + 1) / 16))
            green_squares[((y - 1) * 2) + (x - 1)] = 1
        if not self.base_top_left:  #en casao de que la base no inicie en la parte superior izquierda 
                                    #se voltea el arreglo para que siempre se perciva de la misma forma
                                    #el mapa
            green_squares = green_squares[::-1]
        for i in range(0, green_squares_len): #Se agrega el arreglo de donde estan nuestras tropas a nuestro estado
            current_state[i + States_count + hot_squares_len] = green_squares[i]
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    #en caso de que nuestro Bot ya este tomando decisiones y se tenga una acción previa
    #decidida por el Bot se manda a llamar la función de aprendisaje para actualizar
    #la tabla Q, en la casilla correspondiente al estado y acción tomados.
        if self.previous_action is not None:
            reward = 0
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))
    #--------------------------------------------------------------------------
    #Dependiendo de las condiciones del estado en el que estemos 
    #limitaremos las acciones que puede tomar nuestro Bot
    #Lista de acciones excluidas
        excluded_actions = []
        #En caso de no tener SpawningPool, no tener espacio de población o 
        #no tener larvas disponibles no se crearan Zerglings (unidades de combate)
        if spawning_pools_count == 0 or free_supply == 0 and len(larvae) == 0 :
            excluded_actions.append(1)

        #Si ya detectamos unidades enemigas dejamos de explorar el mapa
        if len(enemy_y) > 0:
            excluded_actions.append(2) #Esta acción es Scout (se realiza con zerglings)
            
        #En caso de no tener Zerglings no se podran realizar ataques
        if army_count == 0 or len(enemy_y) == 0: 
            excluded_actions.append(3)
            excluded_actions.append(4)
            excluded_actions.append(5)        
            excluded_actions.append(6)
            excluded_actions.append(7)
            excluded_actions.append(8)
            excluded_actions.append(9)
            excluded_actions.append(10)
            excluded_actions.append(11)
            excluded_actions.append(12)
            excluded_actions.append(13)        
            excluded_actions.append(14)
            excluded_actions.append(15)
            excluded_actions.append(16)
            excluded_actions.append(17)
            excluded_actions.append(18)

        #En caso de tener Zerglings se atacara solo cuadrantes con unidades enemigas detectatdas
        if army_count > 0 and len(enemy_y) > 0: 
            for i in range(0, hot_squares_len):
                if hot_squares[i] == 0:
                    excluded_actions.append(i+3) #Se evitan las primeras 3 acciones que no son de ataque

    #--------------------------------------------------------------------------  
    #Decidiremos que acción tomar
        rl_action = self.qlearn.choose_action(str(current_state), excluded_actions)
        #Guardamos la acción como la previa para la segunda sección de nuestra 
        #"accion tomada" y el futuro posible aprendisaje en el siguiente paso
        self.previous_state = current_state
        self.previous_action = rl_action
        smart_action, attack_x, attack_y = self.splitAction(rl_action)
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#en caso de que se decidiera no hacer nada
        if smart_action == ACTION_DO_NOTHING:
           return actions.FUNCTIONS.no_op()
    #####--------------------------------------------------------------------------
    #En caso de haber dicidido crear un Zergling primero seleccionaremos una larva
    #que indicaremos se transformara en la 2 seccion de las acciones en Zergling 
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
    #en caso de haver decidido atacar o explorar, primero se seleccionaran las tropas disponibles (los zergling)
        elif smart_action == ACTION_ATTACK or smart_action == ACTION_Scout: 
            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")
    #####--------------------------------------------------------------------------     
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#Segunda sección de nuestra acción tomada
    elif self.move_number == 1:
        self.move_number = 0 #se resetea nuestra variable para en el siguiente paso tomar otra decición
        smart_action, attack_x, attack_y = self.splitAction(self.previous_action)
    #####--------------------------------------------------------------------------
    #en caso de que se decidiera no hacer nada
        if smart_action == ACTION_DO_NOTHING:
           return actions.FUNCTIONS.no_op()
    #####-------------------------------------------------------------------------- 
    #Indicamos que la larva seleccionada se transforme en Zergling
        elif smart_action == ACTION_BUILD_Zergling:
            if self.unit_type_is_selected(obs, units.Zerg.Larva):
                if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
                    return actions.FUNCTIONS.Train_Zergling_quick("now")       
    #####--------------------------------------------------------------------------
    #indicamos a los Zergling seleccionados que vayan a atacar a la ubicación 
    #(coordenadas X's y Y's) correspondiente a la decición de ataque tomada
        elif smart_action == ACTION_ATTACK: 
            if self.unit_type_is_selected(obs, units.Zerg.Zergling):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    x_offset = random.randint(-1, 1) #Se variara al asar la ubicación exacta
                    y_offset = random.randint(-1, 1) #para aumentar la posibilidad de encontrar al contrincante
                    return actions.FUNCTIONS.Attack_minimap("now", self.transformLocation((int(attack_x)+(x_offset*4)), (int(attack_y)+(y_offset*4))))  
    #####--------------------------------------------------------------------------
    #En caso de tener Zergling/s y no ver en el mapa tropas enemigas esta acción permite:
    #mandar a una ubicacion de entre 5 diferentes a nuestras tropas, 
    #una de estas ubicaciones es donde esperariamos encontrar la base enemiga (self.assault_coordinates)
        elif smart_action == ACTION_Scout: 
            if self.unit_type_is_selected(obs, units.Zerg.Zergling):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    scout_option = random.randint(0, 4)
                    if   scout_option == 0:                        
                        return actions.FUNCTIONS.Attack_minimap("now", [16, 16])
                    elif scout_option == 1:                        
                        return actions.FUNCTIONS.Attack_minimap("now", [16, 48])
                    elif scout_option == 2:                        
                        return actions.FUNCTIONS.Attack_minimap("now", [48, 16])
                    elif scout_option == 3:                        
                        return actions.FUNCTIONS.Attack_minimap("now", [48, 48])
                    elif scout_option == 4:                        
                        return actions.FUNCTIONS.Attack_minimap("now", self.assault_coordinates)                   
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#En caso de llegar aquí regresamos al juego que en este paso el Agente no haga nada
    return actions.FUNCTIONS.no_op()
'''
_______________________________________________________________________________
'''
def main(unused_argv): #la sección principal, 
                       #desde donde se configuran los parametros de la partida en turno
  agent = ZergAgent()
  try:
    while True:
      with sc2_env.SC2Env( #Se manda a llamar el juego
          map_name="AbyssalReef", #Se indica en que mapa se realizara la partida
          players=[sc2_env.Agent(sc2_env.Race.zerg), #Un jugador sera nuestro Bot de raza Zerg
                   sc2_env.Bot(sc2_env.Race.random,            #2° jugador sera el Bot del juego 
                               sc2_env.Difficulty.very_easy)], #en dificultad very_easy
          agent_interface_format=features.AgentInterfaceFormat(
                  #dimenciones de pantalla y minimapa 
              feature_dimensions=features.Dimensions(screen=84, minimap=64), 
              use_feature_units=True),
          step_mul=16,
          game_steps_per_episode=0, #con 0 indicamos que el juego no termine en un numero determinado de pasos
          visualize=False) as env:  #seleccionamos que no se visualicen las capas proporcionadas por pysc2
         
        agent.setup(env.observation_spec(), env.action_spec())
       
        timesteps = env.reset()
        agent.reset()
       
        while True:
            #durante cada juego a cada paso se manda a llamar la sección "step" de nuestro Bot
          step_actions = [agent.step(timesteps[0])] 
          if timesteps[0].last():
            break #si el juego termina salimos del bucle
            #pasamos al juego las acciones (o partes de estas) que tomo nuestro Bot
          timesteps = env.step(step_actions) 
     
  except KeyboardInterrupt:
    pass

#esta sección es la que manda a llamar al main, la encargada de que todo empiece a correr 
if __name__ == "__main__":
  app.run(main)
