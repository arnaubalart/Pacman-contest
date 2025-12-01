# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        food_list = self.get_food(successor).as_list()
        capsules = self.get_capsules(successor)
        features['successor_score'] = -len(food_list)
        opponents = self.get_opponents(game_state)

        #Compute distance to the nearest normal food
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        
        #compute distance to the nearest power capsule
        if len(capsules) > 0:
            min_distance_c = min([self.get_maze_distance(my_pos, c) for c in capsules])
            features['distance_to_capsules'] = min_distance_c

        #evitamos pararnos
        if action == Directions.STOP:
            features['stop'] = 1

        #comprobamos si hay algun enemigo cerca
        enemies = [game_state.get_agent_state(o) for o in opponents]
        visible = [e for e in enemies if e.get_position() is not None]
        scared = any(e.scared_timer > 0 for e in visible)

        # Evitamos cruzar frontera si hay algun enemigo cerca
        my_old = game_state.get_agent_position(self.index)
        is_red = self.red
        mid = game_state.get_walls().width // 2

        def enemy_side(pos):
            return pos[0] >= mid if is_red else pos[0] < mid

        if my_old is None or my_pos is None:
            features['dangerous_entry'] = 0
        else:
            crossing = (not enemy_side(my_old)) and enemy_side(my_pos)

        # Inicializar enemy_distance para evitar KeyError antes de usarla
        if len(visible) > 0:
            dists = [self.get_maze_distance(my_pos, e.get_position()) for e in visible]
            nearest = min(dists)
        else:
            nearest = 9999
        features['enemy_distance'] = nearest

        if crossing and features['enemy_distance'] < 5:
            features['dangerous_entry'] = 1
        else:
            features['dangerous_entry'] = 0
        
        if scared or not enemy_side(my_pos):
            features['enemy_distance'] = 100
            features['danger'] = 0
            features['dead_end_risk'] = 0
        else:
            features['danger'] = 1 if nearest <= 6 else 0

        carrying = successor.get_agent_state(self.index).num_carrying
        features["carrying_food"] = carrying

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100, 
            'distance_to_food': -2.5,
            'distance_to_capsules': -25,
            'enemy_distance': 4,
            'danger': -80,
            'stop': -20,
            'carrying_food': 10,
            }

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.last_food = []
        self.target = None

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        # guardamos la comida inicial para comparar despues
        self.last_food = self.get_food_you_are_defending(game_state).as_list()

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        #no queremos ser pacman, queremos ser fantasma 
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Miramos si hay enemigos visibles
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            # Si hay invasores, vamos a por ellos
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:           
            #Miramos si nos han comido comida
            current_food = self.get_food_you_are_defending(successor).as_list()
            if len(current_food) < len(self.last_food):
                # Calcular la diferencia para saber donde ir
                diff = set(self.last_food) - set(current_food)
                if len(diff) > 0:
                    self.target = list(diff)[0]
            self.last_food = current_food

            # Si tenemos un objetivo de comida desaparecida, vamos alli
            if self.target is not None:
                features['investigate_distance'] = self.get_maze_distance(my_pos, self.target)
            
            # Si hemos llegado o no hay objetivo, patrullamos el centro
            if self.target is None or my_pos == self.target:
                self.target = None
                #Buscamos la comida mas cercana al centro para protegerla
                mid_x = game_state.data.layout.width // 2
                best_dist = 9999
                best_food = None
                for food in current_food:
                    dist_to_center = abs(food[0] - mid_x)
                    if dist_to_center < best_dist:
                        best_dist = dist_to_center
                        best_food = food
                
                if best_food:
                    features['center_patrol'] = self.get_maze_distance(my_pos, best_food)

        #Si somos pacman por error, volvemos
        if my_state.is_pacman:
            features['retreat'] = 1
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        weights = {
            'num_invaders': -1000,
            'invader_distance': -10,
            'on_defense': 100,
            'center_patrol': -2,
            'investigate_distance': -5,
            'retreat': -100,  #Importante para no quedarnos atascados en el lado enemigo
            'stop': -100,
            'reverse': -2,
        }
        
        # Si estamos asustados no perseguimos tanto
        my_state = game_state.get_agent_state(self.index)
        if my_state.scared_timer > 0:
            weights['invader_distance'] = -1 
            weights['num_invaders'] = -100
            
        return weights
