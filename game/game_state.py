
import numpy as np

def game_board_state_generator(h, w):
    while(True):
        game_board = np.zeros(shape = [h * w, 3])
        player_pos, enemy_pos = np.random.choice(np.arange(h * w), 2, False)
        game_board[player_pos, 1] = 1
        game_board[enemy_pos, 0] = 1
        game_board = game_board.reshape(h,w,3)
        yield game_board

def create_player_action(game_board):
    h, w, _ = game_board.shape
    enemy_pos = np.unravel_index(np.argmax(game_board[:,:,0]), [h,w])
    player_pos = np.unravel_index(np.argmax(game_board[:,:,1]), [h,w])

    move_vec = np.array(enemy_pos) - np.array(player_pos)
    distance = np.linalg.norm(move_vec)
    if distance > 1.5:
        move_vec = (move_vec / distance * 1.5).astype(int)
        target_pos = player_pos + move_vec  
    else: 
        move_vec = np.array([0,0])
    move_id = move_vec[0] * 3 + move_vec[1] + 4
    move_action = np.zeros(9)
    move_action[move_id]=1
    return(move_action)

def create_data_for_ai(h, w):
    game_board_generator = game_board_state_generator(h, w)
    while(True):
        game_board = next(game_board_generator)
        player_action = create_player_action(game_board)
        yield game_board.reshape(1, h, w, 3), player_action.reshape(1,9)

def batch_data_for_ai(h, w, batch_size = 100):
    data_generator = create_data_for_ai(h, w)
    while(True):
        X, y = next(data_generator)
        for i in range(batch_size - 1):
            new_X, new_y = next(data_generator)
            X = np.append(X, new_X, 0)
            y = np.append(y, new_y, 0)
        yield X, y
        
class Game_Environment:
    action_space = range(9)
    
    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.d = 1
        self.reset()
        
    def reset(self): # randomize new initial state
        self.reward = 0
        state = np.zeros(shape = [self.h * self.w, 3])
        player_pos = np.random.choice(np.arange(self.h * self.w))
        state[player_pos, 1] = 1
        self.player_pos = np.array([player_pos // self.w, player_pos % self.w])
        self.state = state.reshape(self.h, self.w, 3)
        enemy_pos = [-1, -1]
        
        while (enemy_pos[0] < 0) | (enemy_pos[0] >= self.w) | (enemy_pos[1] < 0) | (enemy_pos[1] >= self.h) | np.array_equal(enemy_pos, self.player_pos):
            enemy_pos[0] = self.player_pos[0] + np.random.randint(-self.d, self.d + 1)
            enemy_pos[1] = self.player_pos[1] + np.random.randint(-self.d, self.d + 1)  
        self.state[enemy_pos[0], enemy_pos[1], 0] = 1
        self.enemy_pos = enemy_pos
        self.d = min(self.d + 1, self.h, self.w)
        self.done = False

    
    def step(self, action): 
        new_player_pos = self.player_pos + np.array([(action // 3) - 1, (action % 3) - 1])
        
        if (new_player_pos[0] == self.w) | (new_player_pos[0] < 0) | (new_player_pos[1] == self.h) | (new_player_pos[1] < 0) | (action == 4):
            reward = -1

        else:
            self.state[self.player_pos[0], self.player_pos[1], 1] = 0
            self.state[new_player_pos[0], new_player_pos[1], 1] = 1
            reward = -0.05 * (3 - action % 2)
            self.player_pos = new_player_pos
            
            if np.array_equal(new_player_pos, self.enemy_pos):
                self.done = True
                reward = 1
        
        return reward
        
        