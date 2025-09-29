import gymnasium as gym
from gymnasium import spaces
import pygame
from gymnasium.envs.registration import register
from enum import Enum
import numpy as np

# Register the module as gym env
register(
    id='airplane-boarding-v0',       # id is usable in gym.make()
    entry_point='airplane_boarding:AirplaneEnv'
)

class PassengerStatus(Enum):
    MOVING = 0
    STALLED = 1
    STOWING = 2
    SEATED = 3

    def __str__(self):
        match self:
            case PassengerStatus.MOVING:
                return "MOVING"
            case PassengerStatus.STALLED:
                return "STALLED"
            case PassengerStatus.STOWING:
                return "STOWING"
            case PassengerStatus.SEATED:
                return "SEATED"

class Passenger:
    def __init__(self, seat_num, row_num):
        self.seat_num = seat_num
        self.row_num = row_num
        self.is_holding_luggage = True
        self.status = PassengerStatus.MOVING
    
    def __str__(self):
        return f"P{self.seat_num:02d}"

class LobbyRow:
    def __init__(self, row_num, seats_per_row):
        self.row_num = row_num
        self.passengers = [Passenger(row_num * seats_per_row + i, row_num) for i in range(seats_per_row)]

class Lobby:
    def __init__(self, num_of_rows, seats_per_row):
        self.num_of_rows = num_of_rows
        self.seats_per_row = seats_per_row
        self.lobby_rows = [LobbyRow(row_num, self.seats_per_row) for row_num in range(self.num_of_rows)]

    def remove_passenger(self, row_num):
        passenger = self.lobby_rows[row_num].passengers.pop()
        return passenger

    def count_passengers(self):
        return sum(len(row.passengers) for row in self.lobby_rows)

class BoardingLine:
    def __init__(self, num_of_rows):
        self.num_of_rows = num_of_rows
        self.line = [None for _ in range(num_of_rows)]
    
    def add_passenger(self, passenger):
        self.line.append(passenger)
    
    def is_onboarding(self):
        return any(passenger is not None for passenger in self.line)
    
    def num_passengers_stalled(self):
        return sum(1 for passenger in self.line if passenger is not None and passenger.status == PassengerStatus.STALLED)
    
    def num_passengers_moving(self):
        return sum(1 for passenger in self.line if passenger is not None and passenger.status == PassengerStatus.MOVING)
    
    def move_forward(self):
        # Snapshot to avoid cascading moves
        old_line = list(self.line)
        new_line = list(self.line)

        for i, passenger in enumerate(old_line):
            if passenger is None or i == 0 or passenger.status == PassengerStatus.STOWING:
                continue

            if passenger.status in (PassengerStatus.MOVING, PassengerStatus.STALLED):
                if old_line[i - 1] is None:
                    passenger.status = PassengerStatus.MOVING
                    new_line[i - 1] = passenger
                    new_line[i] = None
                else:
                    passenger.status = PassengerStatus.STALLED

        self.line = new_line

        # Trim trailing empty slots
        for j in range(len(self.line) - 1, self.num_of_rows - 1, -1):
            if self.line[j] is None:
                self.line.pop(j)

class Seat:
    def __init__(self, seat_num, row_num):
        self.seat_num = seat_num
        self.row_num = row_num
        self.passenger = None
    
    def seat_passenger(self, passenger: Passenger):
        assert self.seat_num == passenger.seat_num, "Seat number mismatch!"

        if passenger.is_holding_luggage:
            passenger.status = PassengerStatus.STOWING
            passenger.is_holding_luggage = False
            return False
        else:
            self.passenger = passenger
            self.passenger.status = PassengerStatus.SEATED
            return True
        
    def __str__(self):
        return f"P{self.seat_num:02d}" if self.passenger else f"S{self.seat_num:02d}"

class AirplaneRow:
    def __init__(self, row_num, seats_per_row):
        self.row_num = row_num
        self.seats = [Seat(row_num*seats_per_row+i, row_num) for i in range(seats_per_row)]
    
    def try_sit_passenger(self, passenger: Passenger):
        found = next((seat for seat in self.seats if seat.seat_num == passenger.seat_num), None)
        return found.seat_passenger(passenger) if found else False

class AirplaneEnv(gym.Env):
    metadata = {'render_modes': ['human', 'terminal'], 'render_fps':1}

    def __init__(self, render_mode=None, num_of_rows=10, seats_per_row=5):
        self.seats_per_row = seats_per_row
        self.num_of_rows = num_of_rows
        self.num_of_seats = num_of_rows * seats_per_row
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Airplane Boarding Simulation")
            self.SEAT_SIZE, self.PADDING, self.AISLE_WIDTH = 40, 10, 50
            self.FONT_SIZE, self.LEGEND_HEIGHT = 18, 160
            screen_width = self.AISLE_WIDTH + (self.seats_per_row * (self.SEAT_SIZE + self.PADDING))
            screen_height = (self.num_of_rows * (self.SEAT_SIZE + self.PADDING)) + self.LEGEND_HEIGHT
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, self.FONT_SIZE)
            self.COLORS = {
                "background": (240, 240, 240),
                "seat_empty": (180, 180, 180),
                "seat_occupied": (100, 100, 100),
                "aisle": (210, 210, 210),
                PassengerStatus.MOVING: (70, 180, 70),
                PassengerStatus.STALLED: (220, 50, 50),
                PassengerStatus.STOWING: (250, 150, 50),
                "text": (0, 0, 0),
            }

        self.action_space = spaces.Discrete(self.num_of_rows)
        self.observation_space = spaces.Box(
            low=-1,
            high=self.num_of_seats - 1,
            shape=(self.num_of_seats * 2,),
            dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.airplane_rows = [AirplaneRow(r, self.seats_per_row) for r in range(self.num_of_rows)]
        self.lobby = Lobby(self.num_of_rows, self.seats_per_row)
        self.boarding_line = BoardingLine(self.num_of_rows)
        self.render()
        return self._get_observation(), {}
    
    def _get_observation(self):
        obs = []
        for passenger in self.boarding_line.line:
            if passenger is None:
                obs += [-1, -1]
            else:
                obs += [passenger.seat_num, passenger.status.value]
        while len(obs) < self.num_of_seats * 2:
            obs += [-1, -1]
        return np.array(obs, dtype=np.int32)
    
    def step(self, row_num):
        assert 0 <= row_num < self.num_of_rows
        reward = 0
        passenger = self.lobby.remove_passenger(row_num)
        self.boarding_line.add_passenger(passenger)

        if self.lobby.count_passengers() > 0:
            self._move()
            reward = self._calculate_reward()
        else:
            while self.is_onboarding():
                self._move()
                reward += self._calculate_reward()

        terminated = not self.is_onboarding()
        return self._get_observation(), reward, terminated, False, {}
    
    def _calculate_reward(self):
        return -self.boarding_line.num_passengers_stalled() + self.boarding_line.num_passengers_moving()
    
    def is_onboarding(self):
        return self.lobby.count_passengers() > 0 or self.boarding_line.is_onboarding()
    
    def _move(self):
        for r, passenger in enumerate(self.boarding_line.line):
            if passenger is None:
                continue
            if r >= len(self.airplane_rows):
                break
            if self.airplane_rows[r].try_sit_passenger(passenger):
                self.boarding_line.line[r] = None
        self.boarding_line.move_forward()
        self.render()
    
    def render(self):
        if self.render_mode is None:
            return
        if self.render_mode == 'terminal':
            self._render_terminal()
        elif self.render_mode == 'human':
            self._render_human()

    def _render_human(self):
        if self.screen is None: return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return
        self.screen.fill(self.COLORS["background"])
        aisle_x = self.seats_per_row // 2 * (self.SEAT_SIZE + self.PADDING)
        pygame.draw.rect(self.screen, self.COLORS["aisle"],
                         (aisle_x, 0, self.AISLE_WIDTH, self.num_of_rows * (self.SEAT_SIZE + self.PADDING)))
        for r_idx, row in enumerate(self.airplane_rows):
            for s_idx, seat in enumerate(row.seats):
                seat_x = s_idx * (self.SEAT_SIZE + self.PADDING)
                if s_idx >= self.seats_per_row // 2:
                    seat_x += self.AISLE_WIDTH
                seat_y = r_idx * (self.SEAT_SIZE + self.PADDING)
                color = self.COLORS["seat_occupied"] if seat.passenger else self.COLORS["seat_empty"]
                pygame.draw.rect(self.screen, color, (seat_x, seat_y, self.SEAT_SIZE, self.SEAT_SIZE), border_radius=5)
                text = self.font.render(f"S{seat.seat_num:02d}", True, self.COLORS["text"])
                self.screen.blit(text, text.get_rect(center=(seat_x + self.SEAT_SIZE/2, seat_y + self.SEAT_SIZE/2)))
        for i, passenger in enumerate(self.boarding_line.line):
            if passenger and i < self.num_of_rows:
                x = aisle_x + self.AISLE_WIDTH / 2
                y = i * (self.SEAT_SIZE + self.PADDING) + self.SEAT_SIZE/2
                pygame.draw.circle(self.screen, self.COLORS[passenger.status], (x, y), self.SEAT_SIZE/2 - 2)
                text = self.font.render(f"P{passenger.seat_num:02d}", True, self.COLORS["text"])
                self.screen.blit(text, text.get_rect(center=(x, y)))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _render_terminal(self):
        print("Seats".center(19) + " | Aisle Line")
        for row in self.airplane_rows:
            for seat in row.seats:
                print(seat, end=" ")
            if row.row_num < len(self.boarding_line.line):
                passenger = self.boarding_line.line[row.row_num]
                status = "" if passenger is None else passenger.status
                print(f"| {passenger} {status}", end=" ")
            print()
        print("\nLobby:")
        for row in self.lobby.lobby_rows:
            for passenger in row.passengers:
                print(passenger, end=" ")
            if row.passengers: print()
        print("\n")

    def close(self):
        if self.screen is not None:
            pygame.display.quit(); pygame.quit(); self.screen = None

    def action_masks(self) -> list[bool]:
        return [len(row.passengers) > 0 for row in self.lobby.lobby_rows]

def check_my_env():
    from gymnasium.utils.env_checker import check_env
    env = gym.make('airplane-boarding-v0', render_mode=None)
    check_env(env.unwrapped)

if __name__ == "__main__":
    env = gym.make('airplane-boarding-v0', num_of_rows=10, seats_per_row=5, render_mode='human')
    obs, _ = env.reset()
    done, total_reward, step_count = False, 0, 0
    while not done:
        action = env.action_space.sample()
        masks = env.unwrapped.action_masks()
        if not masks[action]: continue
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        step_count += 1
        print(f"Step {step_count} Action: {action} Reward: {reward}")
    env.close()
    print(f"Total Reward: {total_reward}")
