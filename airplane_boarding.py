
import gymnasium as gym
from gymnasium import spaces
import pygame
from gymnasium.envs.registration import register
from enum import Enum
import numpy as np
from boarding_strategies import make_env, random_strategy, back_to_front, front_to_back, wilma

# Register the module as gym env
register(
    id='airplane-boarding-v0',
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
        return self.lobby_rows[row_num].passengers.pop()

    def count_passengers(self):
        return sum(len(row.passengers) for row in self.lobby_rows)

class BoardingLine:
    def __init__(self, num_of_rows):
        self.num_of_rows = num_of_rows
        self.line = [None for _ in range(num_of_rows)]

    def add_passenger(self, passenger):
        self.line.append(passenger)

    def is_onboarding(self):
        return len(self.line) > 0 and not all(p is None for p in self.line)

    def num_passengers_stalled(self):
        return sum(1 for p in self.line if p and p.status == PassengerStatus.STALLED)

    def num_passengers_moving(self):
        return sum(1 for p in self.line if p and p.status == PassengerStatus.MOVING)

    def move_forward(self):
        for i, passenger in enumerate(self.line):
            if passenger is None or i == 0 or passenger.status == PassengerStatus.STOWING:
                continue
            if (passenger.status in [PassengerStatus.MOVING, PassengerStatus.STALLED]) and self.line[i-1] is None:
                passenger.status = PassengerStatus.MOVING
                self.line[i-1] = passenger
                self.line[i] = None
            else:
                passenger.status = PassengerStatus.STALLED

        for i in range(len(self.line)-1, self.num_of_rows-1, -1):
            if self.line[i] is None:
                self.line.pop(i)

class Seat:
    def __init__(self, seat_num, row_num):
        self.seat_num = seat_num
        self.row_num = row_num
        self.passenger = None

    def seat_passenger(self, passenger):
        assert self.seat_num == passenger.seat_num
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
        self.seats = [Seat(row_num * seats_per_row + i, row_num) for i in range(seats_per_row)]

    def try_sit_passenger(self, passenger):
        found_seats = [s for s in self.seats if s.seat_num == passenger.seat_num]
        if found_seats:
            return found_seats[0].seat_passenger(passenger)
        return False

class AirplaneEnv(gym.Env):
    metadata = {'render_modes': ['human', 'terminal'], 'render_fps': 1}

    def __init__(self, render_mode=None, num_of_rows=10, seats_per_row=5):
        self.seats_per_row = seats_per_row
        self.num_of_rows = num_of_rows
        self.num_of_seats = num_of_rows * seats_per_row

        self.render_mode = render_mode
        self.screen = self.clock = None
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
                "background": (240, 240, 240), "seat_empty": (180, 180, 180),
                "seat_occupied": (100, 100, 100), "aisle": (210, 210, 210),
                PassengerStatus.MOVING: (70, 180, 70), PassengerStatus.STALLED: (220, 50, 50),
                PassengerStatus.STOWING: (250, 150, 50), "text": (0, 0, 0),
            }

        self.action_space = spaces.Discrete(self.num_of_rows)
        self.observation_space = spaces.Box(low=-1, high=self.num_of_seats - 1, shape=(self.num_of_seats * 2,), dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.airplane_rows = [AirplaneRow(row_num, self.seats_per_row) for row_num in range(self.num_of_rows)]
        self.lobby = Lobby(self.num_of_rows, self.seats_per_row)
        self.boarding_line = BoardingLine(self.num_of_rows)
        self.render()
        return self._get_observation(), {}

    def _get_observation(self):
        observation = []
        for passenger in self.boarding_line.line:
            if passenger is None:
                observation += [-1, -1]
            else:
                observation += [passenger.seat_num, passenger.status.value]
        while len(observation) < self.num_of_seats * 2:
            observation += [-1, -1]
        return np.array(observation, dtype=np.int32)

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

        return self._get_observation(), reward, not self.is_onboarding(), False, {}

    def _calculate_reward(self):
        return -self.boarding_line.num_passengers_stalled() + self.boarding_line.num_passengers_moving()

    def is_onboarding(self):
        return self.lobby.count_passengers() > 0 or self.boarding_line.is_onboarding()

    def _move(self):
        for i, passenger in enumerate(self.boarding_line.line):
            if passenger and i < self.num_of_rows:
                if self.airplane_rows[i].try_sit_passenger(passenger):
                    self.boarding_line.line[i] = None
        self.boarding_line.move_forward()
        self.render()

    def render(self):
        if self.render_mode == "terminal":
            self._render_terminal()
        elif self.render_mode == "human":
            self._render_human()

    def close(self):
        if self.screen:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def action_masks(self):
        return [bool(row.passengers) for row in self.lobby.lobby_rows]

    def _render_terminal(self):
        print("Seats".center(19) + " | Aisle Line")
        for row in self.airplane_rows:
            print(" ".join(str(seat) for seat in row.seats), end=" ")
            if row.row_num < len(self.boarding_line.line):
                passenger = self.boarding_line.line[row.row_num]
                print(f"| {passenger} {passenger.status}" if passenger else "", end="")
            print()
        print("\nLine entering plane:")
        for passenger in self.boarding_line.line[self.num_of_rows:]:
            if passenger:
                print(f"{passenger} {passenger.status}")
        print("\nLobby:")
        for row in self.lobby.lobby_rows:
            print(" ".join(str(p) for p in row.passengers))

    def _render_human(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        self.screen.fill(self.COLORS["background"])
        aisle_x = self.seats_per_row // 2 * (self.SEAT_SIZE + self.PADDING)
        pygame.draw.rect(self.screen, self.COLORS["aisle"],
                         (aisle_x, 0, self.AISLE_WIDTH, self.num_of_rows * (self.SEAT_SIZE + self.PADDING)))

        for r_idx, row in enumerate(self.airplane_rows):
            for s_idx, seat in enumerate(row.seats):
                x = s_idx * (self.SEAT_SIZE + self.PADDING)
                if s_idx >= self.seats_per_row // 2:
                    x += self.AISLE_WIDTH
                y = r_idx * (self.SEAT_SIZE + self.PADDING)
                color = self.COLORS["seat_occupied"] if seat.passenger else self.COLORS["seat_empty"]
                pygame.draw.rect(self.screen, color, (x, y, self.SEAT_SIZE, self.SEAT_SIZE), border_radius=5)
                text = self.font.render(str(seat), True, self.COLORS["text"])
                self.screen.blit(text, text.get_rect(center=(x + self.SEAT_SIZE / 2, y + self.SEAT_SIZE / 2)))

        for i, passenger in enumerate(self.boarding_line.line[:self.num_of_rows]):
            if passenger:
                px = aisle_x + self.AISLE_WIDTH / 2
                py = i * (self.SEAT_SIZE + self.PADDING) + self.SEAT_SIZE / 2
                pygame.draw.circle(self.screen, self.COLORS[passenger.status], (px, py), self.SEAT_SIZE / 2 - 2)
                text = self.font.render(str(passenger), True, self.COLORS["text"])
                self.screen.blit(text, text.get_rect(center=(px, py)))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "back", "front", "wilma"],
                        help="Choose boarding strategy: random, back, front, wilma")
    args = parser.parse_args()

    env = make_env(render_mode='human')

    if args.strategy == "random":
        steps, reward = random_strategy(env)
    elif args.strategy == "back":
        steps, reward = back_to_front(env)
    elif args.strategy == "front":
        steps, reward = front_to_back(env)
    elif args.strategy == "wilma":
        steps, reward = wilma(env)

    print(f"\nStrategy '{args.strategy}' finished in {steps} steps with total reward {reward}")
    env.close()

