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

    # String representation of the PassengerStatus enum
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
    
    # String representation of Passenger class, ie, seat number - 2 digit
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
        count = 0
        for row in self.lobby_rows:
            count += len(row.passengers)
        
        return count

class BoardingLine:
    def __init__(self, num_of_rows):
        # Initialize the aisle
        self.num_of_rows = num_of_rows
        self.line = [None for i in range(num_of_rows)]
    
    def add_passenger(self, passenger):
        self.line.append(passenger)
    
    def is_onboarding(self):
        if(len(self.line) > 0 and not all(passenger is None for passenger in self.line)):     # error - Passenger instead of passenger
            return True
        
        return False

    def num_congested_positions(self):
        # NEW METHOD: count pairs of passengers directly behind each other
        count = 0
        for i in range(1, len(self.line)):
            if self.line[i] is not None and self.line[i-1] is not None:
                count += 1
        return count
        
    def num_passengers_stalled(self):
        count = 0
        for passenger in self.line:
            if passenger is not None and passenger.status == PassengerStatus.STALLED:
                count += 1
            
        return count
    
    def num_passengers_moving(self):
        count = 0
        for passenger in self.line:
            if passenger is not None and passenger.status == PassengerStatus.MOVING:
                count += 1
            
        return count

    def num_passengers_seated(self):
        count = 0
        for passenger in self.line:
            if passenger is not None and passenger.status == PassengerStatus.SEATED:
                count += 1
        return count
    
    def move_forward(self):

        for i, passenger in enumerate(self.line):
            # Skip -> no passenger is in that spot or
            # passenger is at front of line or
            # passenger is stowing luggage
            if passenger is None or i == 0 or passenger.status == PassengerStatus.STOWING:
                continue

            # Move if no one is blocking
            if(passenger.status == PassengerStatus.STALLED or passenger.status == PassengerStatus.MOVING) and self.line[i-1] is None:
                passenger.status = PassengerStatus.MOVING
                self.line[i-1] = passenger
                self.line[i] = None
            else:
                passenger.status = PassengerStatus.STALLED
        
        # Truncate the empty spots at teh end of line
        for i in range(len(self.line)-1, self.num_of_rows-1, -1):
            if self.line[i] is None:
                self.line.pop(i)

class Seat:
    def __init__(self, seat_num, row_num):
        self.seat_num = seat_num
        self.row_num = row_num
        self.passenger = None
    
    # Attempt to sit passenger
    def seat_passenger(self, passenger: Passenger):

        assert self.seat_num == passenger.seat_num, "Seat number does not match Passenger's seat number"

        if passenger.is_holding_luggage:
            # passenger starts stowing 
            passenger.status = PassengerStatus.STOWING
            passenger.is_holding_luggage = False
            return False
        else:
            # seat the passenger
            self.passenger = passenger
            self.passenger.status = PassengerStatus.SEATED
            return True
        
    def __str__(self):
        if self.passenger is None:
            return f"S{self.seat_num:02d}"
        else:
            return f"P{self.seat_num:02d}"
        
class AirplaneRow:
    def __init__(self, row_num, seats_per_row):
        self.row_num = row_num
        self.seats = [Seat(row_num*seats_per_row+i, row_num) for i in range(seats_per_row)]
    
    def try_sit_passenger(self, passenger:Passenger):
        # Check if the passenger's seat is in this row
        found_seats = list(filter(lambda seats: seats.seat_num == passenger.seat_num, self.seats))

        if found_seats:
            found_seat: Seat = found_seats[0]
            return found_seat.seat_passenger(passenger)
        
        return False
    
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

            # Define constants for drawing
            self.SEAT_SIZE = 40
            self.PADDING = 10
            self.AISLE_WIDTH = 50
            self.FONT_SIZE = 18
            self.LEGEND_HEIGHT = 160

            # Calculate screen dimensions
            screen_width = self.AISLE_WIDTH + (self.seats_per_row * (self.SEAT_SIZE + self.PADDING))
            screen_height = (self.num_of_rows * (self.SEAT_SIZE + self.PADDING)) + self.LEGEND_HEIGHT
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, self.FONT_SIZE)

            # Colors
            self.COLORS = {
                "background": (240, 240, 240),
                "seat_empty": (180, 180, 180),
                "seat_occupied": (100, 100, 100),
                "aisle": (210, 210, 210),
                PassengerStatus.MOVING: (70, 180, 70),  # Green
                PassengerStatus.STALLED: (220, 50, 50),  # Red
                PassengerStatus.STOWING: (250, 150, 50),  # Orange
                "text": (0, 0, 0),
            }
        # Drive the Action space
        self.action_space = spaces.Discrete(self.num_of_rows)

        # Define the Observation space
        # The observation space is used to validate the observation returned by reset() and step()
        # [0, -1, 1, -1, 2, -1, ..., 6, 2, 7, 1, ...]
        self.observation_space = spaces.Box(
            low = -1,
            high = self.num_of_seats - 1,
            shape = (self.num_of_seats * 2,),
            dtype = np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)   # gym reuires this call to control randomness and reproduce scenarios

        self.airplane_rows = [AirplaneRow(row_num, self.seats_per_row) for row_num in range(self.num_of_rows)]
        self.lobby = Lobby(self.num_of_rows, self.seats_per_row)
        self.boarding_line = BoardingLine(self.num_of_rows)

        self.render()

        return self._get_observation(), {}
    
    # Return the array of the number of passengers in line
    def _get_observation(self):
        observation = []
        
        for passenger in self.boarding_line.line:
            if passenger is None:
                observation.append(-1)
                observation.append(-1)
            else:
                observation.append(passenger.seat_num)
                observation.append(passenger.status.value)

        for i in range(len(self.boarding_line.line), self.num_of_seats):
            observation.append(-1)
            observation.append(-1)
        
        return np.array(observation, dtype=np.int32)
    
    def step(self, row_num):
        assert row_num >= 0 and row_num < self.num_of_rows, f"Invalid row number {row_num}"

        reward = 0
        passenger = self.lobby.remove_passenger(row_num)
        self.boarding_line.add_passenger(passenger)

        # if there are passengers in the lobby, mobe the line once
        if self.lobby.count_passengers() > 0:
            self._move()
            reward = self._calculate_reward()
        else:
            # No more passengers in the lobby, so no more actions to choose from, move the line unitl all the passengers are seated
            while self.is_onboarding():
                self._move()
                reward += self._calculate_reward()

        if self.is_onboarding():
            terminated = False
        else:
            terminated = True

        # Gym requires the observation, reward, terminated, truncated and info dictionary
        return self._get_observation(), reward, terminated, False, {}
    
     def _calculate_reward(self):
        moving = self.boarding_line.num_passengers_moving()
        stalled = self.boarding_line.num_passengers_stalled()
        seated = self.boarding_line.num_passengers_seated()  # NEW: Count seated passengers
        congested = self.boarding_line.num_congested_positions()  # NEW: Count congestion (passengers directly behind each other)

        reward = 0
        reward += 1.0 * moving                # Encourage moving
        reward -= 1.0 * stalled               # Penalize stalling
        reward += 2.0 * seated                # Reward being seated
        reward -= 2.0 * congested             # Penalize congestion (NEW)

        if not self.is_onboarding():         # NEW: Bonus for finishing onboarding
            reward += 50.0

        return reward

    def is_onboarding(self):
        # If there are passengers in the lobby or in the boarding line, return True
        if self.lobby.count_passengers() > 0 or self.boarding_line.is_onboarding():
            return True
        
        return False
    
    def _move(self):

        for row_num, passenger in enumerate(self.boarding_line.line):
            if passenger is None:
                continue
            # If outside of airplane's aisle
            if row_num >= len(self.airplane_rows):
                break
            # Try to sit passenger, if successful, remove from line
            if self.airplane_rows[row_num].try_sit_passenger(passenger):
                self.boarding_line.line[row_num] = None
        
        # Move line forward
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
        if self.screen is None:
            return

        # Handle window close event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # Clear the ENTIRE screen once at the beginning
        self.screen.fill(self.COLORS["background"])

        # ... (The existing code for drawing the aisle, seats, and passengers in the aisle goes here. No changes needed in that part.)
        aisle_x_start = self.seats_per_row // 2 * (self.SEAT_SIZE + self.PADDING)
        pygame.draw.rect(self.screen, self.COLORS["aisle"],
                         (aisle_x_start, 0, self.AISLE_WIDTH, self.num_of_rows * (self.SEAT_SIZE + self.PADDING)))
        for r_idx, row in enumerate(self.airplane_rows):
            for s_idx, seat in enumerate(row.seats):
                seat_x = (s_idx * (self.SEAT_SIZE + self.PADDING))
                if s_idx >= self.seats_per_row // 2:
                    seat_x += self.AISLE_WIDTH
                seat_y = r_idx * (self.SEAT_SIZE + self.PADDING)
                color = self.COLORS["seat_occupied"] if seat.passenger else self.COLORS["seat_empty"]
                pygame.draw.rect(self.screen, color, (seat_x, seat_y, self.SEAT_SIZE, self.SEAT_SIZE), border_radius=5)
                text_surf = self.font.render(f"S{seat.seat_num:02d}", True, self.COLORS["text"])
                text_rect = text_surf.get_rect(center=(seat_x + self.SEAT_SIZE / 2, seat_y + self.SEAT_SIZE / 2))
                self.screen.blit(text_surf, text_rect)
        for i, passenger in enumerate(self.boarding_line.line):
            if passenger is not None and i < self.num_of_rows:
                passenger_x = aisle_x_start + self.AISLE_WIDTH / 2
                passenger_y = i * (self.SEAT_SIZE + self.PADDING) + self.SEAT_SIZE / 2
                pygame.draw.circle(self.screen, self.COLORS[passenger.status], (passenger_x, passenger_y),
                                   self.SEAT_SIZE / 2 - 2)
                text_surf = self.font.render(f"P{passenger.seat_num:02d}", True, self.COLORS["text"])
                text_rect = text_surf.get_rect(center=(passenger_x, passenger_y))
                self.screen.blit(text_surf, text_rect)

        # --- CODE TO DRAW THE LEGEND ---
        legend_y_start = self.num_of_rows * (self.SEAT_SIZE + self.PADDING)
        pygame.draw.line(self.screen, (150, 150, 150), (0, legend_y_start), (self.screen.get_width(), legend_y_start),
                         2)

        legend_items = [
            (PassengerStatus.MOVING, "Passenger: Moving"),
            (PassengerStatus.STALLED, "Passenger: Stalled"),
            (PassengerStatus.STOWING, "Passenger: Stowing Luggage"),
            ("seat_empty", "Seat: Empty"),
            ("seat_occupied", "Seat: Occupied")
        ]

        start_x = self.PADDING * 2
        item_y_offset = legend_y_start + self.PADDING * 2

        for i, (key, text) in enumerate(legend_items):
            color = self.COLORS[key]
            # col = i % 2
            # row = i // 2
            # item_x = start_x + (self.screen.get_width() / 2) * col
            current_item_y = item_y_offset + i * (self.FONT_SIZE + self.PADDING)
            pygame.draw.rect(self.screen, color, (start_x, current_item_y, self.SEAT_SIZE, self.FONT_SIZE))
            text_surf = self.font.render(text, True, self.COLORS["text"])
            self.screen.blit(text_surf, (start_x + (self.SEAT_SIZE * 1.5) + self.PADDING, current_item_y))

        # This updates the entire window with everything drawn in this frame
        pygame.display.flip()

        # Control the frame rate
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

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
        
        print("\nLine entering plane:")
        for i in range(self.num_of_rows, len(self.boarding_line.line)):
            passenger = self.boarding_line.line[i]
            
            print(f"{passenger} {passenger.status}")
        
        print("\nLobby:")
        for row in self.lobby.lobby_rows:
            for passenger in row.passengers:
                print(passenger, end=" ")
            
            if(len(row.passengers) > 0):
                print()
        
        print("\n")

    # This method is used to mask the actions that are allowed
    # action_masks() is the function signature required by the MaskabePPO class
    def action_masks(self) -> list[bool]:
        mask = []

        for row in self.lobby.lobby_rows:
            if len(row.passengers) == 0:
                mask.append(False)
            else:
                mask.append(True)
        return mask

# Check validity of the environment
def check_my_env():
    from gymnasium.utils.env_checker import check_env
    env = gym.make('airplane-boarding-v0', render_mode=None)
    check_env(env.unwrapped)

if __name__ == "__main__":
    env = gym.make('airplane-boarding-v0', num_of_rows=10, seats_per_row=5, render_mode='human')

    observation, _= env.reset()
    terminated = False
    total_reward = 0
    step_count = 0

    while not terminated:
        # Choose random action
        action = env.action_space.sample()

        # Skip action if invalid
        masks = env.unwrapped.action_masks()
        if(masks[action] == False):
            continue
        
        # Perform action
        observation, reward, terminated, _, _ = env.step(action)
        total_reward += reward

        step_count += 1

        print(f"Step {step_count} Action: {action}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}\n")
    env.close()
    print(f"Total Reward: {total_reward}")
