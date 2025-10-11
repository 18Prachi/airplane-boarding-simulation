import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from enum import Enum
import numpy as np

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
# When running this code, you can ignore this warning: "UserWarning: WARN: Overriding environment airplane-boarding-v0 already in registry."
register(
    id='airplane-boarding-v0',
    entry_point='main:AirplaneEnv', # module_name:class_name
)

class PassengerStatus(Enum):
    MOVING  = 0
    STALLED = 1
    STOWING = 2
    SEATED  = 3

    # Returns the string representation of the PassengerStatus enum.
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

    # Returns the string representation of the Passenger class i.e. 2 digit seat number
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
        if (len(self.line) > 0 and not all(passenger is None for passenger in self.line)):
            return True

        return False

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

    def move_forward(self):

        for i, passenger in enumerate(self.line):
            # Skip, if no passenger in that spot or
            #   passenger is at the front of the line or
            #   passenger is stowing luggage
            if passenger is None or i==0 or passenger.status == PassengerStatus.STOWING:
                continue

            # Move passenger forward, if no one is blocking
            if (passenger.status == PassengerStatus.STALLED or passenger.status == PassengerStatus.MOVING) and self.line[i-1] is None:
                passenger.status = PassengerStatus.MOVING
                self.line[i-1] = passenger
                self.line[i] = None
            else:
                passenger.status = PassengerStatus.STALLED

        # Truncate the empty spots at the end of the line
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
            # Passenger starts Stowing luggage
            passenger.status = PassengerStatus.STOWING
            passenger.is_holding_luggage = False
            return False
        else:
            # Sit passenger in seat
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
        self.seats = [Seat(row_num * seats_per_row + i, row_num) for i in range(seats_per_row)]

    def try_sit_passenger(self, passenger: Passenger):
        # Check if passenger's seat is in this row
        found_seats = list(filter(lambda seats: seats.seat_num == passenger.seat_num, self.seats))

        if found_seats:
            found_seat: Seat = found_seats[0]
            return found_seat.seat_passenger(passenger)

        return False


class AirplaneEnv(gym.Env):
    metadata = {'render_modes': ['human','terminal'], 'render_fps': 1}

    def __init__(self, render_mode=None, num_of_rows=3, seats_per_row=5):

        self.seats_per_row = seats_per_row
        self.num_of_rows = num_of_rows
        self.num_of_seats = num_of_rows * seats_per_row

        self.render_mode = render_mode

        # Define the Action space.
        self.action_space = spaces.Discrete(self.num_of_rows)

        # Define the Observation space.
        # The observation space is used to validate the observation returned by reset() and step().
        # [0,-1,1,-1,2,-1....,6,2,7,1.....]
        self.observation_space = spaces.Box(
            low=-1,
            high=self.num_of_seats-1,
            shape=(self.num_of_seats * 2,),
            dtype=np.int32
        )


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # gym requires this call to control randomness and reproduce scenarios.

        self.airplane_rows = [AirplaneRow(row_num, self.seats_per_row) for row_num in range(self.num_of_rows)]
        self.lobby = Lobby(self.num_of_rows, self.seats_per_row)
        self.boarding_line = BoardingLine(self.num_of_rows)
        #for new feature of step
        self.current_step = 0

        self.render()

        return self._get_observation(), {}

    # Returns an array of the number of passengers in line
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
        self.current_step += 1 #Counting current steps
        assert row_num>=0 and row_num<self.num_of_rows, f"Invalid row number {row_num}"

        reward = 0

        passenger = self.lobby.remove_passenger(row_num)
        self.boarding_line.add_passenger(passenger)

        # If there are passengers in the lobby, move the line once
        if self.lobby.count_passengers()>0:
            self._move()
            reward = self._calculate_reward()
        else:
            # No more passengers in the lobby, so no more actions to choose from, move the line until all passengers are seated
            while self.is_onboarding():
                self._move()
                reward += self._calculate_reward()

        if self.is_onboarding():
            terminated = False
        else:
            terminated = True

        # Gym requires returning the observation, reward, terminated, truncated, and info dictionary.
        return self._get_observation(), reward, terminated, False, {}

    def _calculate_reward(self):
        reward = -self.boarding_line.num_passengers_stalled() #+ self.boarding_line.num_passengers_moving()
        return reward

    def is_onboarding(self):
        # If there are passengers in the lobby or in the boarding line, return True
        if self.lobby.count_passengers() > 0 or self.boarding_line.is_onboarding():
            return True

        return False
    
    def count_passengers_seated(self):
        count = 0
        for row in self.airplane_rows:
            for seat in row.seats:
                if seat.passenger is not None and seat.passenger.status == PassengerStatus.SEATED:
                    count+=1
        return count
    
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

        print("\nDebugging Metrics:")
        print(f"Step: {getattr(self, 'current_step', 0)}")
        print(f"Passengers stalled: {self.boarding_line.num_passengers_stalled()}")
        print(f"Passengers seated: {self.count_passengers_seated()}")
        print(f"Passengers in lobby: {self.lobby.count_passengers()}")
        if not self.is_onboarding():
            print(f"Total boarding time: {getattr(self, 'current_step', 0)}")

        print("\n")


    # This method is used to mask the actions that are allowed
    # action_masks() is the function signature required by the MaskablePPO class
    def action_masks(self) -> list[bool]:
        mask = []

        for row in self.lobby.lobby_rows:
            if len(row.passengers) == 0:
                mask.append(False)
            else:
                mask.append(True)

        return mask

# Check validity of the environment
def my_check_env():
    from gymnasium.utils.env_checker import check_env
    env = gym.make('airplane-boarding-v0', render_mode=None)
    check_env(env.unwrapped)

if __name__ == "__main__":
    # my_check_env()

    env = gym.make('airplane-boarding-v0', num_of_rows=3, seats_per_row=5, render_mode='terminal')

    observation, _ = env.reset()
    terminated = False
    total_reward = 0
    step_count = 0

    while not terminated:
        # Choose random action
        action = env.action_space.sample()

        # Skip action if invalid
        masks = env.unwrapped.action_masks()
        if(masks[action]==False):
            continue

        # Perform action
        observation, reward, terminated, _, _ = env.step(action)
        total_reward += reward

        step_count+=1

        print(f"Step {step_count} Action: {action}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}\n")

    print(f"Total Reward: {total_reward}")