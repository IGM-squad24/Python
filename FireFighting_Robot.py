import random
import time
# Firefighting Robot

class FirefightingRobot:
    def __init__(self):
        # Rooms are represented by a dictionary with fire status: True means fire present
        self.rooms = {"A": random.choice([True, False]),
                      "B": random.choice([True, False]),
                      "C": random.choice([True, False])}
        self.current_room = "A"
        # Memory to store visited rooms and their status
        self.visited = {}

    def perceive(self):
        """Perceive the fire status of the current room."""
        status = self.rooms[self.current_room]
        print(f"Robot in room {self.current_room}: Fire status = {'On fire' if status else 'Safe'}")
        return status

    def decide_action(self, fire_status):
        """Decide action based on the current room's fire status."""
        if fire_status:
            return "extinguish"
        else:
            # Decide next room to move to: simple policy using order A -> B -> C -> stop.
            # Save current room's status to memory.
            self.visited[self.current_room] = "safe"
            # Determine next room in alphabetical order that hasn't been confirmed safe.
            room_order = ["A", "B", "C"]
            current_index = room_order.index(self.current_room)
            for next_room in room_order[current_index + 1:]:
                if next_room not in self.visited:
                    return f"move_to_{next_room}"
            # If no unvisited room left, check earlier rooms in case fire reignited.
            for prev_room in room_order[:current_index]:
                if prev_room not in self.visited:
                    return f"move_to_{prev_room}"
            return "do_nothing"

    def act(self, action):
        """Execute the chosen action and update internal state."""
        if action == "extinguish":
            print(f"Extinguishing fire in room {self.current_room}.")
            self.rooms[self.current_room] = False  # Fire is put out.
            self.visited[self.current_room] = "safe"
        elif action.startswith("move_to_"):
            new_room = action.split("_")[-1]
            print(f"Moving from room {self.current_room} to room {new_room}.")
            self.current_room = new_room
        elif action == "do_nothing":
            print("All rooms are safe. Stopping operations.")
        else:
            print("No valid action determined.")

def simulate_firefighting_robot():
    print("\n--- Firefighting Robot Simulation ---")
    robot = FirefightingRobot()
    step = 0
    # Continue until all rooms are confirmed safe
    while True:
        step += 1
        print(f"\nStep {step}:")
        current_status = robot.perceive()
        action = robot.decide_action(current_status)
        robot.act(action)
        # Check if all rooms are safe
        if all(status is False for status in robot.rooms.values()):
            print("Final room statuses:", robot.rooms)
            break
        # Limit steps to avoid infinite loops in simulation
        if step >= 10:
            print("Simulation ending after 10 steps.")
            break
        time.sleep(1)

