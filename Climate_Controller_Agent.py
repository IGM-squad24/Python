import random
import time

# Climate Controller Agent

class ClimateControlAgent:
    def __init__(self, threshold=25):
        self.threshold = threshold
        self.cooling_active = False

    def sense_temperature(self, current_temp):
        """Sense the current temperature."""
        return current_temp

    def decide_action(self, current_temp):
        """Decide action based on sensed temperature."""
        if current_temp > self.threshold:
            return "activate_cooling"
        elif current_temp <= self.threshold and self.cooling_active:
            return "deactivate_cooling"
        else:
            return "do_nothing"

    def act(self, action):
        """Perform action and update internal state."""
        if action == "activate_cooling":
            if not self.cooling_active:
                print("Activating cooling system.")
                self.cooling_active = True
            else:
                print("Cooling system already active.")
        elif action == "deactivate_cooling":
            if self.cooling_active:
                print("Deactivating cooling system.")
                self.cooling_active = False
            else:
                print("Cooling system already off.")
        else:
            print("Temperature is stable. No action needed.")

def simulate_climate_control():
    print("\n--- Climate Control Simulation ---")
    agent = ClimateControlAgent(threshold=25)
    # Simulate temperature changes over time
    for i in range(10):
        # Simulate a temperature reading between 20°C and 30°C
        current_temp = random.uniform(20, 30)
        print(f"\nTime step {i+1}: Current temperature = {current_temp:.2f}°C")
        sensed_temp = agent.sense_temperature(current_temp)
        action = agent.decide_action(sensed_temp)
        agent.act(action)
        # Pause for demonstration (remove or adjust in real use)
        time.sleep(1)
