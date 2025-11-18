from flockrl_sim.environment.obstacles import Environment, EnvironmentBuilder
from flockrl_sim.environment.obstacles_types import Bounds, Obstacle

DEFAULT_BOUNDS: Bounds = (-5.0, 5.0, -5.0, 5.0, -4.0, 4.0)

# Create an environment manually
env = Environment(bounds=DEFAULT_BOUNDS, obstacles=[], seed=42)
print("Initial environment:")
print(env.summary())

# Add a manual obstacle
obs1 = Obstacle(id="wall_1", type="wall", position=(1.0, 2.0, 0.0), orientation=(0.0, 0.0, 0.0))
env.add_obstacle(obs1)
print("\nAfter adding one obstacle:")
print(env.summary())

# Retrieve the obstacle
found = env.get_obstacle_by_id("wall_1")
print("\nRetrieved obstacle:", found)

# Build environment using builder
builder = EnvironmentBuilder(Environment(bounds=DEFAULT_BOUNDS, obstacles=[], seed=123))
builder.add_random_obstacles(n=3)
built_env = builder.build()

print("\nBuilt environment:")
print(built_env.summary())

# Check that obstacles were added
for obs in built_env.obstacles:
    print(obs)
