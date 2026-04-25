from openreward.environments import Server
from env import MinesweeperEnvironment

if __name__ == "__main__":
    server = Server([MinesweeperEnvironment])
    server.run()
