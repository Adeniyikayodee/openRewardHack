from openreward.environments import Server

from cybench import CyBench

if __name__ == "__main__":
    server = Server([CyBench])
    server.run()
