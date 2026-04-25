from openreward.environments import Server

from mobileenv import MobileEnv

if __name__ == "__main__":
    server = Server([MobileEnv])
    server.run()
