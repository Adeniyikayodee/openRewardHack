from .server import LondonDynamicRouting
from openreward.environments import Server

import os
port = int(os.environ.get("PORT", 8000))
Server([LondonDynamicRouting]).run(port=port)
