# FLUIDS OpenAI Gym Interface

This repostitory presents an OpenAI Gym interface to the [FLUIDS simulator](https://github.com/BerkeleyAutomation/Urban_Driving_Simulator/tree/v2). The core FLUIDS simulator must be installed for this to run.

```
pip3 install -e .
```

To register and use with OpenAI Gym, add the following lines to your Gym code.

```
import gym
import gym_fluids
```

The FLUIDS environment is now available as "fluids-v2".