import sys, getopt
import torch.optim as optim
import json
import pprint

from environments.env_generation import EnvironmentGenerator
from environments.env_representation import EnvironmentRepresentation
from environments.environment import Environment

from networks.simple_q_network import SimpleDeepQNetworkGenerator

from deep_rl.deep_q_agent import DeepQAgent
from deep_rl.trainer import DeepRLTrainer

SHORT_OPTIONS = ""
LONG_OPTIONS = [
    "loadArguments=",
    "heightRequired",
    "dim=",
    "hFreq=",
    "oFreq=",
    "fillRatio=",
    "loadEnv=",
    "movePunish=",
    "obstaclePunish=",
    "discoverReward=",
    "coverageReward=",
    "maxStepMultiplier=",
    "networkGen=",
    "rlAgent=",
    "epsilonDecay=",
    "optim=",
    "nbEpisodes=",
    "printEvery=",
    "saveEvery=",
    "savePath="
]

GENERATORS = {
    "simpleQ": SimpleDeepQNetworkGenerator
}

AGENTS = {
    "deepQ": DeepQAgent
}

OPTIMIZERS = {
    "rmsProp": optim.RMSprop
}

def main(argv):
    try:
        options, args = getopt.getopt(argv, SHORT_OPTIONS, LONG_OPTIONS)
    except getopt.GetoptError:
        print("badly formatted command line arguments")

    arguments = {
        "heightRequired": False,
        "dim": (16, 16),
        "hFreq": (2, 2),
        "oFreq": (2, 2),
        "fillRatio": 0.14,
        "loadEnv": None,
        "movePunish": 0.05,
        "obstaclePunish": 0.5,
        "discoverReward": 1.0,
        "coverageReward": 50.0,
        "maxStepMultiplier": 2,
        "networkGen": "simpleQ",
        "rlAgent": "deepQ",
        "optim": "rmsProp",
        "epsilonDecay": 2000,
        "nbEpisodes": 2000,
        "printEvery": 50,
        "saveEvery": 250,
        "savePath": "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/"
    }

    for option, argument in options:
        if option == "--loadArguments":
            with open(f"{argument}") as input_file:
                input_data = json.load(input_file)
                arguments.update(input_data)

                pprint.pprint(arguments)

        if option == "--heightRequired":
            arguments["heightRequired"] = True

        if option == "--dim":
            arguments["dim"] = tuple(tuple(map(int, argument.split(","))))


        if option == "--hFreq":
            arguments["hFreq"] = tuple(map(int, argument.split(",")))

        if option == "--oFreq":
            arguments["oFreq"] = tuple(map(int, argument.split(",")))

        if option == "--fillRatio":
            arguments["fillRatio"] = float(argument)

        if option == "--loadEnv":
            arguments["loadEnv"] = tuple(argument.split(","))

        if option == "--movePunish":
            arguments["movePunish"] = float(argument)

        if option == "--obstaclePunish":
            arguments["obstaclePunish"] = float(argument)

        if option == "--discoverReward":
            arguments["discoverReward"] = float(argument)

        if option == "--coverageReward":
            arguments["coverageReward"] = float(argument)

        if option == "--maxStepMultiplier":
            arguments["maxStepMultiplier"] = int(argument)

        if option == "--networkGen":
            if argument in GENERATORS:
                arguments["networkGen"] = argument
            else:
                raise Exception("TRAIN.py: given network generator is not defined...")

        if option == "--optim":
            if argument in OPTIMIZERS:
                arguments["optim"] = argument
            else:
                raise Exception("TRAIN.py: given optimizer is not defined...")

        if option == "--rlAgent":
            if argument in AGENTS:
                arguments["rlAgent"] = argument
            else:
                raise Exception("TRAIN.py: given agent is not defined...")

        if option == "--epsilonDecay":
            arguments["epsilonDecay"] = int(argument)

        if option == "--nbEpisodes":
            arguments["nbEpisodes"] = int(argument)

        if option == "--printEvery":
            arguments["printEvery"] = int(argument)

        if option == "--saveEvery":
            arguments["saveEvery"] = int(argument)

        if option == "--savePath":
            arguments["savePath"] = argument

    with open(f"{arguments['savePath']}arguments.txt", 'w') as output_file:
        json.dump(arguments, output_file)

    env_generator = EnvironmentGenerator(arguments["heightRequired"])
    env_generator.set_dimension(arguments["dim"])
    env_generator.set_height_frequency(arguments["hFreq"])
    env_generator.set_obstacle_frequency(arguments["oFreq"])
    env_generator.set_fill_ration(arguments["fillRatio"])

    env = Environment(env_generator)
    if arguments["loadEnv"] is not None:
        env_repr = EnvironmentRepresentation()
        env_repr.load(arguments["loadEnv"][0], arguments["loadEnv"][1])
        arguments["dim"] = env_repr.get_dimension()
        env.set_environment_representation(env_repr)

    env.MOVE_PUNISHMENT = arguments["movePunish"]
    env.OBSTACLE_PUNISHMENT = arguments["obstaclePunish"]
    env.DISCOVER_REWARD = arguments["discoverReward"]
    env.COVERAGE_REWARD = arguments["coverageReward"]
    env.MAX_STEP_MULTIPLIER = arguments["maxStepMultiplier"]

    network_generator = GENERATORS[arguments["networkGen"]](
        arguments["dim"],
        env.get_input_depth(),
        env.get_nb_actions()
    )
    optim_class = OPTIMIZERS[arguments["optim"]]
    agent = AGENTS[arguments["rlAgent"]](
        network_generator,
        optim_class,
        env.get_nb_actions()
    )
    agent.EPSILON_DECAY = arguments["epsilonDecay"]

    DeepRLTrainer.NB_EPISODES = arguments["nbEpisodes"]
    DeepRLTrainer.INFO_EVERY = arguments["printEvery"]
    DeepRLTrainer.SAVE_EVERY = arguments["saveEvery"]

    trainer = DeepRLTrainer(env, agent, arguments["savePath"])
    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])