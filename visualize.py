import sys, getopt
import numpy as np
import math
import pygame
import torch

from load import default_arguments, load_arguments, initialize_objects
from environments.general_environment import GeneralEnvironment

SHORT_OPTIONS = ""
LONG_OPTIONS = [
    "loading_path=",
    "viz_episode_index=",
    "loadEnv=",
    "visDim=",
    "stateSize=",
    "fps=",
    "pause",
    "trace"
]

COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "maroon4": (139, 28, 98, 255),
    "magenta": (255,0,230),
    "forest_green": (0,50,0),
    "tan": (230,220,170),
    "coffee_brown": (200,190,140),
    "moon_glow": (235,245,255),
    "red": (255, 0, 0),
    "agent": (205, 92, 92)
}


def state_to_surface(maps, nb_repeats, info, positions):
    dim_x, dim_y = maps["obstacle_map"].shape

    unscaled_img = np.zeros((dim_x, dim_y, 3))
    unscaled_img[maps["obstacle_map"].astype(bool)] = np.array(COLORS["coffee_brown"])
    unscaled_img[maps["coverage_map"].astype(bool)] = np.array(COLORS["forest_green"])

    curr_x, curr_y = info["current_position"]

    # AGENT SIZE
    if "agent_size" in info:
        agent_size = info["agent_size"]
        mask = GeneralEnvironment.get_radius_map(agent_size)
        offset = agent_size // 2
        local_img = unscaled_img[
                    curr_x - offset: curr_x + offset + 1,
                    curr_y - offset: curr_y + offset + 1
        ]
        local_img[mask] = local_img[mask] * 0.5 + 0.5 * np.array(COLORS["agent"])

    if maps["obstacle_map"][info["current_position"]]:
        unscaled_img[info["current_position"]] = np.array(COLORS["red"])
    else:
        unscaled_img[info["current_position"]] = np.array(COLORS["white"])

    # # test points to verify orientation
    # unscaled_img[5, 0, :] = np.array((255, 0, 0))
    # unscaled_img[0, 5, :] = np.array((0, 255, 0))

    scaled_img = np.repeat(unscaled_img, nb_repeats[0], axis=0)
    scaled_img = np.repeat(scaled_img, nb_repeats[1], axis=1)

    surface = pygame.surfarray.make_surface(scaled_img)

    # FIELD OF VIEW
    if "fov" in info:
        offset = info["fov"] // 2
        min_x, max_x = (curr_x - offset) * nb_repeats[0], (curr_x + offset + 1) * nb_repeats[0]
        min_y, max_y = (curr_y - offset) * nb_repeats[1], (curr_y + offset + 1) * nb_repeats[1]

        points = [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ]

        if "angle" in info:
            angle = info["angle"]

            center_x = (curr_x + 0.5) * nb_repeats[0]
            center_y = (curr_y + 0.5) * nb_repeats[1]

            rel_points_x = points[:, 0] - center_x
            rel_points_y = points[:, 1] - center_y

            points_x = math.cos(angle) * rel_points_x - math.sin(angle) * rel_points_y + center_x
            points_y = math.sin(angle) * rel_points_x + math.cos(angle) * rel_points_y + center_y

            points = np.transpose(
                np.stack([points_x, points_y])
            )

        pygame.draw.lines(surface, color=(235, 245, 255),
                           closed=True, points=points)

    # ROTATION
    if "angle" in info:
        angle = info["angle"]

        point_1 = ((curr_x + 0.5) * nb_repeats[0],
                   (curr_y + 0.5) * nb_repeats[1])
        point_2 = (point_1[0] + math.cos(angle) * nb_repeats[0],
                   point_1[1] + math.sin(angle) * nb_repeats[1])
        pygame.draw.line(surface, color=(0, 245, 255),
                         start_pos=point_1, end_pos=point_2)

    if len(positions) > 0:
        start_x, start_y = positions[0]
        pygame.draw.circle(surface, color=(255, 0, 0),
                           center=((start_x + 0.5) * nb_repeats[0], (start_y + 0.5) * nb_repeats[1]), radius=5)
        if len(positions) > 1:
            pos_arr = np.array(positions)
            pos_arr[:, 0] = (pos_arr[:, 0] + 0.5) * nb_repeats[0]
            pos_arr[:, 1] = (pos_arr[:, 1] + 0.5) * nb_repeats[1]

            pygame.draw.lines(surface, color=(255, 0, 0),
                              closed=False, points=pos_arr)

    surface = pygame.transform.flip(surface, False, True)

    return surface


def main(argv):
    # ARGUMENTS
    try:
        options, args = getopt.getopt(argv, SHORT_OPTIONS, LONG_OPTIONS)
    except getopt.GetoptError:
        print("badly formatted command line arguments")

    arguments = default_arguments()
    arguments["loading_path"] = "./results/8x_multi/"
    for option, val in options:
        if option == "--loading_path":
            arguments["loading_path"] = val
            break
    print(arguments["loading_path"])
    arguments.update(load_arguments(arguments["loading_path"], "arguments.json"))        

    arguments["viz_episode_index"] = 250     # 可视化时采用第几轮训练的结果
    arguments["viz_episode_index"] = arguments["nbEpisodes"] # 使用最后的训练结果可视化
    arguments["visDim"] = (512, 512)
    arguments["stateSize"] = 128
    arguments["fps"] = 2
    arguments["pause"] = False
    arguments["trace"] = False

    for option, val in options:
        if option == "--viz_episode_index":
            arguments["viz_episode_index"] = int(val)

        if option == "--loadEnv":
            arguments["loadEnv"] = tuple(val.split(","))

        if option == "--visDim":
            arguments["visDim"] = tuple(map(int, val.split(",")))

        if option == "--stateSize":
            arguments["stateSize"] = int(val)

        if option == "--fps":
            arguments["fps"] = int(val)

        if option == "--pause":
            arguments["pause"] = True

        if option == "--trace":
            arguments["trace"] = True

    arguments["cuda"] = False
    print(arguments)

    # INITIALISE OBJECTS
    env, agent, _ = initialize_objects(arguments)
    agent.load(arguments["loading_path"], arguments["viz_episode_index"])
    agent.evaluate()

    # INITIALISE PYGAME
    pygame.init()
    screen = pygame.display.set_mode((arguments["visDim"][0],
                                      arguments["visDim"][1] + arguments["stateSize"]))
    screen.fill(COLORS["white"])
    nb_repeats = (arguments["visDim"][0] // arguments["dim"][0],
                  arguments["visDim"][1] // arguments["dim"][1])
    state_repeats = arguments["stateSize"] // env.get_state_shape()[1]
    clock = pygame.time.Clock()
    font = pygame.font.Font("freesansbold.ttf", 32)

    running = True

    # ENVIRONMENT SETUP
    done = False
    current_state = env.reset()
    total_reward = 0.0
    reward = 0.0

    info = env.get_info({})
    maps = {
        "obstacle_map": env.get_obstacle_map(),
        "coverage_map": env.get_coverage_map()
    }

    next_step = not arguments["pause"]
    positions = []
    if arguments["trace"]:
        positions.append(info['current_position'])

    while running:
        screen.fill(COLORS["white"])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                keys_pressed = pygame.key.get_pressed()

                # reset environment if R-key is pressed
                if keys_pressed[pygame.K_r]:
                    current_state = env.reset()
                    maps["obstacle_map"] = env.get_obstacle_map()
                    maps["coverage_map"] = env.get_coverage_map()
                    info = env.get_info({})
                    done = False
                    positions = []
                    if arguments["trace"]:
                        positions.append(info['current_position'])

                    total_reward = 0.0
                    reward = 0.0

                    print()
                    print()
                    print("NEW EPISODE")

                if keys_pressed[pygame.K_n]:
                    next_step = True

        env_surface = state_to_surface(maps, nb_repeats, info, positions)
        screen.blit(env_surface, (0, 0))

        for i in range(current_state.shape[0]):
            unscaled = current_state[i]
            unscaled = np.array(unscaled * 255, dtype=int)
            unscaled = np.stack([unscaled, unscaled, unscaled])
            unscaled = np.moveaxis(unscaled, 0, 2)

            scaled_img = np.repeat(unscaled, state_repeats, axis=0)
            scaled_img = np.repeat(scaled_img, state_repeats, axis=1)

            surface = pygame.surfarray.make_surface(scaled_img)
            surface = pygame.transform.flip(surface, False, True)

            pygame.draw.rect(
                surface,
                (255, 0, 0),
                (0, 0, arguments["stateSize"], arguments["stateSize"]),
                1
            )

            screen.blit(surface, (i * arguments["stateSize"], arguments["visDim"][1]))

        # text = font.render(f"REWARD: {round(reward, 3)} --- TOTAL REWARD: {round(total_reward, 3)}",
        #                    True, COLORS['black'], COLORS['white'])
        # screen.blit(text, (100, 25))

        if not done and next_step:
            action = agent.select_action(
                torch.tensor(current_state, dtype=torch.float)
            )
            current_state, reward, done, info = env.step(action)
            total_reward += reward
            if arguments["trace"]:
                positions.append(info['current_position'])
            maps["obstacle_map"] = env.get_obstacle_map()
            maps["coverage_map"] = env.get_coverage_map()

            if arguments["pause"]:
                next_step = False

        pygame.display.update()
        clock.tick(arguments["fps"])

    pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1:])
