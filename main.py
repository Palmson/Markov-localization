import pygame
import random
from random import choices
from enum import Enum
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
TILE_SIZE = 50
GRID_WIDTH, GRID_HEIGHT = WIDTH // TILE_SIZE, HEIGHT // TILE_SIZE
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
CLOCK = 10

epsilon = 1 / (WIDTH * HEIGHT * WIDTH * HEIGHT)


class Direction(Enum):
    Up = 1
    Right = 2
    Down = 3
    Left = 4


class Position():
    x = 0
    y = 0
    direction = Direction.Right

    def __init__(self, x, y, direction=Direction.Right):
        self.x = x
        self.y = y
        self.direction = direction


class Step:
    direction = Direction.Right
    sensedDistance = 0
    realPosition = Position(0, 0)

    def __init__(self, direction, sensedDistance, position):
        self.direction = direction
        self.sensedDistance = sensedDistance
        self.realPosition = position


def normalize(matrix):
    retVal = matrix.copy()
    retVal = retVal - retVal.mean()
    retVal = retVal / np.abs(retVal).max()

    return retVal


def calcPrior(oldPosterior, direction):
    global epsilon, map
    
    steps = [3, 4, 5, 6, 7]
    stepProbability = [0, 0, 0, 0.05, 0.2, 0.5, 0.2, 0.05]
    prior = np.full((WIDTH, HEIGHT), 0.0)

    for i in range(0, WIDTH):
        for j in range(0, HEIGHT):
            if direction == Direction.Right or direction == Direction.Left:
                if map[i, :].__contains__(1):
                    if direction == Direction.Right:
                        for stepSize in steps:
                            if map[i, j] == 1:
                                break
                            else:
                                if j - stepSize >= 0:
                                    skip = False
                                    for t in range(0, stepSize + 1):
                                        if j - t >= 0 and map[i, j - t] == 1:
                                            skip = True
                                            break
                                    if not skip:
                                        if j - stepSize >= 0:
                                            prior[i, j] += oldPosterior[i, j - stepSize] * stepProbability[stepSize]
                                    else:
                                        break
                    else:
                        for stepSize in steps:
                            if map[i, j] == 1:
                                break
                            else:
                                if j + stepSize < WIDTH:
                                    skip = False
                                    for t in range(0, stepSize + 1):
                                        if j + t < WIDTH and map[i, j + t] == 1:
                                            skip = True
                                            break
                                    if not skip:
                                        if j + stepSize < WIDTH:
                                            prior[i, j] += oldPosterior[i, j + stepSize] * stepProbability[stepSize]
                                    else:
                                        break
                else:
                    for stepSize in steps:
                        if direction == Direction.Right:
                            if j - stepSize >= 0:
                                prior[i, j] += oldPosterior[i, j - stepSize] * stepProbability[stepSize]
                        if direction == Direction.Left:
                            if j + stepSize < HEIGHT:
                                prior[i, j] += oldPosterior[i, j + stepSize] * stepProbability[stepSize]

            elif direction == Direction.Down or direction == Direction.Up:
                if map[:, j].__contains__(1):
                    if direction == Direction.Up:
                        for stepSize in steps:
                            if map[i, j] == 1:
                                break
                            else:
                                if i + stepSize < HEIGHT:
                                    skip = False
                                    for t in range(0, stepSize + 1):
                                        if i + t < HEIGHT and map[i + t, j] == 1:
                                            skip = True
                                            break
                                    if not skip:
                                        if i + stepSize < HEIGHT:
                                            prior[i, j] += oldPosterior[i + stepSize, j] * stepProbability[stepSize]
                                    else:
                                        break

                    else:
                        for stepSize in steps:
                            if map[i, j] == 1:
                                break
                            else:
                                if i - stepSize >= 0:
                                    skip = False
                                    for t in range(0, stepSize + 1):
                                        if 0 <= i - t and map[i - t, j] == 1:
                                            skip = True
                                            break
                                    if not skip:
                                        if i - stepSize > 0:
                                            prior[i, j] += oldPosterior[i - stepSize, j] * stepProbability[
                                                stepSize]
                                    else:
                                        break
                else:
                    for stepSize in steps:
                        if direction == Direction.Up:
                            if i + stepSize < HEIGHT:
                                prior[i, j] += oldPosterior[i + stepSize, j] * stepProbability[stepSize]

                        if direction == Direction.Down:
                            if i - stepSize >= 0:
                                prior[i, j] += oldPosterior[i - stepSize, j] * stepProbability[stepSize]

    return prior


def calcPosterior(sensorValue, direction, prior):
    global epsilon, map

    sensorProbability = {
        sensorValue - 2: 0.05,
        sensorValue - 1: 0.2,
        sensorValue: 0.5,
        sensorValue + 1: 0.2,
        sensorValue + 2: 0.05
    }

    posterior = np.full((WIDTH, HEIGHT), 0.0)

    for i in range(0, WIDTH):
        for j in range(0, HEIGHT):
            if direction == Direction.Right or direction == Direction.Left:
                if map[i, :].__contains__(1):
                    if direction == Direction.Right:
                        if map[i, j] == 1:
                            continue
                        else:
                            for k in range(max(0, sensorValue - 2), sensorValue + 2 + 1):
                                if j + k + 1 == HEIGHT or (j + k + 1 < HEIGHT and map[i, j + k + 1] == 1):
                                    posterior[i, j] = prior[i, j] * sensorProbability[k]
                    elif direction == Direction.Left:
                        if map[i, j] == 1:
                            continue
                        else:
                            for k in range(max(0, sensorValue - 2), sensorValue + 2 + 1):
                                if j - k - 1 == -1 or (j - k - 1 >= 0 and map[i, j - k - 1] == 1):
                                    posterior[i, j] = prior[i, j] * sensorProbability[k]

                else:
                    if direction == Direction.Right:
                        for k in range(sensorValue - 2, sensorValue + 2 + 1):
                            if WIDTH > WIDTH - k - 1 >= 0:
                                posterior[i, HEIGHT - k - 1] = prior[i, HEIGHT - k - 1] * sensorProbability[k]
                    if direction == Direction.Left:
                        for k in range(sensorValue - 2, sensorValue + 2 + 1):
                            if 0 <= k < HEIGHT:
                                posterior[i, k] = prior[i, k] * sensorProbability[k]

            elif direction == Direction.Down or direction == Direction.Up:
                if map[:, j].__contains__(1):
                    if direction == Direction.Up:
                        if map[i, j] == 1:
                            continue
                        else:
                            for k in range(max(0, sensorValue - 2), sensorValue + 2 + 1):
                                if i - k - 1 == -1 or (i - k - 1 >= 0 and map[i - k - 1, j] == 1):
                                    posterior[i, j] = prior[i, j] * sensorProbability[k]

                    else:
                        if map[i, j] == 1:
                            continue
                        else:
                            for k in range(max(0, sensorValue - 2), sensorValue + 2 + 1):
                                if i + k + 1 == HEIGHT or (i + k + 1 < HEIGHT and map[i + k + 1, j] == 1):
                                    posterior[i, j] = prior[i, j] * sensorProbability[k]

                else:
                    if direction == Direction.Up:
                        for k in range(max(0, sensorValue - 2), sensorValue + 2 + 1):
                            if 0 <= k < WIDTH:
                                posterior[k, j] = prior[k, j] * sensorProbability[k]

                    if direction == Direction.Down:
                        for k in range(sensorValue - 2, sensorValue + 2 + 1):
                            if WIDTH > WIDTH - k - 1 >= 0:
                                posterior[WIDTH - k - 1, j] = prior[WIDTH - k - 1, j] * sensorProbability[k]

    posterior[posterior < epsilon] = epsilon

    posterior = posterior / np.sum(posterior)

    return posterior


def getSensorDerivation():
    population = [-2, -1, 0, 1, 2]
    weights = [0.05, 0.2, 0.5, 0.2, 0.05]

    return choices(population, weights)[0]

def getStepSize():
    population = [3, 4, 5, 6, 7]
    weights = [0.05, 0.2, 0.5, 0.2, 0.05]

    return choices(population, weights)[0]

def doStep(direction):
    global realPos
    stepSize = getStepSize()
    
    if direction == Direction.Up:

        if realPos.y - stepSize < 0:
            stepSize = realPos.y
            
        realPos.y = realPos.y - stepSize
        
    elif direction == Direction.Right:

        if realPos.x + stepSize > WIDTH:
            stepSize = WIDTH - realPos.x

        realPos.x = realPos.x + stepSize

    elif direction == Direction.Down:

        if realPos.y + stepSize > HEIGHT:
            stepSize = HEIGHT - realPos.y

        realPos.y = realPos.y + stepSize

    elif direction == Direction.Left:

        if realPos.x - stepSize < 0:
            stepSize = realPos.x

        realPos.x = realPos.x - stepSize

def senseDistance(direction):
    global realPos
    distance = 0

    if direction == Direction.Up:
        for i in range(1, realPos.y + 1):
            if realPos.y - i < 0:
                break
            if map[realPos.x - i, realPos.y] == 0:
                distance += 1
            else:
                break

    elif direction == Direction.Right:
        for i in range(1, WIDTH - realPos.x):
            if realPos.x + i > WIDTH:
                break
            if map[realPos.x, realPos.y + i] == 0:
                distance += 1
            else:
                break

    elif direction == Direction.Down:
        for i in range(1, HEIGHT - realPos.y):
            if realPos.y + i > HEIGHT:
                break
            if map[realPos.x, realPos.y + i] == 0:
                distance += 1
            else:
                break
            
    elif direction == Direction.Left:
        for i in range(1, realPos.x + 1):
            if realPos.x - i < 0:
                break
            if map[realPos.x - i, realPos.y] == 0:
                distance += 1
            else:
                break
            
    return distance

def wait_for_button():
    global paused
    if keys[pygame.K_SPACE]:
        paused = False
    else:
        paused = True

# Function to update probabilities based on last action
def update_probability(last_action_index):
    new_transition_matrix = []
    for i in range(4):
        new_row = []
        for j in range(4):
            if i == last_action_index:
                new_row.append(0.8 if i == j else 0.05)
            else:
                new_row.append(0.05 if i == j else 0.1)
        new_transition_matrix.append(new_row)
    return new_transition_matrix

def display_text(x, y, text='0.0'):
    tile_center_x = (x * TILE_SIZE) + (TILE_SIZE // 2)
    tile_center_y = (y * TILE_SIZE) + (TILE_SIZE // 2)

    font = pygame.font.Font(None, 24)  
    text_surface = font.render(text, True, BLACK) 
    text_rect = text_surface.get_rect()
    text_rect.center = (tile_center_x, tile_center_y)
    screen.blit(text_surface, text_rect)


# Initialize window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot Tile Grid Simulation")
clock = pygame.time.Clock()

# Generate random grid with red or green tiles and walls
grid = [[random.choice([RED, GREEN, GRAY]) for _ in range(WIDTH)] for _ in range(HEIGHT)]
map = np.empty((HEIGHT, WIDTH))
map[:] = 0
for x in range(HEIGHT):
    for y in range(WIDTH):
        if grid[x][y] == GRAY:
            map[x][y] = 1
# Randomly select eight coordinates
target = []
for _ in range(8):
    flag = False
    while not flag:
        x = random.randint(0, WIDTH)  # Random row index
        y = random.randint(0, HEIGHT)  # Random column index
        if map[x][y] == 0:
            flag = True
    target.append((x, y))

steps = [Step(direction=Direction.Right, sensedDistance=0, position=Position(target[0][0], target[0][1])),
         Step(direction=Direction.Up, sensedDistance=0, position=Position(target[1][0], target[1][1])),
         Step(direction=Direction.Left, sensedDistance=0, position=Position(target[2][0], target[2][1])),
         Step(direction=Direction.Down, sensedDistance=0, position=Position(target[3][0], target[3][1])),
         Step(direction=Direction.Right, sensedDistance=0, position=Position(target[4][0], target[4][1])),
         Step(direction=Direction.Up, sensedDistance=1, position=Position(target[5][0], target[5][1])),
         Step(direction=Direction.Left, sensedDistance=5, position=Position(target[6][0], target[6][1])),
         Step(direction=Direction.Down, sensedDistance=1, position=Position(target[7][0], target[7][1]))]

probabilities = np.full((HEIGHT, WIDTH), 1 / (HEIGHT * WIDTH - np.sum(map)))

for _ in range(1):
    x = random.randint(0, GRID_WIDTH - 1)
    y = random.randint(0, GRID_HEIGHT - 1)
    grid[x][y] = GRAY

# Initial position of the robot
robot_x = random.randint(0, GRID_WIDTH - 1)
robot_y = random.randint(0, GRID_HEIGHT - 1)
currentPosition = Position(robot_x, robot_y)

# Main loop
running = True
paused = False
last_action_index = None
step_counter = 0
while running:
    keys = pygame.key.get_pressed()  # Checking pressed keys
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused

    if not paused:
        screen.fill(WHITE)
        if step_counter == 0:

            # Draw grid
            for x in range(GRID_HEIGHT):
                for y in range(GRID_WIDTH):
                    pygame.draw.rect(screen, grid[x][y], (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
                    if grid[x][y] != GRAY:
                        display_text(x, y)

            # Draw robot
            pygame.draw.rect(screen, BLACK,
                             (currentPosition.x * TILE_SIZE, currentPosition.y * TILE_SIZE, TILE_SIZE, TILE_SIZE))

            pygame.display.flip()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            paused = True
            # Delay
            clock.tick(CLOCK)
        elif step_counter < 8:
            step = steps[step_counter]
            currentPosition = step.realPosition
            prior = calcPrior(probabilities, step.direction)
            distance = step.sensedDistance
            posterior = calcPosterior(distance, step.direction, prior)

            # Draw grid
            for x in range(GRID_HEIGHT):
                for y in range(GRID_WIDTH):
                    pygame.draw.rect(screen, grid[x][y], (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
                    if grid[x][y] != GRAY:
                        display_text(x, y, f"{round(prior[x][y], 10)}")

            # Draw robot
            pygame.draw.rect(screen, BLACK,
                             (currentPosition.x * TILE_SIZE, currentPosition.y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
            pygame.display.update()
            # Draw grid
            for x in range(GRID_HEIGHT):
                for y in range(GRID_WIDTH):
                    pygame.draw.rect(screen, grid[x][y], (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
                    if grid[x][y] != GRAY:
                        display_text(x, y, f"{round(posterior[x][y], 10)}")
            # Draw robot
            pygame.draw.rect(screen, BLACK,
                             (currentPosition.x * TILE_SIZE, currentPosition.y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
            
            pygame.display.update()
            probabilities = posterior

            paused = True

        else:
            running = False

        step_counter += 1

    else:
        wait_for_button()

# Quit Pygame
pygame.quit()
