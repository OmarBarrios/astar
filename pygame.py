import pygame
import math
from queue import PriorityQueue
from PIL import Image
import numpy as np
import random

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

# Definición de colores
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

class Spot:
    def __init__(self, row, col, width, total_rows, color):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = color
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():  # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():  # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False


def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    # Calcular la distancia euclidiana
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()


def algorithm(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False


def create_grid_from_image(image_path, rows, cols):
    image = Image.open(image_path)
    image = image.resize((cols, rows))
    image = image.rotate(90, expand=True)
    pixels = list(image.getdata())
    matrix = [pixels[i:i + cols] for i in range(0, len(pixels), cols)]

    grid = []
    gap = WIDTH // cols
    for i in range(rows):
        grid.append([])
        for j in range(cols):
            color = matrix[rows - 1 - i][j]
            # Cambiar los píxeles grises a negro
            if color == (128, 128, 128):
                color = (0, 0, 0)
            spot = Spot(i, j, gap, rows, color)
            grid[i].append(spot)

    return grid


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col

def find_robots_and_targets(grid, rows, cols):
    robots = []
    targets = []
    for i in range(rows):
        for j in range(cols):
            spot = grid[i][j]
            if spot.is_start():
                robots.append((i, j))
            elif spot.is_end():
                targets.append((i, j))
    return robots, targets

def move_robot(grid, robot_pos):
    row, col = robot_pos
    robot_spot = grid[row][col]
    neighbors = robot_spot.neighbors

    if neighbors:
        new_pos = random.choice(neighbors).get_pos()
        new_row, new_col = new_pos
        grid[new_row][new_col].make_start()  # Pintar un píxel a su alrededor de su mismo color
        robot_spot.reset()  # Restablecer el color del robot en su posición original
        return new_pos
    else:
        return robot_pos


def main(win, width, image_path):
    ROWS = 150
    COLS = 150
    grid = create_grid_from_image(image_path, ROWS, COLS)

    robots, targets = find_robots_and_targets(grid, ROWS, COLS)

    moving_robots = False

    run = True
    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    moving_robots = True

                if event.key == pygame.K_c:
                    robots, targets = find_robots_and_targets(grid, ROWS, COLS)
                    grid = create_grid_from_image(image_path, ROWS, COLS)

        if moving_robots:
            for i, robot_pos in enumerate(robots):
                new_pos = move_robot(grid, robot_pos)
                robots[i] = new_pos

    pygame.quit()


main(WIN, WIDTH, 'Mapa_Ex01.png')
