import pygame
import torch
import math


class Grid:
    def __init__(self, x_size, y_size, x_pos, y_pos, x_len, y_len):
        self.matrix = torch.zeros((x_size, y_size))
        self.x_size = x_size
        self.y_size = y_size
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.x_len = x_len
        self.y_len = y_len

    def redraw(self, x, y):
        val = self.matrix[y][x].item() * 255
        color = (val, val, val)
        pygame.draw.rect(pygame.display.get_surface(),
                         color,
                         [self.x_pos + x * self.x_len, self.y_pos + y * self.y_len, self.x_len, self.y_len])
        pygame.display.update([self.x_pos + x * self.x_len, self.y_pos + y * self.y_len, self.x_len, self.y_len])

    def try_paint(self):
        sign = 0

        if pygame.mouse.get_pressed()[0] == 1:
            sign = 1
        if pygame.mouse.get_pressed()[2] == 1:
            sign = -1

        if sign == 0:
            return

        intensity = 0.100
        mouse_x, mouse_y = pygame.mouse.get_pos()
        mouse_x -= self.x_pos
        mouse_y -= self.y_pos
        where = []
        val = [(0, 1), (-1, 0.1), (+1, 0.1)]
        for sign_x, intensity_x in val:
            for sign_y, intensity_y in val:
                x = math.floor((mouse_x + sign_x * self.x_len / 3) / self.x_len)
                y = math.floor((mouse_y + sign_y * self.y_len / 3) / self.y_len)
                if 0 <= x < self.matrix.shape[0] and 0 <= y < self.matrix.shape[1]:
                    where.append((x, y, intensity_y * intensity_x))
        for (x, y, intensity_v) in where:
            self.matrix[y][x] += sign * intensity * intensity_v
            self.matrix[y][x] = min(max(self.matrix[y][x].item(), 0), 1)
            self.redraw(x, y)

    def redraw_full(self):
        for x in range(self.matrix.shape[0]):
            for y in range(self.matrix.shape[1]):
                self.redraw(x, y)

    def clear(self):
        self.matrix = self.matrix.fill_(0)
        self.redraw_full()
