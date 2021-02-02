from typing import Tuple
from enum import Enum


class Action:

    def __init__(self, start_coordinates: Tuple[int, int], direction_vector: Tuple[int, int]):
        self.start_coordinates = start_coordinates
        self.direction_vector = direction_vector

    @property
    def adjacent_cell(self) -> Tuple[int, int]:
        return self.start_coordinates[0] + self.direction_vector[0], self.start_coordinates[1] + self.direction_vector[1]

    @property
    def landing_cell(self) -> Tuple[int, int]:
        return self.start_coordinates[0] + (self.direction_vector[0] * 2), self.start_coordinates[1] + (self.direction_vector[1] * 2)

    def __hash__(self):
        return hash(self.start_coordinates) ^ hash(self.direction_vector)


class Shape(Enum):
    Diamond = 1
    Triangle = 2
