from dataclasses import dataclass
from enum import Enum
import numpy as np

class DetectionLeg(Enum):
    LEFT = 1,
    RIGHT = 2

    def to_json(self):
        if self == DetectionLeg.LEFT:
            return "LEG_LEFT"
        elif self == DetectionLeg.RIGHT:
            return "LEG_RIGHT"
        return "UNKNOWN_LEG"


class DetectionROI(Enum):
    FEMUR_HEAD = 1,
    KNEE = 2,
    ANKLE = 3


@dataclass
class Point:
    x: float
    y: float



@dataclass
class DetectionBox:
    upper_left: Point
    bottom_right: Point
    confidence: float


@dataclass
class DetectionResult:
    femur_head: Point
    ost_point: Point
    knee_outer: Point
    knee_inner: Point
    ankle_inner: Point
    ankle_outer: Point

    femur_head_radius: float
    correction_angle_in_deg: float


@dataclass
class Line:
    a: float
    b: float

    @classmethod
    def from_points(cls, point_a: Point, point_b: Point) -> 'Line':
        """ Calculates line coefficients based on two points. Returns line instance. """

        a, b = np.polyfit([point_a.x, point_b.x], [point_a.y, point_b.y], deg=1)
        return cls(a, b)

    def get_y(self, x: float) -> float:
        """ Returns y coordinate for given x. """

        return self.a * x + self.b

    def cross_point(self, other: 'Line') -> Point:
        """ Returns cross point between self and other line. """

        x = (self.b - other.b) / (other.a - self.a)
        y = self.get_y(x)
        return Point(x, y)


@dataclass
class CalculationResult:
    f_point: Point
    f_line: Line
    a_point: Point
    a_line: Line
    c_point: Point
    angle: float


