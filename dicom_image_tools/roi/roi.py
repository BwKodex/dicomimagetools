from ..helpers.point import Point, CenterPosition


class Roi:
    def __init__(self, center: CenterPosition):
        if not isinstance(center, Point) and not isinstance(center, dict):
            raise TypeError("The center must be given as a point or a dict")

        if isinstance(center, Point):
            self.Center = center
        else:
            if 'x' not in center.keys() or 'y' not in center.keys() or 'z' not in center.keys():
                raise ValueError("The center dict must be on the form dict(x: float, y: float, z: Optional[float}")
            self.Center = Point(x=float(center.get('x')), y=float(center.get('y')), z=float(center.get('z')))

        if sum([self.Center.x is None, self.Center.y is None]) > 0:
            raise ValueError("At least the x and y positions of the center must be specified")
