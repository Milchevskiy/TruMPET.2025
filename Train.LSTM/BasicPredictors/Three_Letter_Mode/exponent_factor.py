import math

def exponent_factor(point_number: int, shift: int) -> float:
    argument = 6 * math.fabs(float(shift)) / point_number
    value = math.exp(- (argument ** 2) / 2)
    return value
