

def compute_lap_time(reward_log, dt):
    return len(reward_log) * dt

def deviation_from_center(track, positions):
    from shapely.geometry import Point
    return [track.distance(Point(x, y)) for x, y in positions]


def success_criteria(info):
    return info.get("distance", 0) >= 0.99 * info.get("lap_length", 1.0)
