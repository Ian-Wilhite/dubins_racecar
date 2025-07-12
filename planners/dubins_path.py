import math
import matplotlib.pyplot as plt

def dubins_path_planning(s_x, s_y, s_yaw, g_x, g_y, g_yaw, curvature, step_size=0.1):
    """
    Dubins path planning
    :param s_x: x position of the start point [m]
    :param s_y: y position of the start point [m]
    :param s_yaw: yaw angle of the start point [rad]
    :param g_x: x position of the goal point [m]
    :param g_y: y position of the end point [m]
    :param g_yaw: yaw angle of the end point [rad]
    :param curvature: curvature for curve [1/m]
    :param step_size: step size between two path points [m]
    :return: x_list (array) – x positions of the path
             y_list (array) – y positions of the path
             yaw_list (array) – yaw angles of the path
             modes (array) – mode list of the path
             lengths (array) – length list of the path segments
    """

    ex = g_x - s_x
    ey = g_y - s_y
    lex = math.sqrt(ex**2 + ey**2)
    lex_c = lex * curvature

    # calculate start and goal circles
    l_sx = s_x - math.sin(s_yaw) / curvature
    l_sy = s_y + math.cos(s_yaw) / curvature
    r_sx = s_x + math.sin(s_yaw) / curvature
    r_sy = s_y - math.cos(s_yaw) / curvature
    l_gx = g_x - math.sin(g_yaw) / curvature
    l_gy = g_y + math.cos(g_yaw) / curvature
    r_gx = g_x + math.sin(g_yaw) / curvature
    r_gy = g_y - math.cos(g_yaw) / curvature

    # calculate LSL, RSR, LSR, RSL, RLR, LRL paths
    paths = []
    paths.append(LSL(l_sx, l_sy, l_gx, l_gy, lex_c))
    paths.append(RSR(r_sx, r_sy, r_gx, r_gy, lex_c))
    paths.append(LSR(l_sx, l_sy, r_gx, r_gy, lex_c))
    paths.append(RSL(r_sx, r_sy, l_gx, l_gy, lex_c))
    paths.append(RLR(r_sx, r_sy, l_gx, l_gy, lex_c))
    paths.append(LRL(l_sx, l_sy, r_gx, r_gy, lex_c))

    paths = [p for p in paths if p is not None and p[0] is not None]
    # find the shortest path
    min_path_len = float("inf")
    best_path_index = -1
    for i, path in enumerate(paths):
        if path[0] <= 0.0:
            continue
        if min_path_len > path[0]:
            min_path_len = path[0]
            best_path_index = i

    # generate the best path
    path_x, path_y, path_yaw, mode, lengths = generate_course(
        s_x, s_y, s_yaw, paths[best_path_index], curvature, step_size
    )

    return path_x, path_y, path_yaw, mode, lengths


def LSL(c_sx, c_sy, c_gx, c_gy, lex_c):
    return dubins_path(c_sx, c_sy, c_gx, c_gy, lex_c)


def RSR(c_sx, c_sy, c_gx, c_gy, lex_c):
    return dubins_path(c_sx, c_sy, c_gx, c_gy, lex_c)


def LSR(c_sx, c_sy, c_gx, c_gy, lex_c):
    lex_c_sq = lex_c**2
    d_sq = (c_sx - c_gx)**2 + (c_sy - c_gy)**2
    if d_sq >= lex_c_sq:
        d = math.sqrt(d_sq)
        theta = math.atan2(c_gy - c_sy, c_gx - c_sx)
        t = -theta
        p = math.sqrt(max(0.0, d_sq - lex_c_sq))
        q = theta
        return t, p, q
    else:
        return None, None, None


def RSL(c_sx, c_sy, c_gx, c_gy, lex_c):
    lex_c_sq = lex_c**2
    d_sq = (c_sx - c_gx)**2 + (c_sy - c_gy)**2
    if d_sq >= lex_c_sq:
        d = math.sqrt(d_sq)
        theta = math.atan2(c_gy - c_sy, c_gx - c_sx)
        t = theta
        p = math.sqrt(max(0.0, d_sq - lex_c_sq))
        q = -theta
        return t, p, q
    else:
        return None, None, None


def RLR(c_sx, c_sy, c_gx, c_gy, lex_c):
    d_sq = (c_sx - c_gx)**2 + (c_sy - c_gy)**2
    if d_sq <= (2 * lex_c)**2:
        d = math.sqrt(d_sq)
        theta = math.atan2(c_gy - c_sy, c_gx - c_sx)
        t = theta - math.acos(d / (2 * lex_c))
        p = 2 * math.pi - math.acos(d / (2 * lex_c))
        q = -t
        return t, p, q
    else:
        return None, None, None


def LRL(c_sx, c_sy, c_gx, c_gy, lex_c):
    d_sq = (c_sx - c_gx)**2 + (c_sy - c_gy)**2
    if d_sq <= (2 * lex_c)**2:
        d = math.sqrt(d_sq)
        theta = math.atan2(c_gy - c_sy, c_gx - c_sx)
        t = -theta + math.acos(d / (2 * lex_c))
        p = 2 * math.pi - math.acos(d / (2 * lex_c))
        q = -t
        return t, p, q
    else:
        return None, None, None


def dubins_path(c_sx, c_sy, c_gx, c_gy, lex_c):
    d_sq = (c_sx - c_gx)**2 + (c_sy - c_gy)**2
    d = math.sqrt(d_sq)
    theta = math.atan2(c_gy - c_sy, c_gx - c_sx)
    t = -theta
    p = d
    q = theta
    return t, p, q


def generate_course(s_x, s_y, s_yaw, path, curvature, step_size):
    path_x = [s_x]
    path_y = [s_y]
    path_yaw = [s_yaw]

    t, p, q = path
    lengths = [t, p, q]
    mode = ["L", "S", "L"]

    # Generate first segment
    for _ in range(int(t / step_size)):
        s_x += step_size * math.cos(s_yaw)
        s_y += step_size * math.sin(s_yaw)
        s_yaw += step_size * curvature
        path_x.append(s_x)
        path_y.append(s_y)
        path_yaw.append(s_yaw)

    # Generate second segment
    for _ in range(int(p / step_size)):
        s_x += step_size * math.cos(s_yaw)
        s_y += step_size * math.sin(s_yaw)
        path_x.append(s_x)
        path_y.append(s_y)
        path_yaw.append(s_yaw)

    # Generate third segment
    for _ in range(int(q / step_size)):
        s_x += step_size * math.cos(s_yaw)
        s_y += step_size * math.sin(s_yaw)
        s_yaw += step_size * curvature
        path_x.append(s_x)
        path_y.append(s_y)
        path_yaw.append(s_yaw)

    return path_x, path_y, path_yaw, mode, lengths
