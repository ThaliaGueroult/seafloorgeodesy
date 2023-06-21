import math
import numpy as np

def calc_ray_path(distance, y_d, y_s, l_depth, l_sv, nlyr):
    '''
    calculate ray path and travel time (one-way)
    input
        distance: horizontal distance between both ends
        y_d     : height (depth) of deeper end (< 0)
        y_s     : height (depth) of shallower end (< 0)
        l_depth : depth at each node
        l_sv    : sound speed at each node
        nlyr    : number of node for sound speed profile

    output
        t_ag : takeoff angle (in rad. ; Zenith direction = pi)
        t_tm : one-way travel time
    '''
    
    # Setting parameters
    loop1 = 200
    loop2 = 20
    eps1 = 1.0e-7
    eps2 = 1.0e-14

    # Initialize variables
    x_hori = np.zeros(6)
    tadeg = np.array([0.0, 20.0, 40.0, 60.0, 70.0, 70.0])
    ta_rough = np.pi * (180.0 - tadeg) / 180.0

    layer_thickness = l_depth[1:] - l_depth[:-1]
    layer_sv_trend = (l_sv[1:] - l_sv[:-1]) / layer_thickness

    # Set layer numbers and sound speeds at both ends
    layer_d, sv_d = layer_setting(nlyr, y_d, l_depth, l_sv, layer_sv_trend)
    layer_s, sv_s = layer_setting(nlyr, y_s, l_depth, l_sv, layer_sv_trend)

    x_hori = np.zeros(6)
    r_nm = 0

    # Rough scan for take-off angle
    for i in range(5):
        if i == 4:
            break

        j = i + 1
        # Calculate horizontal distance (x_hori) for the given takeoff angle (ta_rough[j])
        x_hori[j], a0 = ray_path(ta_rough[j], nlyr, layer_d, layer_s, sv_d, sv_s, y_d, y_s, l_depth, l_sv)

        # Negative distance does not make sense
        if x_hori[j] < 0.0:
            continue

        # Only if x_hori[i] <= distance <= x_hori[i+1],
        # takeoff angle should be between ta_rough[i] and ta_rough[i+1]
        diff1 = distance - x_hori[i]
        diff2 = x_hori[j] - distance
        if diff1 * diff2 < 0.0:
            continue

        r_nm = 1

        # Rename variables
        x1 = x_hori[i]
        x2 = x_hori[j]
        t_angle1 = ta_rough[i]
        t_angle2 = ta_rough[j]

        # Detailed search for takeoff angle (conv. criteria is "eps1")
        for k in range(loop1):
            x_diff = x1 - x2  # Horizontal difference of two paths
            if abs(x_diff) < eps1:
                break

            # Calculate horizontal distance (x0) for averaged takeoff angle (t_angle0)
            # x0 should be x1 or x2 (depending on the sign of x0-distance)
            t_angle0 = (t_angle1 + t_angle2) / 2.0
            x0, a0 = ray_path(t_angle0, nlyr, layer_d, layer_s, sv_d, sv_s, y_d, y_s, l_depth, l_sv)

            a0 = -a0
            t0 = distance - x0
            if t0 * diff1 > 0.0:
                x1 = x0
                t_angle1 = t_angle0
            else:
                x2 = x0
                t_angle2 = t_angle0

            # Search ends when x1 - x2 is converged (conv. criteria is "eps1")
            if abs(x1 - x2) < eps1:
                break

        # Travel time (t_nm) is obtained for the correct takeoff angle (ta_nm)
        ta_nm = (t_angle1 + t_angle2) / 2.0
        t_nm, a_nm = ray_path(ta_nm, nlyr, layer_d, layer_s, sv_d, sv_s, y_d, y_s, l_depth, l_sv)

        return t_nm

    if r_nm == 0:
        return -999.9

def layer_setting(nlyr, y, l_depth, l_sv, layer_sv_trend):
    layer = 0
    sv = 0.0

    for i in range(nlyr):
        if y > l_depth[i] and y <= l_depth[i + 1]:
            layer = i + 1
            sv = l_sv[i] + layer_sv_trend[i] * (y - l_depth[i])
            break

    return layer, sv

def ray_path(t_angle, nlyr, layer_d, layer_s, sv_d, sv_s, y_d, y_s, l_depth, l_sv):
    x = np.zeros(6)
    a = np.zeros(6)
    x_c = np.zeros(6)
    z_c = np.zeros(6)
    x_s = np.zeros(6)
    z_s = np.zeros(6)

    x[0] = 0.0
    a[0] = 0.0
    x_c[0] = 0.0
    z_c[0] = y_d
    x_s[0] = 0.0
    z_s[0] = y_s

    for i in range(nlyr):
        layer = layer_d[i]
        sv = sv_d[i]
        z_c[i + 1] = z_c[i] + layer_d[i]
        x_c[i + 1] = x_c[i] + layer_d[i] / math.tan(a[i]) * (1.0 - sv / sv_s)

        layer = layer_s[i]
        sv = sv_s[i]
        z_s[i + 1] = z_s[i] + layer_s[i]
        x_s[i + 1] = x_s[i] + layer_s[i] / math.tan(a[i]) * (1.0 - sv / sv_s)

        if i < nlyr - 1:
            x[i + 1] = x_c[i + 1]
            a[i + 1] = a[i]
        else:
            x[i + 1] = x_s[i + 1]
            a[i + 1] = a[i]

    x_diff = x_s[nlyr] - x_c[nlyr]
    a_diff = a[nlyr] - math.tan(t_angle)

    return x_diff, a_diff

# Example usage
distance = 100.0
y_d = 0.0
y_s = 0.0
l_depth = np.array([0.0, 1000.0, 2000.0, 3000.0])
l_sv = np.array([1500.0, 1800.0, 2000.0, 2200.0])
nlyr = 3

travel_time = calc_ray_path(distance, y_d, y_s, l_depth, l_sv, nlyr)
print(f"Travel time: {travel_time}")
