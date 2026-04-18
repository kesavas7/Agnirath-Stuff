def find_max_energy_segment(segments, telemetry):
    PANEL_EFFICIENCY = 0.21
    PANEL_AREA = 4.6
    #name of all the single letter variables have changed to proper ones
    #a dictionary is created to solve the time complexity problem 
    #like when it had 2 for loop the time complexity was o(m*n) 
    #and now it is o(m+n) since it is a dictionary
    energy_map = {}
    for seg_id, irradiance, factor in telemetry:
        energy = irradiance * factor * PANEL_EFFICIENCY * PANEL_AREA
        energy_map[seg_id] = energy

    max_energy = float('-inf')
    best_segment = None

    for seg in segments:
        if seg in energy_map:
            if energy_map[seg] > max_energy:
                max_energy = energy_map[seg]
                best_segment = seg

    return best_segment
