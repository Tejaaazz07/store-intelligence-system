def get_zone(x_center):

    if x_center < 250:
        return "left_zone"

    elif x_center < 500:
        return "middle_zone"

    else:
        return "right_zone"