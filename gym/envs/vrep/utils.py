import numpy


def goal_func_3(delta, thres):
    norm = numpy.linalg.norm(delta)
    if norm <= thres[0]:
        return numpy.exp(-norm)
    elif norm <= thres[1]:
        return 0
    else:
        return -1 * (norm - thres[1])


def goal_func_2(delta, thres):
    norm = numpy.linalg.norm(delta)
    return numpy.exp(-norm) if norm <= thres else 0.0


def aux_func(delta, thres):
    norm = numpy.linalg.norm(delta)
    return (numpy.exp(-norm) - 1) if norm <= thres else -1.0
