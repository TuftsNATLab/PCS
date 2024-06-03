import numpy as np

flat_map = lambda f, xs: [y for ys in xs for y in f(ys)]

def get_cdf(data):
    """Returns the CDF of the given data.
    
       Args:
           data: A list of numerical values.
           
       Returns:
           An pair of lists (x, y) for plotting the CDF.
    """

    sorted_data = sorted(data)
    p = 1.0/len(sorted_data)
    p = 1. * np.arange(start=1,stop=1+len(sorted_data)) / (len(sorted_data))
    return sorted_data, p


def swap(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]

def within(src, target):
    if abs(src - target) < 1e-5:
        return True
    return False

def util_print(tl):
    print("================")
    for t in tl:
        print(t.task_id)
    print("================")

def util_membership(m, s):
    for t in s:
        if m.task_id == t.task_id:
            return True
    return False


def util_print_heap(h):
    h_temp = h[:]

    print("==================================")
    while len(h_temp) > 0:
        event = heappop(h_temp)
        print("event_type: %s event_time: %f event_id: %d" % (event.event_type, event.event_time, event.event_id))
    print("==================================")


def load_cdf(fname,delimiter=','):
    with open(fname) as fd:
        file_data = fd.readlines()[1:]
    data = []

    for entry in file_data:
        val, cdf = entry.rstrip().split(delimiter)
        data.append([float(val), float(cdf)])

    assert(data[-1][1] == 1.0)

    return data

def gen_data_from_cdf(cdf_fname, num_points=1000, dtype=int, interpolation=True):


    cdf = load_cdf(cdf_fname)

    data = []
    for _ in range(num_points):
        point = np.random.uniform(0.0, 1.0)
        for i in range(len(cdf)):
            if point <= cdf[i][1]:
                if i == 0:
                    if interpolation:
                        data.append(dtype(np.random.uniform(0,cdf[i][0])))
                    else:
                        data.append(dtype(cdf[i][0]))
                else:
                    if interpolation:
                        data.append(dtype(np.random.uniform(cdf[i-1][0],cdf[i][0])))
                    else:
                        data.append(dtype(cdf[i][0]))
                break
    return data


def comp_thresholds(job_sizes, cov_thresh=[1.0]):
    

    job_sizes = sorted(job_sizes)

    thresholds = []
    
    n=0
    mu=0
    s=0
    s2=0
    cov = 0


    class_num = 0


    for i in range(len(job_sizes)):
        xn = job_sizes[i]
        s+=xn
        s2+=(xn*xn)
        n+=1
        mu = ((n-1.0)*mu + xn)/n
        var = (1.0/n)*(s2 + (n*mu*mu) - (mu*2.0*s))

        cov = var/(mu*mu)


        if cov > cov_thresh[class_num % len(cov_thresh)]:
            thresholds.append(int(job_sizes[i-1]))
            n=0
            mu=0
            s=0
            s2=0
            class_num += 1


    thresholds = thresholds + [float('inf')]    
    return thresholds



def app_to_pt(gpus, delta):
    

    assert(len(gpus) == len(delta))

    total_service = sum(map(lambda g, t: g * t, gpus, delta))

    t_service = [total_service]

    for i in range(len(gpus)):
        t_service.append(t_service[-1] - (gpus[i] * delta[i]))

    p = lambda l1,l2, t: 0.5 * t * (l1+l2)
    
    t_priority=0

    for i in range(len(t_service)-1):
        t_priority += p(t_service[i], t_service[i+1], delta[i])
    return t_priority

