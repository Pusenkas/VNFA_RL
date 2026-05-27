# @title
import csv
import numpy as np
from geopy import distance

SPEED_OF_LIGHT = 3 * 10**8

def parse_sfc_data(filename, sfc_size=None):
    data = dict()
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if line[5] != 'Terminated' or line[7] == '':
                continue
            job = int(line[2])
            data[job] = data.get(job, []) + [float(line[7])]
    data = list(data.values())
    result = []
    for elem in data:
        if 2 < len(elem) < 9:
            result.append(elem)
    if sfc_size:
        result = result[:sfc_size]

    resource_req = []
    F = 0
    for i in range(len(result)):
        for j in range(len(result[i])):
            resource_req.append(result[i][j])
            result[i][j] = i
            F += 1
        result[i] = len(result[i])
    return F, np.array(resource_req, dtype=np.float32), np.array(result, dtype=int)

def sample_sfc(sfc, resources, sfc_size=None, seed=42):
    if sfc_size == None:
        sfc_size = sfc.size
    rng = np.random.default_rng(seed=seed)
    sfc_sample = rng.choice(sfc, size=sfc_size)
    function_amount = np.sum(sfc_sample)
    resources_sample = rng.choice(resources, size=function_amount)
    return function_amount, resources_sample, sfc_sample
    
    

def parse_edge_data(user_data_filename, node_data_filename, user_data_size=None, node_data_size=None):
    applications = []
    with open(user_data_filename, 'r') as file:
        reader = csv.reader(file)
        header = list(next(reader))
        for line in reader:
            applications.append((float(line[0]), float(line[1])))

    compute_nodes = []
    with open(node_data_filename, 'r') as file:
        reader = csv.reader(file)
        header = list(next(reader))
        for line in reader:
            compute_nodes.append((float(line[1]), float(line[2])))

    if user_data_size:
        while len(applications) < user_data_size:
            applications += applications
        applications = applications[:user_data_size]
    if node_data_size:
        while len(compute_nodes) < node_data_size:
            compute_nodes += compute_nodes
        compute_nodes = compute_nodes[:node_data_size]

    delay1_distance = []
    for i, app in enumerate(applications):
        delay1_distance.append([])
        for comp in compute_nodes:
            delay1_distance[i].append(distance.distance(app, comp).m)

    delay2_distance = []
    for i, comp1 in enumerate(compute_nodes):
        delay2_distance.append([])
        for comp2 in compute_nodes:
            delay2_distance[i].append(distance.distance(comp1, comp2).m)

    delay1 = []
    for distances in delay1_distance:
        delay1.append([1000 * distance / SPEED_OF_LIGHT for distance in distances])

    delay2 = []
    for distances in delay2_distance:
        delay2.append([1000 * distance / SPEED_OF_LIGHT for distance in distances])

    return (len(delay1), len(delay2), np.array(delay1, dtype=np.float32), np.array(delay2, dtype=np.float32))