class GreedyError(Exception):
    """Exception raised when resources are insufficient."""
    pass

def find_greedy_solution1(function_amount, app_amount, comp_amount, app_comp_delay, comp_comp_delay, sfc, available_resources, required_resources):
    res = []
    resource_left = available_resources.copy()
    counter = 0
    sfc_number = 0
    while counter < function_amount:
        res.append([])
        delays = sorted(enumerate(app_comp_delay[sfc_number]), key=lambda x: x[1])
        for node, node_delay in delays:
            if resource_left[node] >= required_resources[counter]:
                res[sfc_number].append(node)
                resource_left[node] -= required_resources[counter]
                prev = node
                break
        else:
            raise GreedyError("")
        counter += 1
        for i in range(sfc[sfc_number] - 1):
            delays = sorted(enumerate(comp_comp_delay[prev]), key=lambda x: x[1])
            for node, node_delay in delays:
                if resource_left[node] >= required_resources[counter]:
                    res[sfc_number].append(node)
                    resource_left[node] -= required_resources[counter]
                    prev = node
                    break
            else:
                raise GreedyError("")
            counter += 1
        sfc_number += 1
    return res

def find_obj(app_comp_delay, comp_comp_delay, res):
    delay_arr = []
    for user, placements in enumerate(res):
        prev = placements[0]
        delay = app_comp_delay[user][prev]
        for node in placements[1:]:
            delay += comp_comp_delay[prev][node]
            prev = node
        delay_arr.append(delay)    
    # print(delay_arr)
    return max(delay_arr)
