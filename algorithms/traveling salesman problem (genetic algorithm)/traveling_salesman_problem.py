import random

def import_city_dict():
    city_dict = {}
    with open('berlin52.tsp') as f:
        for line in f:
            parts = line.split(" ")
            try:
                city_dict[int(parts[0])] = (int(float(parts[1])), int(float(parts[2])))
            except:
                pass
    return city_dict

def generate_city_list():
    city_list = []
    for i in range(0, 52):
        city_list.append(i+1)
    random.shuffle(city_list)
    return city_list

def initial_population(pop_size):
    population = []
    for i in range(pop_size):
        population.append(generate_city_list())

def euclidean_distance(city1, city2):
    distance = ((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)**0.5
    return distance

def evaluate_solution(city_list, city_dict):
    distance: float = 0
    for i in range(-1, len(city_list)-1):
        distance += euclidean_distance(city_dict[city_list[i]], city_dict[city_list[i+1]])
    return distance

def genetic_algorithm(pop_size,generations, mutation_rate, city_dict):

    global_best_distance = float("inf")
    global_best_chromosome = None

    population = initial_population(pop_size)

    for gen in range(generations):



def main():
    city_list = generate_city_list()
    print(city_list)
    city_dict = import_city_dict()
    print(city_dict)
    solution_distance = evaluate_solution(city_list, city_dict)

    print(solution_distance)



if __name__ == "__main__":
    main()

