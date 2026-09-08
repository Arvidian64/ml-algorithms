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

def generate_population(pop_size):
    population = []
    for i in range(pop_size):
        population.append(generate_city_list())
    return population

def tournament_selection(population, tournament_size, city_dict):
    parents = []
    for i in range(len(population)):
        participants = random.sample(population, tournament_size)

        best_candidate = None
        shortest_distance = float("inf")

        for participant in participants:
            dist = evaluate_solution(participant, city_dict)
            if dist < shortest_distance:
                shortest_distance = dist
                best_candidate = participant

        parents.append(best_candidate)

    return parents

def ordered_crossover(parent1, parent2):
    size = len(parent1)
    # Väljer två slumpmässiga platser att skära på
    i, j = sorted(random.sample(range(size), 2))

    child = [None] * size
    # Kopierar förälder ett
    child[i : j + 1] = parent1[i : j + 1]
    # Använder ett set för att dubbelkolla medlemskap/dubletter
    copied_cities = set(child[i : j + 1])

    # Startindex blir direkt efter där vi placerat in förälder ett
    child_idx = (j + 1) % size
    # Fyller med resterande städer från förälder två i så bevarad ordning som möjligt
    for step in range(size):
        p2_idx = (j + 1 + step) % size
        candidate_city = parent2[p2_idx]

        if candidate_city not in copied_cities:
            child[child_idx] = candidate_city
            child_idx = (child_idx + 1) % size

    return child

def mutate_inversion(city_list, mutation_rate):
    if random.random() > mutation_rate:
        return city_list

    mutated = city_list.copy()

    i,j = random.sample(range(len(mutated)), 2)

    start = min(i, j)
    end = max(i, j)

    if end - start < 2:
        start -= random.randint(2, 26)

    # Slicing and reversing a piece happens here
    mutated[start : end + 1] = reversed(mutated[start : end + 1])

    return mutated

def euclidean_distance(city1, city2):
    distance = ((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)**0.5
    return distance

def evaluate_solution(city_list, city_dict):
    distance: float = 0
    for i in range(-1, len(city_list)-1):
        distance += euclidean_distance(city_dict[city_list[i]], city_dict[city_list[i+1]])
    return distance

def genetic_algorithm(pop_size, generations, mutation_rate, city_dict):

    global_best_distance = float("inf")
    global_best_chromosome = None

    population = generate_population(pop_size)

    for gen in range(generations):
        distances = []
        for city_list in population:
            eval_distance = evaluate_solution(city_list, city_dict)
            distances.append(eval_distance)
            if eval_distance < global_best_distance:
                global_best_distance = eval_distance
                global_best_chromosome = city_list
                print(f"Gen{gen + 1:}\nNew optimal solution at distance: {eval_distance}")
        parents = tournament_selection(population, 16, city_dict)
        for j in range(0, len(parents), 2):
            population.pop(0)
            population.append(ordered_crossover(parents[j], parents[j + 1]))
        for i in range(len(population)):
            city_list = population.pop(0)
            population.append(mutate_inversion(city_list, mutation_rate))


    return global_best_chromosome, global_best_distance


def main():
    # city_list = generate_city_list()
    # print(city_list)
    # print(city_dict)
    # solution_distance = evaluate_solution(city_list, city_dict)
    # print(solution_distance)

    city_dict = import_city_dict()
    solution, distance = genetic_algorithm(256, 256, 0.08, city_dict)

    print(f"-----------\nBest solution found had a distance of:\n{distance}\nCity list:\n{solution}")



if __name__ == "__main__":
    main()