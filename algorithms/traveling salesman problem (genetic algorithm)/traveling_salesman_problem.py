import random
import matplotlib.pyplot as plt

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

#Runs a tournament for each parent, chooses individual with lowest distance-index
def tournament_selection(population, fitness_scores, tournament_size=5):
    tournament_winners = []
    for _ in range(2):
        participants = random.sample(range(len(population)), tournament_size)
        best = min(participants, key=lambda index: fitness_scores[index])
        tournament_winners.append((population[best]))
        
    return tournament_winners[0], tournament_winners[1]

#Cuts each parent in half, each child gets a half of each parent
def crossover(parent_one, parent_two):
    cut_parent_in_half = int(len(parent_one) /2)

    child_one = parent_one[:cut_parent_in_half]
    child_two = parent_two[:cut_parent_in_half]

    child_one.extend([city for city in parent_two if city not in child_one])
    child_two.extend([city for city in parent_one if city not in child_two])

    return child_one, child_two

def genetic_algorithm(pop_size, generations, mutation_rate, city_dict):
    global_best_distance = float("inf")
    global_best_chromosome = None

    #logs - maybe ?for 5. Illustrate how performance of the population evolves with generations, with a figure in assigment ?
    best_fitness_scores = []
    mean_fitness_scores = []

    population = generate_population(pop_size)

    for gen in range(generations):
        new_population = []

        fitness_scores = [evaluate_solution(chromosome, city_dict) for chromosome in population]
       # print('FITNESS', sorted(fitness_scores))
        #for logging
        best_fitness_scores.append(min(fitness_scores))
        mean_fitness_scores.append(sum(fitness_scores) / len(fitness_scores))
        
        
        best_chromosome_index = fitness_scores.index(min(fitness_scores))
        best_chromosome_distance = fitness_scores[best_chromosome_index]
       # print('BEST CHROMOSEOME INDEX', best_chromosome_index)
       # print('BEST DISTANCE', best_chromosome_distance)
        
        while len(new_population) < len(population):

            parent_one, parent_two = tournament_selection(population, fitness_scores)

            child_one, child_two = crossover(parent_one, parent_two)
            child_one = mutate_inversion(child_one, mutation_rate)
            child_two = mutate_inversion(child_two, mutation_rate)

            new_population.append(child_one)

            if len(new_population) < pop_size:
                new_population.append(child_two)

        population = new_population
        #print(population)
    best_solution = population[fitness_scores.index(min(fitness_scores))]
    print(best_solution)
    
        
    for city_list in population:
        eval_distance = evaluate_solution(city_list, city_dict)
        if eval_distance < global_best_distance:
            global_best_distance = eval_distance
            global_best_chromosome = city_list
            print(f"Gen{gen + 1:}\nNew optimal solution at distance: {eval_distance}")
    for i in range(len(population)):
        city_list = population.pop(0)
        population.append(mutate_inversion(city_list, mutation_rate))

    plots(best_fitness_scores, mean_fitness_scores)

    return global_best_chromosome, global_best_distance
    #return best_solution

def plots(best_fitness_scores, mean_fitness_scores):
    plt.figure(figsize=(15, 10))
    plt.plot(best_fitness_scores, label="BEST DISTANCE")
    plt.plot(mean_fitness_scores, label="MEAN DISTANCE")
    plt.xlabel("GENERATION")
    plt.ylabel("DISTANCE")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=8000, color="red")
    plt.savefig("TSP_distance_plot.png")
    plt.show()

def main():
    # city_list = generate_city_list()
    # print(city_list)
    # print(city_dict)
    # solution_distance = evaluate_solution(city_list, city_dict)
    # print(solution_distance)

    city_dict = import_city_dict()
                                        #popsize, generations, mut-rate
    solution, distance = genetic_algorithm(256, 256, 0.35, city_dict)
    #best_solution = genetic_algorithm(256, 256, 0.7, city_dict)

    print(f"-----------\nBest solution found had a distance of:\n{distance}\nCity list:\n{solution}")
    #print(f"-----------\nBest solution found had a distance of:\n{best_solution}")
    
if __name__ == "__main__":
    main()
