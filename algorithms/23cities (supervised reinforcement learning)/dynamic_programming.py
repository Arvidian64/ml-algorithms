import itertools

#read file & create a graph representation of the cities and distances
def read_file_to_graph():
    city_graph = {}
    with open('city1.txt', 'r') as f:
        for line in itertools.islice(f, 3, None): 
            name_1, name_2, distance = line.strip().split(' ')
            city_graph.setdefault(name_1, []).append((name_2, int(distance)))
            city_graph.setdefault(name_2, []).append((name_1, int(distance)))
    return city_graph

#"dynamic programming" algorithm 
def floyd_warshall(city_graph):
    cities = list(city_graph.keys())
    num_cities = len(cities)
    city_indexes = {}
    for idx, city in enumerate(cities):
        city_indexes.update({city: idx})

    #initialize matrixes of distances and cities
    distance = [[float('inf')] * num_cities for _ in range(num_cities)]
    next_city = [[None] * num_cities for _ in range(num_cities)]  

    #fill in matrix 
    for i in range(num_cities):
        distance[i][i] = 0 #diagonal is 0 
    for city, neighbors in city_graph.items():
        for neighbor, weight in neighbors:
            i, j = city_indexes[city], city_indexes[neighbor]
            distance[i][j] = weight 
            next_city[i][j] = neighbor 

    #updating matrixes
    #(m for possible "middle" cities(between s & d), s for start city, d for destination city)
    for m in range(num_cities):
        for s in range(num_cities):
            for d in range(num_cities):
                #this is translated from a C++ implementation 
                if distance[s][m] != float('inf') and distance[m][d] != float('inf') and distance[s][d] > distance[s][m] + distance[m][d]:
                    distance[s][d] = min(distance[s][d], distance[s][m] + distance[m][d])
                    next_city[s][d] = next_city[s][m]

    return cities, city_indexes, distance, next_city

#prints minimum travel cost from every city to the goal city
def road_optimal_values(goal, cities, city_indexes, distance):
    F = city_indexes[goal]
    for city in cities:
        C = city_indexes[city]
        print(C)
        if distance[C][F] != float('inf'):
            print(f'Shortest distance from {city} to {goal} is: {distance[C][F]}')

#prints shorest path (the one with the smallest total distance)
def print_shorest_path(cities, city_indexes, next_city, goal):
    F= city_indexes[goal]
    for city in cities: 
        C = city_indexes[city] 
        path = []
        while C is not None:
            path.append(cities[C])
            C = city_indexes[next_city[C][F]] if next_city[C][F] is not None else None              
        print('Path:', ' -> '.join(path))

def main():
    city_graph = read_file_to_graph()
    cities, city_indexes, distance, next_city = floyd_warshall(city_graph)
    goal = 'F'

    road_optimal_values(goal, cities, city_indexes, distance)
    print_shorest_path(cities, city_indexes, next_city, goal)

if __name__ == "__main__":
    main()