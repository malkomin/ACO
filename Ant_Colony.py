import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
cities = ['A','B','C','D','E','F','G']
path = ['A', 'F', 'D', 'E']
def create_environment(cities, path):
    '''
    :param cities: Cities in environment
    :param path: The most efficient route of environment
    :return:
     phe :: Pheromone levels of cities
     env :: Distances between cities
    '''
    def distance_matrix(cities, path):
        N =len(cities)
        env = pd.DataFrame(data=np.ones((N,N)) * 10, columns=cities, index = cities)
        ## -1: disallowed passage
        for i in range(N):
            env.loc[cities[i], cities[i]] = -1

        for i in range(len(path)-1):
            env.loc[path[i], path[i+1]] = 1
            env.loc[path[i+1], path[i]] = 1

        return env

    def pheremon_matrix(cities, eps = 0.0001):
        """

        :param cities: All cities in environment
        :param eps: Default pheromone level on paths
        :return: distance matrix , pheromone matrix
        """
        N =len(cities)
        return pd.DataFrame(data=np.ones((N,N)) * eps, columns=cities, index = cities)

    return distance_matrix(cities, path), pheremon_matrix(cities)


env, phe = create_environment(cities,path)

class Ant:
    def __init__(self,number, env=env, phe=phe,
                 start='A', end='E',
                 alpha=0.5, beta=0.5):
        '''
        :param number: Unique number of ant
        :param env: Represents distance matrix
        :param phe: Pheromone matrix
        :param start: Starting point
        :param end: Destination point
        :param alpha: Constant for calculate probability  default = 0.5
        :param beta: Constant for calculate probability  default = 0.5
        After 0.6 becoming non-realistic like all ants are so smart or dummy,they always chooses same way that other ants choice
        :param road: Next possible cities
        :param cities: All cities


        '''
        self.number = number
        self.env, self.phe = env, phe
        self.alpha, self.beta = alpha, beta
        self.cities = list(self.env.columns)
        self.start = start
        self.current_city = start
        self.destination = end
        self.route = [self.current_city]
        self.road = self.cities.copy()
        self.distances = env.loc[self.current_city, env.loc[self.current_city] > 0]
        self.pheromones = phe.loc[self.current_city, env.loc[self.current_city] > 0]

    def move(self):
        """
        move :: Moving ant to the next city.
        :return:
        """
        if self.current_city == self.destination: #Check if ant reaches the destination point
            return
        self.road.remove(self.current_city)
        self.distances = env.loc[self.current_city, env.loc[self.current_city] > 0][self.road]  # Distances matrix
        self.pheromones = phe.loc[self.current_city, env.loc[self.current_city] > 0][self.road]  # Pheromones matrix
        preferences = (self.pheromones ** self.alpha) / (self.distances ** self.beta)  # Formula for evaluate possibility
        probabilities = preferences / preferences.sum()  # Ant's possibilities for next city
        #print("Ant ", self.number , " prob :\n" , probabilities)
        self.current_city = np.random.choice(a=probabilities.index,
                                                 size=1,
                                                 p=probabilities.values)[0]  #Choosing city
        #Adding city to the route and update pheromone of the route that ant passed
        self.route.append(self.current_city)
        self.positive_pheromone_update()


    def positive_pheromone_update(self, delta =1):
        """
        Spreads ant's pheromone to the road
        :param delta: A constant
        """
        i, j = self.route[-1], self.route[-2]
        phe[i][j] += delta / env[i][j]
        phe[j][i] += delta / env[j][i]

    def cost(self):
        """
        :return: Calculates ants cost while going destination then returns it.
        """
        result = 0
        for i in range(len(self.route) - 1):
            result += env.loc[self.route[i], self.route[i + 1]]
        return result



class Ant_Colony:
    def __init__(self, ant_count=100, env =env,phe = phe):
        """
        :param ant_count: Counts of ants
        :param env: Distances matrix
        :param phe: Pheromone matrix
        :param ants: All ants
        """
        self.count = ant_count
        self.ants = []
        for i in range(ant_count):
            self.ants.append(Ant(i+1,env,phe))


    def negative_pheromone_update(self,ant ,env_rate=0.01):
        """
        Updates pheromone level on previous ways after ant passes
        :param ant: Current ant
        :param env_rate: Decay rate
        :return:
        """
        if len(ant.route) <= 2: #If ant just passed to second city, no need for decay effect.
            return
        else:
            previous = ant.route[:-1] #Previous routes
            for i in range(len(previous),0,1):
                j = i-1
                phe[previous[i]][previous[j]] *= 1-env_rate
                phe[previous[j]][previous[i]] *= 1-env_rate

    def start(self,time = 10):
        """
        Starts scavenging and ants move
        :param time: How many times ant will (iteration)
        :return:
        """
        for t in range(time):
            for ant in self.ants:
                ant.current_city = ant.start
                ant.route = [ant.current_city]
                ant.road = ant.cities.copy()
                for i in range(len(ant.cities)):
                    ant.move() #Ant move
                    self.negative_pheromone_update(ant) #Update pheromone on passed ways
            

    def route(self):
        """
        :return: Prints all ant's route in colony
        """
        for ant in self.ants:
            print("Ant ", ant.number, " route: ", ant.route)

colony = Ant_Colony(50) #colony
colony.start(10)
colony.route()
cost = [(a.cost(), a.route) for a in colony.ants]
print("Cost : " , cost )
idx = 0
list1 = []
for i in range(len(cost)):
    list1.append(cost[i][0])
idx = list1.index(min(list1))
print("Best route : ", cost[idx])
print(phe)







