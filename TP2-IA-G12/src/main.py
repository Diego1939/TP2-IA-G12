import config
import values
import csv
import random
from deap import base, creator, tools, algorithms

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

musico_id_counter = 0

def generar_musico():
    global musico_id_counter
    musico = {
        "id": musico_id_counter,
        "tipo": random.choice(values.TIPOS),
        "habilidad_tecnica": random.randint(50, 100),
        "genero_favorito": random.choice(values.GENEROS),
        "carisma": random.randint(50, 100),
        "disponibilidad": random.randint(0, 100),
        "ideologias": random.choice(values.IDEOLOGIAS),
        "ambicion": random.randint(0, 100),
        "ubicacion_geografica": random.choice(values.UBICACIONES)
    }
    musico_id_counter += 1
    return musico

def generar_banda():
    return [generar_musico() for _ in range(5)]  # Se puede tomar de un archivo csv

def calcular_aptitud(banda):
    compat_genero = sum(1 for m1 in banda for m2 in banda if m1["genero_favorito"] == m2["genero_favorito"])
    compat_ideologias = sum(1 for m1 in banda for m2 in banda if m1["ideologias"] == m2["ideologias"])
    compat_ubicacion = sum(1 for m1 in banda for m2 in banda if m1["ubicacion_geografica"] == m2["ubicacion_geografica"])
    compat_disponibilidad = sum(1 for m1 in banda for m2 in banda if abs(m1["disponibilidad"] - m2["disponibilidad"]) < 20)

    quimica = compat_genero + compat_ideologias + compat_ubicacion + compat_disponibilidad
    habilidad_tecnica = sum(m["habilidad_tecnica"] for m in banda) / len(banda)
    carisma = sum(m["carisma"] for m in banda) / len(banda)
    compromiso = (sum(m["ambicion"] for m in banda) / len(banda)) + (sum(m["disponibilidad"] for m in banda) / len(banda))

    k1, k2, k3, k4 = 0.3, 0.1, 0.3, 0.3
    aptitud = k1 * quimica + k2 * habilidad_tecnica + k3 * carisma + k4 * compromiso
    return aptitud,

def cxBanda(ind1, ind2):
    combined = ind1 + ind2
    random.shuffle(combined)
    tipos_musicos = {}

    for musico in combined:
        tipo = musico["tipo"]
        if tipo not in tipos_musicos:
            tipos_musicos[tipo] = musico

    new_musicos = list(tipos_musicos.values())

    while len(new_musicos) < 5:
        tipo_faltante = random.choice([t for t in values.TIPOS if t not in tipos_musicos])
        nuevo_musico = generar_musico()
        nuevo_musico["tipo"] = tipo_faltante
        new_musicos.append(nuevo_musico)

    ind1[:] = new_musicos[:5]
    ind2[:] = new_musicos[:5]

    return ind1, ind2

def mutar_banda(individual):
    for i in range(len(individual)):
        if random.random() < config.CONFIG.MUTATION_PROB:
            tipo = individual[i]["tipo"]
            id_original = individual[i]["id"]
            nuevo_musico = generar_musico()

            # Mantener el ID original y el tipo
            nuevo_musico["id"] = id_original
            nuevo_musico["tipo"] = tipo

            individual[i] = nuevo_musico
    return individual,

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, generar_banda)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", calcular_aptitud)

if config.CONFIG.SELECTION_TYPE == 'Tournament':
    toolbox.register("select", tools.selTournament, tournsize=config.CONFIG.TOURNAMENT_SIZE)
elif config.CONFIG.SELECTION_TYPE == 'Roulette':
    toolbox.register("select", tools.selRoulette)
elif config.CONFIG.SELECTION_TYPE == 'Rank':
    toolbox.register("select", tools.selRank)

if config.CONFIG.CROSSOVER_TYPE == 'cxBanda':
    toolbox.register("mate", cxBanda)

toolbox.register("mutate", mutar_banda)

def execute_ga_with_deap():
    population = toolbox.population(n=config.CONFIG.POPULATION_SIZE)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum(fit[0] for fit in x) / len(x))
    stats.register("max", max)

    with open("../resources/resultados.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Generación", "Fitness Promedio", "Fitness Máximo"])

        for gen in range(config.CONFIG.NUMBER_OF_GENERATIONS):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            if gen % config.CONFIG.GENERATIONAL_LEAP == 0:
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring
            record = stats.compile(population)
            hof.update(population)

            writer.writerow([gen, record["avg"], record["max"]])

            print(f"Generación {gen}: {record}")

    best_banda = hof[0]
    print("Best individual:")
    for idx, musico in enumerate(best_banda, start=1):
        print(f"Integrante {idx} (ID: {musico['id']}):")
        for key, value in musico.items():
            print(f"  {key}: {value}")
    print("Fitness value:", best_banda.fitness.values[0])

    with open("../resources/mejor_banda.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Integrante", "ID", "Tipo", "Habilidad Técnica", "Género Favorito", "Carisma",
                         "Disponibilidad", "Ideologías", "Ambición", "Ubicación Geográfica"])
        for idx, musico in enumerate(best_banda, start=1):
            writer.writerow([f"Integrante {idx}", musico["id"], musico["tipo"], musico["habilidad_tecnica"],
                             musico["genero_favorito"], musico["carisma"], musico["disponibilidad"],
                             musico["ideologias"], musico["ambicion"], musico["ubicacion_geografica"]])

if __name__ == "__main__":
    execute_ga_with_deap()
