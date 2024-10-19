import config
import values
import csv
import random
from deap import base, creator, tools

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
    banda = []
    tipos_usados = set()

    for tipo in values.TIPOS:
        musico = generar_musico()
        while musico["tipo"] != tipo:
            musico = generar_musico()
        banda.append(musico)
        tipos_usados.add(tipo)

    while len(banda) < 5:
        nuevo_musico = generar_musico()
        if nuevo_musico["tipo"] not in tipos_usados:
            banda.append(nuevo_musico)
            tipos_usados.add(nuevo_musico["tipo"])

    return banda

def calcular_aptitud(banda):
    compat_genero = sum(1 for m1 in banda for m2 in banda if m1["genero_favorito"] == m2["genero_favorito"])
    compat_ideologias = sum(1 for m1 in banda for m2 in banda if m1["ideologias"] == m2["ideologias"])
    compat_ubicacion = sum(1 for m1 in banda for m2 in banda if m1["ubicacion_geografica"] == m2["ubicacion_geografica"])
    compat_disponibilidad = sum(1 for m1 in banda for m2 in banda if abs(m1["disponibilidad"] - m2["disponibilidad"]) < 15)

    quimica = compat_genero + compat_ideologias + 5 * compat_ubicacion + compat_disponibilidad
    habilidad_tecnica = sum(m["habilidad_tecnica"] for m in banda) / len(banda)
    carisma = sum(m["carisma"] for m in banda) / len(banda)
    compromiso = (sum(m["ambicion"] for m in banda) / len(banda)) + (sum(m["disponibilidad"] for m in banda) / len(banda))

    k1, k2, k3, k4 = 0.3, 0.1, 0.3, 0.3
    aptitud = k1 * quimica + k2 * habilidad_tecnica + k3 * carisma + k4 * compromiso
    return aptitud,

def cxBanda(ind1, ind2):
    punto_corte = random.randint(1, len(ind1) - 1)
    hijo1 = ind1[:punto_corte] + ind2[punto_corte:]
    hijo2 = ind2[:punto_corte] + ind1[punto_corte:]

    return hijo1, hijo2

def mutar_banda(individual, musicos_disponibles):
    if random.random() < config.CONFIG.MUTATION_PROB:
        indice_a_mutar = random.randint(0, len(individual) - 1)
        musico_a_mutar = individual[indice_a_mutar]
        musicos_mismo_tipo = [m for m in musicos_disponibles if m["tipo"] == musico_a_mutar["tipo"]]

        if musicos_mismo_tipo:
            nuevo_musico = random.choice(musicos_mismo_tipo)
            individual[indice_a_mutar] = nuevo_musico
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
    toolbox.register("select", tools.selBest)

if config.CONFIG.CROSSOVER_TYPE == 'CxSimple':
    toolbox.register("mate", cxBanda)

toolbox.register("mutate", mutar_banda)

def execute_ga_with_deap():

    population = toolbox.population(n=config.CONFIG.POPULATION_SIZE)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)  # Cambia 'tools' por 'toolbox'


    musicos_disponibles = [musico for banda in population for musico in banda]

    print("Población original:")
    for idx, banda in enumerate(population, start=1):
        print(f"Banda {idx}:")
        for musico in banda:
            print(f"  ID: {musico['id']}, Tipo: {musico['tipo']}, Habilidad Técnica: {musico['habilidad_tecnica']}, "
                  f"Género Favorito: {musico['genero_favorito']}, Carisma: {musico['carisma']}, "
                  f"Disponibilidad: {musico['disponibilidad']}, Ideologías: {musico['ideologias']}, "
                  f"Ambición: {musico['ambicion']}, Ubicación Geográfica: {musico['ubicacion_geografica']}")

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum(fit[0] for fit in x) / len(x))
    stats.register("max", max)

    with open("../resources/resultados.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Generación", "Fitness Promedio", "Fitness Máximo"])

        for gen in range(config.CONFIG.NUMBER_OF_GENERATIONS):
            for ind in population:
                ind.fitness.values = toolbox.evaluate(ind)

            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            if gen % config.CONFIG.GENERATIONAL_LEAP == 0:
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                toolbox.mutate(mutant, musicos_disponibles)
                del mutant.fitness.values
                if not mutant.fitness.valid:
                    mutant.fitness.values = toolbox.evaluate(mutant)

            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring
            record = stats.compile(population)
            hof.update(population)

            # Escribir los resultados en el archivo dentro del contexto 'with'
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