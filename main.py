from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import random
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)


# Leer los datos desde el archivo CSV
def leer_datos():
    try:
        libros = pd.read_csv('books_clean.csv', encoding='latin1', delimiter=',', on_bad_lines='skip', low_memory=False)
        libros['price'] = pd.to_numeric(libros['price'],errors='coerce')
        libros['language'] = libros['language'].fillna('Spanish')

        return libros
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return pd.DataFrame()


libros = leer_datos()

def generar_poblacion_inicial(tamaño_poblacion, tamaño_individuo, libros):
    return [
        [random.randint(0, len(libros) - 1) for _ in range(tamaño_individuo)]
        for _ in range(tamaño_poblacion)
    ]

    return poblacion

# Evaluar la aptitud de un individuo
def evaluar_aptitud(individuo, libros, generos_favoritos, autores_favoritos, puntuacion_minima, formatos_preferidos, lenguajes_preferidos):
    # Inicializar puntuaciones
    puntuacion_generos = 0
    puntuacion_autores = 0
    puntuacion_formatos = 0
    puntuacion_lenguajes = 0
    puntuacion_minima_total = 0
    libros_que_cumplen_puntuacion_minima = 0

    # Evaluar cada libro en el individuo
    for idx in individuo:
        if idx < len(libros):
            libro = libros.iloc[idx]

            # Evaluar géneros
            generos_libro = libro["genres"]
            if pd.notna(generos_libro) and generos_libro.strip() != '':
                generos_libro_set = set(generos_libro.split(', '))
                generos_relevantes = generos_libro_set & set(generos_favoritos)
                puntuacion_generos += len(generos_relevantes) * 0.9

            # Evaluar autores
            if libro["author"] in autores_favoritos:
                puntuacion_autores += 0.8

            # Evaluar puntuación mínima
            if libro["rating"] >= puntuacion_minima:
                libros_que_cumplen_puntuacion_minima += 1
                puntuacion_minima_total += libro["rating"]

            # Evaluar formatos
            if libro["bookFormat"] in formatos_preferidos:
                puntuacion_formatos += 0.5

            # Evaluar lenguajes
            if libro["language"] in lenguajes_preferidos:
                puntuacion_lenguajes += 0.8

    # Normalizar puntuaciones
    puntuacion_formatos = puntuacion_formatos / max(len(individuo), 1) * 0.6
    puntuacion_minima_total = (puntuacion_minima_total / max(libros_que_cumplen_puntuacion_minima, 1)) * 0.7
    puntuacion_lenguajes = puntuacion_lenguajes / max(len(individuo), 1) * 0.8

    # Pesos para cada factor
    pesos = [0.9, 0.8, 0.7, 0.6, 0.8]
    factores = [
        puntuacion_generos,
        puntuacion_autores,
        puntuacion_minima_total,
        puntuacion_formatos,
        puntuacion_lenguajes
    ]

    # Calcular la puntuación de aptitud
    puntuacion_aptitud = sum(p * f for p, f in zip(pesos, factores))

    return puntuacion_aptitud


# Seleccionar individuos mediante torneo
def seleccionar_torneo(poblacion, aptitudes, tamaño_torneo=5):
    seleccionados = []
    for _ in range(len(poblacion)):
        indices_torneo = random.sample(range(len(poblacion)), tamaño_torneo)
        ganador_indice = max(indices_torneo, key=lambda idx: aptitudes[idx])
        seleccionados.append(poblacion[ganador_indice])
    return seleccionados

# Cruzar dos individuos
def cruce_un_punto(padre1, padre2):
    punto_cruce = random.randint(1, len(padre1) - 1)
    hijo1 = padre1[:punto_cruce] + padre2[punto_cruce:]
    hijo2 = padre2[:punto_cruce] + padre1[punto_cruce:]
    return hijo1, hijo2


# Mutar un individuo
def mutacion(individuo, num_libros, tasa_mutacion=0.1):
    for i in range(len(individuo)):
        if random.random() < tasa_mutacion:
            individuo[i] = random.randint(0, num_libros - 1)
    return individuo


# Poda de la población
def poda(poblacion, tamaño_maximo, libros, generos_favoritos, autores_favoritos, puntuacion_minima, formatos_preferidos, lenguajes_preferidos):
    aptitudes = [evaluar_aptitud(ind, libros, generos_favoritos, autores_favoritos, puntuacion_minima, formatos_preferidos, lenguajes_preferidos) for ind in poblacion]

    aptitudes_individuos = list(zip(aptitudes, poblacion))

    aptitudes_individuos.sort(reverse=True, key=lambda x: x[0])

    num_a_podar = int(len(poblacion) * 0.20)
    num_nuevos = int(len(poblacion) * 0.20)

    mejor_poblacion = [ind for _, ind in aptitudes_individuos[:-num_a_podar]]

    nuevos_individuos = generar_poblacion_inicial(num_nuevos, tamaño_individuo=len(poblacion[0]), libros=libros)

    nueva_poblacion = mejor_poblacion + nuevos_individuos

    nueva_poblacion = list(map(list, set(map(tuple, nueva_poblacion))))
    if len(nueva_poblacion) > tamaño_maximo:
        nueva_poblacion = nueva_poblacion[:tamaño_maximo]

    return nueva_poblacion

def crear_parejas(poblacion):
    # Asegurarse de que el número de individuos en la población sea par
    if len(poblacion) % 2 != 0:
        poblacion = poblacion[:-1]  # Eliminar el último individuo si el número es impar

    parejas = [(poblacion[i], poblacion[i + 1]) for i in range(0, len(poblacion), 2)]
    return parejas


# Algoritmo genético
def algoritmo_genetico(libros, generos_favoritos, autores_favoritos, puntuacion_minima, formatos_preferidos, lenguajes_preferidos, tamaño_poblacion_inicial=1000, tamaño_poblacion_maximo=300, tamaño_individuo=3, num_generaciones=200):
    mejor_aptitud_generacion, peor_aptitud_generacion, aptitud_promedio_generacion = [], [], []
    generos_favoritos = set(generos_favoritos)
    autores_favoritos = set(autores_favoritos)

    # Inicializar población
    poblacion_inicial = generar_poblacion_inicial(tamaño_poblacion_inicial, tamaño_individuo, libros)
    poblacion = poblacion_inicial

    for generacion in range(num_generaciones):
        # Evaluar aptitudes
        aptitudes = [evaluar_aptitud(ind, libros, generos_favoritos, autores_favoritos, puntuacion_minima, formatos_preferidos, lenguajes_preferidos) for ind in poblacion]

        # Selección por torneo
        poblacion_seleccionada = seleccionar_torneo(poblacion, aptitudes)

        # Crear parejas para la reproducción
        parejas = crear_parejas(poblacion_seleccionada)

        # Cruza y muta
        nueva_poblacion = []
        while len(nueva_poblacion) < tamaño_poblacion_inicial:
            for padre1, padre2 in parejas:
                hijo1, hijo2 = cruce_un_punto(padre1, padre2)
                hijo1 = mutacion(hijo1, len(libros))
                hijo2 = mutacion(hijo2, len(libros))
                nueva_poblacion.extend([hijo1, hijo2])
                if len(nueva_poblacion) >= tamaño_poblacion_inicial:
                    break

        aptitudes = [evaluar_aptitud(ind, libros, generos_favoritos, autores_favoritos, puntuacion_minima, formatos_preferidos, lenguajes_preferidos) for ind in poblacion]
        mejor_aptitud = max(aptitudes)
        peor_aptitud = min(aptitudes)
        aptitud_promedio = sum(aptitudes) / len(aptitudes)

        # Poda de la población
        poblacion = poda(nueva_poblacion, tamaño_poblacion_maximo, libros, generos_favoritos, autores_favoritos, puntuacion_minima, formatos_preferidos, lenguajes_preferidos)

        mejor_aptitud_generacion.append(mejor_aptitud)
        peor_aptitud_generacion.append(peor_aptitud)
        aptitud_promedio_generacion.append(aptitud_promedio)

        print(f"Generación {generacion + 1}: Mejor aptitud = {mejor_aptitud}, Peor aptitud = {peor_aptitud}, Aptitud promedio = {aptitud_promedio}")

    # Obtener el mejor individuo de la última generación
    mejor_individuo = max(poblacion, key=lambda ind: evaluar_aptitud(ind, libros, generos_favoritos, autores_favoritos, puntuacion_minima, formatos_preferidos, lenguajes_preferidos))
    recomendaciones = libros.iloc[mejor_individuo]

    return recomendaciones, aptitud_promedio_generacion, mejor_aptitud_generacion, peor_aptitud_generacion


def crear_graficas(aptitud_promedio, aptitud_mejor, aptitud_peor):
    fig, ax = plt.subplots()
    generaciones = range(len(aptitud_promedio))
    ax.plot(generaciones, aptitud_promedio, label='Aptitud Promedio')
    ax.plot(generaciones, aptitud_mejor, label='Mejor Aptitud')
    ax.plot(generaciones, aptitud_peor, label='Peor Aptitud')

    ax.set_xlabel('Generaciones')
    ax.set_ylabel('Aptitud')
    ax.set_title('Evolución de la Aptitud')
    ax.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64


@app.route('/recomendar', methods=['POST'])
def recomendar():
    data = request.json
    print(data)

    generos_favoritos = data.get('generos_favoritos', [])
    autores_favoritos = data.get('autores_favoritos', [])
    puntuacion_minima = data.get('puntuacion_minima', 0)
    formatos_preferidos = data.get('formatos_preferidos', [])
    lenguaje_preferido = data.get('lenguajes_preferidos', [])

    print(puntuacion_minima)

    recomendaciones, aptitud_promedio, aptitud_mejor, aptitud_peor = algoritmo_genetico(libros, generos_favoritos, autores_favoritos, puntuacion_minima, formatos_preferidos, lenguaje_preferido)

    recomendaciones = recomendaciones.fillna('null')
    resultado = recomendaciones[['title', 'author', 'rating', 'genres', 'bookFormat', 'language', 'description']].to_dict(orient='records')

    grafico_base64 = crear_graficas(aptitud_promedio, aptitud_mejor, aptitud_peor)

    return jsonify({
        'recomendaciones': resultado,
        'grafico': grafico_base64
    })

@app.route('/buscar', methods=['GET'])
def buscar():
    query = request.args.get('query', '')
    generos_preferidos = request.args.getlist('generos_preferidos')
    formatos_preferidos = request.args.getlist('formatos_preferidos')

    if not query and not generos_preferidos and not formatos_preferidos:
        return jsonify([])

    libros = leer_datos()

    if query:
        libros = libros[libros['title'].str.contains(query, case=False, na=False)]

    if generos_preferidos:
        libros['generos'] = libros['genres'].apply(lambda x: set(x.split(', ')) if pd.notna(x) else set())
        libros = libros[libros['generos'].apply(lambda x: x.intersection(generos_preferidos))]

    if formatos_preferidos:
        libros = libros[libros['bookFormat'].isin(formatos_preferidos)]

    resultado = libros[['title', 'author', 'genres']].to_dict(orient='records')
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True)
