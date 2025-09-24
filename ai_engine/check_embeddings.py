"""
check_embeddings.py
Carga el archivo embeddings_db.pkl y muestra
qué jugadores tiene y cuántos embeddings por cada uno.
También imprime un ejemplo de vector.
"""

import pickle

with open("ai_engine/embeddings_db.pkl", "rb") as f:
    embeddings_db = pickle.load(f)

print("Jugadores cargados en la base de embeddings:\n")
for player, vectors in embeddings_db.items():
    print(f"- {player}: {len(vectors)} embeddings")

    # Mostrar un ejemplo recortado
    if vectors:  # si tiene embeddings
        print(f"  Ejemplo (primeras 5 dimensiones): {vectors[0][:5]}\n")

print("Total jugadores:", len(embeddings_db))

