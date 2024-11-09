import string

# Liste des chiffres en toutes lettres avec la première lettre en majuscule
#numbers_in_words = ["Zero", "Un", "Deux", "Trois", "Quatre", "Cinq", "Six", "Sept", "Huit", "Neuf"]

# Liste des lettres de l'alphabet en majuscules sauf 'J' et 'Z'
letters = [ch.upper() for ch in string.ascii_lowercase if ch not in ['j', 'z']]

# Combinaison des deux listes
values = letters

# Création du dictionnaire avec les clés de 0 à 33
dictionary = {i: values[i % len(values)] for i in range(24)}

print(dictionary)
