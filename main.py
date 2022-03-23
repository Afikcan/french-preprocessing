import numpy as np
import pandas as pd

from data_preprocess import return_preprocessed_text

lst = ['C\'est un test pour lemmatizer',
       'plusieurs phrases pour un nettoyage',
       'eh voilà la troisième !',
       'il y\'a de cela 2 semaines']

lst_2 = [
    "Nous avons donc demandé à Maya Gupta, chercheuse chez Google dans le domaine de l'apprentissage automatique, "
    "de nous expliquer tout cela.",
    "Pour nombre d'entre nous, l'apprentissage automatique semble assez futuriste. Pourtant, depuis quelque temps, "
    "on le retrouve de plus en plus dans notre quotidien, que ce soit sous la forme d'un ordinateur Google livrant "
    "une partie de go palpitante ou de la création de réponses automatiques dans Inbox by Gmail.",
    ]

index = 1

french_text = pd.DataFrame(lst_2, columns=['text'])

preprocessed_text = return_preprocessed_text(french_text['text'], 'fr')

print(french_text['text'][index])
print(preprocessed_text[index])
