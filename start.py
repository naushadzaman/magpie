from magpie import MagpieModel

magpie = MagpieModel()
magpie.init_word_vectors('data/hep-categories', vec_dim=100)

labels = ['Gravitation and Cosmology', 'Experiment-HEP', 'Theory-HEP']
magpie.train('data/hep-categories', labels, test_ratio=0.2, nb_epochs=30)

print magpie.predict_from_file('data/hep-categories/1002413.txt')
print magpie.predict_from_text('Stephen Hawking studies black holes')