import random

categories = open('data/categories.txt', 'r')
categories = categories.readlines()
random.shuffle(categories)
categories = categories[:20]
for category, i in zip(categories, range(len(categories))):
    categories[i] = category[:-1]

print(categories)
