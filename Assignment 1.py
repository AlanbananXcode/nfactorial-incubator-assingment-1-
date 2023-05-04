import random
from collections import defaultdict
from typing import List, Dict

import matplotlib.pyplot as plt
import torch


def read_names_file(file_path: str) -> List[str]:
    """
    Чтение файла с именами и сохранение в список
    """
    with open(file_path, "r") as f:
        names = [name.strip().lower() for name in f.readlines()]
    return names


def build_bigram_model(names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Построение модели биграмм для списка имен
    """
    bigram_model = defaultdict(lambda: defaultdict(int))
    for name in names:
        name = "^" + name + "$"  # добавляем токены начала и конца
        for i in range(len(name) - 1):
            bigram_model[name[i]][name[i+1]] += 1
    for char in bigram_model:
        total_count = sum(bigram_model[char].values())
        for next_char in bigram_model[char]:
            bigram_model[char][next_char] /= total_count
    return bigram_model


def generate_name(bigram_model: Dict[str, Dict[str, float]]) -> str:
    """
    Генерация имени с помощью модели биграмм
    """
    name = "^"
    char = random.choice(list(bigram_model.keys()))  # выбираем случайную первую букву
    while char != "$":
        name += char
        next_char_probs = torch.Tensor(list(bigram_model[char].values()))
        next_char_index = torch.multinomial(next_char_probs, 1)[0]  # выбираем следующую букву случайным образом
        char = list(bigram_model[char].keys())[next_char_index]
    return name[1:-1].capitalize()  # удаляем токены начала и конца и делаем первую букву заглавной


def visualize_bigram_model(bigram_model: Dict[str, Dict[str, float]]):
    """
    Визуализация таблицы вероятностей биграмм
    """
    chars = sorted(bigram_model.keys())
    matrix = [[bigram_model[char][next_char] for next_char in chars] for char in chars]
    plt.imshow(matrix, cmap="hot", interpolation="nearest")
    plt.xticks(range(len(chars)), chars)
    plt.yticks(range(len(chars)), chars)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    names = read_names_file("names.txt")
    bigram_model = build_bigram_model(names)
    visualize_bigram_model(bigram_model)
    for i in range(10):
        print(generate_name(bigram_model))
