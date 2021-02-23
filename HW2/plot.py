import matplotlib.pyplot as plt


def show_result(steps, means, stds):
    plt.figure(figsize=(15, 7))
    plt.title('Средняя награда в зависимости от шага обучения')
    plt.xlabel('Шаг замера')
    plt.ylabel('Средняя награда')
    plt.plot(steps, means, lw=2, color='blue')
    plt.fill_between(steps, means + stds, means - stds, facecolor='blue', alpha=0.3)
    plt.savefig('Results.png')
