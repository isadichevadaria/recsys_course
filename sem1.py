"""
Seminar 1 
Построение простой рекомендательной системы на основе популярности товаров

Цель: Разработать простейшие рекомендательные системы, 
на основе имеющихся данных о взаимодействии с фильмами,
проанализировать их эффективность.
"""

import numpy as np

from utils import load_data


def random_recommend(n_recommendations: int = 10, seed: int = 42) -> list[int]:
    """
    Рекомендует случайные фильмы.

    Args:
        n_recommendations: Количество рекомендаций
        seed: Seed для воспроизводимости

    Returns:
        Список ID фильмов
    """
    ratings_df, _ = load_data()
    np.random.seed(seed)
    recommendations = []    

    ### Ваш код здесь ###
    # Изменено – берём только уникальные ID фильмов
    unique_movie_ids = ratings_df['movieId'].unique()
    recommendations = np.random.choice(
        unique_movie_ids,
        size=n_recommendations,
        replace=False  # Cлучайные фильмы без повторов
    )
    recommendations = recommendations.tolist()
    ### Конец вашего кода ###

    return recommendations


def top_n_recommend(
    n_recommendations: int = 10, min_ratings: int = 10
) -> list[tuple[int, float, int]]:
    """
    Рекомендует самые популярные фильмы на основе среднего рейтинга и количества оценок.

    Args:
        n_recommendations: Количество рекомендаций
        min_ratings: Минимальное количество рейтингов для фильма

    Returns:
        Список кортежей (movieId, avg_rating, rating_count)
    """
    ratings_df, movies_df = load_data()
    top_n_recs = []

    ### Ваш код здесь ###
    # Рассчитываем ср. рейтинг и кол-во оценок для каждого movieId
    stats = (
        ratings_df
        .groupby('movieId')
        .agg(
            avg_rating=('rating', 'mean'),  # Средний рейтинг
            rating_count=('rating', 'count'),  # Кол-во оценок
        )
        .reset_index()
    )

    # Фильтруем фильмы с недостаточным количеством оценок
    stats = stats[stats['rating_count'] >= min_ratings]

    # Сортировка по популярности
    stats = stats.sort_values(
        by=['avg_rating', 'rating_count'],  # По ср. рейтингу и по кол-ву оценок
        ascending=[False, False],  # Оба по убыванию
    )

    # Выбираем топ-n фильмов
    stats = stats.head(n_recommendations)

    # Формируем список кортежей
    # (без title – его мы решили убрать на семинаре)
    top_n_recs = [
        (int(row.movieId), float(row.avg_rating), int(row.rating_count))
        for row in stats.itertuples()
    ]
    ### Конец вашего кода ###

    return top_n_recs


def evaluate_rec_systems(
    user_id: int = 610, n_recommendations: int = 10, random_state: int = 42
) -> dict:
    """
    Оценивает эффективность рекомендательной системы.
    Метрика Accuracy для двух подходов: случайные и популярные фильмы.

    Args:
        user_id: ID пользователя для оценки
        n_recommendations: Количество рекомендаций
        random_state: Seed для воспроизводимости

    Returns:
        Словарь с метрикой Accuracy для двух подходов: случайные и популярные фильмы.
    """
    ratings_df, _ = load_data()
    random_accuracy = 0.0
    popular_accuracy = 0.0

    ### Ваш код здесь ###
    # Определяем фильмы, которые пользователь уже оценил
    # set нужен, чтобы быстро проверять пересечения с рекомендациями
    user_movies = set(
        ratings_df[ratings_df['userId'] == user_id]['movieId']
    )

    # Подход 1 – Случайные рекомендации
    random_recs = random_recommend(
        n_recommendations=n_recommendations,
        seed=random_state
    )
    # Считаем, сколько фильмов из random_recs пользователь уже оценивал
    random_hits = sum(
        1 for movie in random_recs if movie in user_movies
    )
    # Считаем долю правильно "угаданных" фильмов
    random_accuracy = random_hits / n_recommendations

    # Подход 2 – Популярные фильмы
    popular_recs = top_n_recommend(
        n_recommendations=n_recommendations
    )
    # Извлекаем только ID фильмов из кортежей
    popular_movie_ids = [
        movie_id for movie_id, _, _ in popular_recs
    ]
    # Считаем, сколько популярных фильмов пользователь уже оценивал
    popular_hits = sum(
        1 for movie in popular_movie_ids if movie in user_movies
    )
    # Считаем долю "попавших" фильмов среди рекомендаций
    popular_accuracy = popular_hits / n_recommendations
    ### Конец вашего кода ###

    return {"random_accuracy": random_accuracy, "popular_accuracy": popular_accuracy}


if __name__ == "__main__":
    # 1. Случайные рекомендации
    print("\n1. СЛУЧАЙНЫЕ РЕКОМЕНДАЦИИ:")
    print("-" * 60)
    random_recs = random_recommend(n_recommendations=10)
    print(f"Рекомендованные ID фильмов: {random_recs}")

    # 2. Популярные фильмы
    # ИЗМЕНЕНО: Убираем title из вывода
    # (Решили исключить его из кортежей на семинаре)
    print("\n2. ПОПУЛЯРНЫЕ ФИЛЬМЫ (рекомендации на основе популярности):")
    print("-" * 60)
    popular_recs = top_n_recommend(n_recommendations=10)
    print(
        f"{'Rank':<5} {'ID':<6} {'Ср рейтинг':<18} {'Кол-во оценок':<15}"
    )
    print("-" * 60)
    for i, (movie_id, avg_rating, rating_count) in enumerate(popular_recs, 1):
        print(
            f"{i:<5} {movie_id:<6} {avg_rating:<18.2f} {rating_count:<15}"
        )

    # 3. Оценка системы
    print("\n3. ОЦЕНКА КАЧЕСТВА СИСТЕМЫ:")
    print("-" * 60)
    metrics = evaluate_rec_systems()
    print(f"Accuracy (случайные рекомендации): {metrics['random_accuracy']:.4f}")
    print(f"Accuracy (популярные фильмы): {metrics['popular_accuracy']:.4f}")
