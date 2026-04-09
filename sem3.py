"""
Семинар 3. Контентная фильтрация
Цель: Разработать методы контентной фильтрации по пользователям и по фильмам.
В качестве контента используем описание жанров для каждого фильма из movies.csv.
Для векторизации жанров используем CountVectorizer с разделителем "|".
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from utils import build_user_item_matrix, id_to_movie, load_data, print_user_rated_items


class ContentRecommender:
    """
    Класс для построения рекомендаций на основе контента - описания жанров.
    Матрица эмбеддингов размером (max_movie_id+1, n_genres), где строки
    соответствуют movieId, а столбцы — one-hot кодированию жанров.
    Матрица строится при инициализации экземпляра класса.
    """

    def __init__(self):
        self.embeddings = None
        self.ui_matrix = build_user_item_matrix()
        self._build_embeddings()

    def _build_embeddings(self):
        _, movies_df = load_data()
        self.movies_df = movies_df.copy()
        self.movies_df["genres"] = self.movies_df["genres"].fillna("")
        vectorizer = CountVectorizer(tokenizer=lambda s: s.split("|"),
                                     lowercase=False,
                                     token_pattern=None)  # Убираем предупреждение
        ###########################################################################
        # Векторизуем жанры фильмов
        genre_matrix_vec = vectorizer.fit_transform(self.movies_df["genres"]).toarray()

        # Создаём матрицу эмбеддингов
        max_movie_id = self.movies_df["movieId"].max()
        self.embeddings = np.zeros((max_movie_id + 1, genre_matrix_vec.shape[1]))

        # Заполняем строки эмбеддингов по movieId
        for idx, movie_id in enumerate(self.movies_df["movieId"]):
            self.embeddings[movie_id] = genre_matrix_vec[idx]
        ###########################################################################
        

    def predict_rating(self, user_id: int, item_id: int, k: int = 5) -> float:
        """
        Предсказывает рейтинг user_id для item_id на основе контентной фильтрации.

        Алгоритм:
        1) Берём вектор целевого фильма: target_vec.
        2) Находим все фильмы, оцененные пользователем.
        3) Считаем косинусное сходство target_vec с векторами оцененных фильмов.
        4) Отбираем топ-k похожих оцененных фильмов (k-параметр).
        5) Предсказываем рейтинг как взвешенное среднее оценок по сходствам.
        6) Если не удаётся предсказать (нет оценок или нулевые векторы), возвращаем 0.0.
        7) Клипируем результат в [0.0, 5.0].

        Args:
            user_id: индекс пользователя
            item_id: индекс фильма
            k: сколько наиболее похожих оцененных фильмов использовать

        Returns:
            float: предсказанный рейтинг
        """
        # Проверяем входные параметры
        n_users, n_items = self.ui_matrix.shape
        if not (0 <= user_id < n_users):
            raise IndexError("user_id out of bounds")
        if not (0 <= item_id < n_items):
            raise IndexError("item_id out of bounds")
        if k <= 0:
            raise ValueError("k must be positive")

        # Вектор целевого фильма
        target_vec = self.embeddings[item_id]

        # Норма целевого вектора
        target_norm = np.linalg.norm(target_vec)
        if target_norm == 0:
            return 0.0

        # Оценки пользователя и индексы оценённых фильмов
        user_ratings = self.ui_matrix[user_id]
        rated_items_idx = np.where(user_ratings > 0)[0]
        if rated_items_idx.size == 0:
            return 0.0

        # Векторы оценённых фильмов
        rated_items_vecs = self.embeddings[rated_items_idx]

        # Скалярные произведения
        dot = rated_items_vecs @ target_vec

        # Нормы оценённых фильмов
        rated_norms = np.linalg.norm(rated_items_vecs, axis=1)
        denom = target_norm * rated_norms

        # Косинусное сходство
        similarities = np.divide(
            dot,
            denom,
            out=np.zeros_like(dot),
            where=denom != 0
        )

        # Берём top-k наиболее похожих фильмов
        sorted_idx = np.argsort(similarities)[::-1][:k]
        topk_similarities = similarities[sorted_idx]
        topk_ratings = user_ratings[rated_items_idx][sorted_idx]

        # Оставляем только положительные сходства
        positive_mask = topk_similarities > 0
        if not np.any(positive_mask):
            return 0.0
        topk_similarities = topk_similarities[positive_mask]
        topk_ratings = topk_ratings[positive_mask]

        # Взвешенное среднее оценок
        pred = np.dot(topk_similarities, topk_ratings) / topk_similarities.sum()

        # Клипируем результат
        return float(np.clip(pred, 0.0, 5.0))

    def predict_items_for_user(
        self, user_id: int, k: int = 5, n_recommendations: int = 5
    ) -> list:
        """
        Рекомендует фильмы пользователю user_id на основе контента фильма.

        Алгоритм:
        1) Берем все фильмы, которые оценил пользователь.
        3) Строим профиль пользователя как взвешенное среднее жанров оцененных фильмов.
        4) Для всех фильмов, которые пользователь не оценил,
        считаем сходство с профилем.
        5) Сортируем по убыванию сходства и возвращаем top-n.
        """
        # Проверяем входные параметры
        n_users, _ = self.ui_matrix.shape
        if not (0 <= user_id < n_users):
            raise IndexError("user_id out of bounds")
        if k <= 0:
            raise ValueError("k must be positive")
        if n_recommendations <= 0:
            raise ValueError("n_recommendations must be > 0")

        # Оценки пользователя и индексы оценённых фильмов
        user_ratings = self.ui_matrix[user_id]
        rated_items_idx = np.where(user_ratings > 0)[0]
        if rated_items_idx.size == 0:
            return []

        # Векторы оценённых фильмов
        rated_items_vecs = self.embeddings[rated_items_idx]

        # Веса оценок пользователя
        weights = user_ratings[rated_items_idx][:, None]
        if weights.sum() == 0:
            return []

        # Профиль пользователя
        user_profile = (rated_items_vecs * weights).sum(axis=0) / weights.sum()

        # Норма профиля пользователя
        profile_norm = np.linalg.norm(user_profile)
        if profile_norm == 0:
            return []

        # Берём только фильмы, которые пользователь не оценивал
        unseen_items = np.where(user_ratings == 0)[0]
        if unseen_items.size == 0:
            return []
        unseen_items_vecs = self.embeddings[unseen_items]

        # Косинусное сходство с профилем пользователя
        dot = unseen_items_vecs @ user_profile
        norms = np.linalg.norm(unseen_items_vecs, axis=1) * profile_norm
        similarities = np.divide(
            dot,
            norms,
            out=np.zeros_like(dot),
            where=norms != 0
        )

        # Берём top-n рекомендаций
        sorted_idx = np.argsort(similarities)[::-1][:n_recommendations]
        topn_recs = unseen_items[sorted_idx]

        # Возвращаем top-n списком
        return [int(i) for i in topn_recs]


# Пример использования для дебага:
if __name__ == "__main__":
    user_id = 10
    item_id = 2
    k = 5
    content_recommender = ContentRecommender()
    print_user_rated_items(user_id, content_recommender.ui_matrix)

    pred_rating = content_recommender.predict_rating(user_id, item_id, k)
    print(f"Predicted rating for user {user_id} and item {item_id}: {pred_rating:.2f}")

    recommendations = content_recommender.predict_items_for_user(
        user_id, k=5, n_recommendations=10
    )
    for rec in recommendations:
        print(f"Recommended movie ID: {rec}, Title: {id_to_movie(rec)}")
