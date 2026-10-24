from evaluation import rank_scores


class Recommender:
    def recommend(self, users, seen, k=10):
        return rank_scores(self.scores(users), seen[users], k)
