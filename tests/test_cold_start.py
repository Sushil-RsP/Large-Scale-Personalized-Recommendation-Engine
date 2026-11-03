"""Unit tests for Cold-Start fallback strategies."""
import pytest
from src.cold_start.fallback import ColdStartEngine

def test_popular_diverse_fallback():
    engine = ColdStartEngine()
    recs = engine.get_popular_diverse_fallback(top_k=20)

    assert len(recs) == 20
    # Check no duplicate movies
    movie_ids = [r["movieId"] for r in recs]
    assert len(movie_ids) == len(set(movie_ids))

    # Check diversity: should span multiple distinct genres
    all_genres = set()
    for r in recs:
        for g in r["genres"].split("|"):
            all_genres.add(g)
    assert len(all_genres) >= 4, f"Diversity check: expected at least 4 genres, got {len(all_genres)}"

def test_content_based_onboarding_fallback():
    engine = ColdStartEngine()
    genres_selected = ["Action", "Sci-Fi"]
    recs = engine.get_content_based_genre_fallback(genres_selected, top_k=15)

    assert len(recs) == 15
    for r in recs:
        # At least one of the selected genres should be present in the item
        item_genres = r["genres"].split("|")
        matches = [g for g in genres_selected if g in item_genres]
        assert len(matches) > 0, f"Item {r['title']} with genres {r['genres']} has no match with {genres_selected}"

