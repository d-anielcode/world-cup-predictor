from touchline.storage.db import Database
from touchline.model.ratings import Ratings


def test_ratings_roundtrip(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init_schema()
    r = Ratings(attack={"USA": 0.3, "Wales": -0.1},
                defense={"USA": 0.2, "Wales": -0.1},
                home_adv=0.27, rho=-0.06)
    db.save_ratings(r)
    loaded = db.load_ratings()
    assert loaded.attack["USA"] == 0.3
    assert loaded.defense["Wales"] == -0.1
    assert abs(loaded.home_adv - 0.27) < 1e-9
    assert abs(loaded.rho - (-0.06)) < 1e-9


def test_save_ratings_keeps_defense_only_teams(tmp_path):
    # A team present only in the defense dict must not be silently dropped.
    db = Database(tmp_path / "t.db")
    db.init_schema()
    db.save_ratings(Ratings(attack={"USA": 0.3}, defense={"USA": 0.2, "Extra": -0.3},
                            home_adv=0.2, rho=-0.05))
    loaded = db.load_ratings()
    assert loaded.defense["Extra"] == -0.3
    assert loaded.attack["Extra"] == 0.0   # defaulted, not lost


def test_save_ratings_replaces_previous(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init_schema()
    db.save_ratings(Ratings(attack={"USA": 0.1}, defense={"USA": 0.1},
                            home_adv=0.2, rho=-0.05))
    db.save_ratings(Ratings(attack={"USA": 0.9}, defense={"USA": 0.9},
                            home_adv=0.3, rho=-0.04))
    loaded = db.load_ratings()
    assert loaded.attack["USA"] == 0.9
    assert abs(loaded.home_adv - 0.3) < 1e-9
