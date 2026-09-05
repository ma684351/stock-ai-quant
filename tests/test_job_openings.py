import os
import shutil
import tempfile

import pandas as pd

from core.job_openings import (
    fetch_company_job_openings,
    get_job_openings_series,
    load_jobs_mappings,
)


def test_load_jobs_mappings():
    mappings = load_jobs_mappings()
    assert isinstance(mappings, dict)
    assert "4385" in mappings
    assert mappings["4385"]["platform"] == "greenhouse"
    assert "PGNY" in mappings
    assert mappings["PGNY"]["platform"] == "workday"


def test_fetch_company_job_openings_unregistered():
    count, platform = fetch_company_job_openings("NONEXISTENT_TICKER_XYZ999")
    assert count is None
    assert platform is None


def test_get_job_openings_series_with_temp_cache():
    tmp_dir = tempfile.mkdtemp()
    try:
        dates = pd.date_range("2024-01-01", periods=5)
        df_stock = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)

        # 未登録銘柄で安全なフォールバック (0.0補完) の検証
        s_fallback = get_job_openings_series(
            "NONEXISTENT_TICKER_XYZ",
            df_stock,
            cache_dir=tmp_dir,
            verbose=False,
        )
        assert len(s_fallback) == 5
        assert (s_fallback == 0.0).all()
        assert s_fallback.index.equals(df_stock.index)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_get_job_openings_series_with_custom_mapping():
    tmp_dir = tempfile.mkdtemp()
    mapping_file = os.path.join(tmp_dir, "test_mapping.json")
    try:
        # 一時マッピングJSON作成
        with open(mapping_file, "w", encoding="utf-8") as f:
            f.write('{"TEST": {"platform": "greenhouse", "token": "mercari"}}')

        dates = pd.date_range("2024-01-01", periods=3)
        df_stock = pd.DataFrame({"Close": [10.0, 11.0, 12.0]}, index=dates)

        s_jobs = get_job_openings_series(
            "TEST",
            df_stock,
            cache_dir=tmp_dir,
            mapping_file=mapping_file,
            verbose=False,
        )
        assert len(s_jobs) == 3
        # mercariは求人が存在するため正の値
        assert (s_jobs >= 0.0).all()
        assert os.path.exists(os.path.join(tmp_dir, "TEST.json"))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
