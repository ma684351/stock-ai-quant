import shutil
import tempfile

import pandas as pd

from core.search_volume import (
    fetch_wikimedia_pageviews,
    get_search_volume_series,
    resolve_wikipedia_article,
)


def test_resolve_wikipedia_article():
    # 既知の米国株
    lang_aapl, title_aapl = resolve_wikipedia_article("AAPL")
    assert lang_aapl == "en"
    assert title_aapl == "Apple_Inc."

    lang_glw, title_glw = resolve_wikipedia_article("GLW")
    assert lang_glw == "en"
    assert title_glw == "Corning_Inc."

    # 既知の日本株
    lang_7203, title_7203 = resolve_wikipedia_article("7203")
    assert lang_7203 == "ja"
    assert title_7203 == "トヨタ自動車"

    lang_6758, title_6758 = resolve_wikipedia_article("6758.T")
    assert lang_6758 == "ja"
    assert title_6758 == "ソニーグループ"


def test_fetch_wikimedia_pageviews_real():
    # 実際に短期間 (3日間) のWikimedia APIを叩いてレスポンス形式をテスト
    res = fetch_wikimedia_pageviews(
        lang="en",
        article="Apple_Inc.",
        start_date="20240101",
        end_date="20240103",
    )
    if res is not None:
        assert "items" in res
        assert len(res["items"]) == 3
        assert "views" in res["items"][0]


def test_get_search_volume_series_with_temp_cache():
    # 一時ディレクトリを使ったキャッシュ動作とフォールバックのテスト
    tmp_dir = tempfile.mkdtemp()
    try:
        dates = pd.date_range("2024-01-01", periods=5)
        df_stock = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)

        # 存在しない架空のティッカーでフォールバック（ゼロ補完）の検証
        s_fallback = get_search_volume_series(
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
