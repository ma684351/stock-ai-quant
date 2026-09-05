import pandas as pd

from core import data_loader
from core.data_loader import clean_ticker_name, is_japanese_ticker, normalize_ticker


def test_is_japanese_ticker():
    # 日本株（4桁数字、サフィックス付き）
    assert is_japanese_ticker("7203") is True
    assert is_japanese_ticker("7203.T") is True
    assert is_japanese_ticker("6758.JP") is True
    assert is_japanese_ticker("9984.TYO") is True
    assert is_japanese_ticker("285A") is True  # 東証の英字入りコード

    # 米国株・為替・指数
    assert is_japanese_ticker("AAPL") is False
    assert is_japanese_ticker("NVDA") is False
    assert is_japanese_ticker("MSFT") is False
    assert is_japanese_ticker("^GSPC") is False
    assert is_japanese_ticker("JPY=X") is False


def test_normalize_ticker():
    # 4桁日本株コードには自動で .T が付与される
    assert normalize_ticker("7203") == "7203.T"
    assert normalize_ticker("6758") == "6758.T"
    assert normalize_ticker("7203.t") == "7203.T"

    # 既にサフィックスがあるものはそのまま
    assert normalize_ticker("7203.T") == "7203.T"
    assert normalize_ticker("6758.JP") == "6758.JP"

    # 米国株などは大文字に正規化される
    assert normalize_ticker("aapl") == "AAPL"
    assert normalize_ticker("  nvda  ") == "NVDA"
    assert normalize_ticker("BRK-B") == "BRK-B"


def test_clean_ticker_name():
    # 特殊文字（. や -）がアンダースコアに変換される
    assert clean_ticker_name("7203.T") == "7203_T"
    assert clean_ticker_name("BRK-B") == "BRK_B"
    assert clean_ticker_name("aapl") == "AAPL"


def test_fetch_macro_data_caching(monkeypatch):
    call_count = 0

    def mock_fetch_market_data(ticker, period="2y"):
        nonlocal call_count
        call_count += 1
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        return pd.DataFrame({"Close": [100.0] * 10}, index=dates)

    monkeypatch.setattr(data_loader, "fetch_market_data", mock_fetch_market_data)
    data_loader._MACRO_CACHE.clear()

    # 1回目の取得（4回ダウンロードされる）
    res1 = data_loader.fetch_macro_data(period="2y")
    assert call_count == 4
    assert len(res1) == 4

    # 2回目の取得（キャッシュが効いてダウンロードは増えない）
    res2 = data_loader.fetch_macro_data(period="2y")
    assert call_count == 4
    assert res1[0] is res2[0]
