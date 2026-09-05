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
