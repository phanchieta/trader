from trader.core.dotenv import load_dotenv


def test_parseia_pares_ignora_comentario_e_vazio(tmp_path, monkeypatch):
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAR", raising=False)
    p = tmp_path / ".env"
    p.write_text("# comentário\n\nFOO=abc\nexport BAR=\"x y\"\nSEM_IGUAL\n", encoding="utf-8")
    got = load_dotenv(p)
    assert got == {"FOO": "abc", "BAR": "x y"}
    import os
    assert os.environ["FOO"] == "abc"
    assert os.environ["BAR"] == "x y"


def test_nao_sobrescreve_por_padrao(tmp_path, monkeypatch):
    monkeypatch.setenv("FOO", "já-existe")
    p = tmp_path / ".env"
    p.write_text("FOO=do-arquivo\n", encoding="utf-8")
    load_dotenv(p)
    import os
    assert os.environ["FOO"] == "já-existe"
    load_dotenv(p, override=True)
    assert os.environ["FOO"] == "do-arquivo"


def test_arquivo_inexistente_ok(tmp_path):
    assert load_dotenv(tmp_path / "nao_existe") == {}
