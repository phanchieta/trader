from trader.live.state import PortfolioState, SymbolState, load, reconcile, save


def test_save_load_roundtrip(tmp_path):
    st = PortfolioState(updated="2026-08-29T20:00:00")
    st.get("PETR4").shares = 300.0
    st.get("PETR4").entry = 38.5
    st.get("PETR4").peak = 41.2
    st.get("VALE3").blocked = True
    p = tmp_path / "s.json"
    save(st, p)
    back = load(p)
    assert back.updated == st.updated
    assert back.get("PETR4") == SymbolState(300.0, 38.5, 41.2, False)
    assert back.get("VALE3").blocked is True


def test_load_inexistente_devolve_vazio(tmp_path):
    assert load(tmp_path / "nao_existe.json").symbols == {}


def test_reconcile_sem_divergencia():
    st = PortfolioState()
    st.get("PETR4").shares = 200.0
    _, disc = reconcile(st, {"PETR4": 200.0})
    assert disc == []


def test_reconcile_ajusta_para_a_corretora():
    st = PortfolioState()
    st.get("PETR4").shares = 200.0
    st.get("PETR4").entry = 40.0
    st, disc = reconcile(st, {"PETR4": 100.0})
    assert len(disc) == 1
    assert st.get("PETR4").shares == 100.0


def test_reconcile_posicao_sumiu_limpa_entry_e_peak():
    st = PortfolioState()
    st.get("PETR4").shares = 200.0
    st.get("PETR4").entry = 40.0
    st.get("PETR4").peak = 44.0
    st.get("PETR4").blocked = True
    st, disc = reconcile(st, {})            # corretora não tem nada
    ss = st.get("PETR4")
    assert (ss.shares, ss.entry, ss.peak, ss.blocked) == (0.0, 0.0, 0.0, False)
