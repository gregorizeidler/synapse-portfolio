from __future__ import annotations
import pandas as pd
import yfinance as yf
from .utils import DATA_DIR, ensure_dirs, load_config

def fetch_prices(cfg) -> pd.DataFrame:
    """
    Baixa dados históricos de preços do Yahoo Finance.
    
    Args:
        cfg: Configuração do projeto
        
    Returns:
        DataFrame com preços de fechamento dos tickers
    """
    tickers = cfg["universe"]["tickers"]
    start = cfg["universe"]["start_date"]
    end = cfg["universe"]["end_date"]
    freq = cfg["universe"]["frequency"]
    include_cash = cfg["universe"].get("include_cash", True)
    cash_sym = cfg["universe"].get("cash_symbol", "CASH")

    # Remove CASH da lista de download (se existir)
    dl_tickers = [t for t in tickers if t != cash_sym]
    
    print(f"[data] Baixando {len(dl_tickers)} tickers do Yahoo Finance...")
    print(f"[data] Período: {start} até {end}")
    
    # Download do Yahoo Finance
    df = yf.download(
        tickers=dl_tickers,
        start=start,
        end=end,
        interval=freq,
        auto_adjust=True,
        progress=True,
        group_by='ticker'
    )
    
    # Extrai preços de fechamento
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].copy() if "Close" in df.columns.get_level_values(0) else df.xs('Close', level=1, axis=1)
    else:
        # Apenas 1 ticker
        close = df[["Close"]].copy() if "Close" in df.columns else df.to_frame(name=dl_tickers[0])
        close.columns = dl_tickers
    
    # Remove linhas completamente vazias
    close = close.dropna(how="all")
    
    # Remove colunas com mais de 50% de NaNs
    threshold = len(close) * 0.5
    close = close.dropna(thresh=threshold, axis=1)
    
    # Forward fill para preencher gaps pequenos
    close = close.ffill().bfill()
    
    if close.empty:
        raise RuntimeError("Yahoo Finance não retornou dados válidos. Verifique os tickers e conexão com internet.")
    
    print(f"[data] Download concluído: {close.shape[1]} ativos, {close.shape[0]} dias")
    
    # Adiciona CASH (ativo sintético com preço constante)
    if include_cash:
        close[cash_sym] = 1.0
        print(f"[data] Adicionado ativo {cash_sym} (preço constante = 1.0)")
    
    return close

def main():
    """
    Pipeline principal de download de dados.
    """
    ensure_dirs()
    cfg = load_config()
    
    try:
        close = fetch_prices(cfg)
        out_path = DATA_DIR / "prices.csv"
        close.to_csv(out_path)
        print(f"[data] ✓ Salvo: {out_path}")
        print(f"[data] ✓ Shape final: {close.shape}")
        print(f"[data] ✓ Ativos: {list(close.columns)}")
    except Exception as e:
        print(f"[data] ✗ ERRO: {e}")
        print("[data] Verifique:")
        print("  1. Conexão com internet")
        print("  2. Tickers válidos no config.yaml")
        print("  3. Biblioteca yfinance instalada (pip install yfinance)")
        raise

if __name__ == "__main__":
    main()
