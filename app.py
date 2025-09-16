# ============================================================
# 1. Importações e configuração inicial
# ============================================================
import streamlit as st
import pandas as pd
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

# Bibliotecas opcionais
try:
    import pdfplumber   # leitura de PDFs
except ImportError:
    pdfplumber = None
    st.error("❌ Biblioteca pdfplumber não instalada. Instale com: pip install pdfplumber")

try:
    import plotly.express as px   # gráficos
    import plotly.graph_objects as go
except ImportError:
    px = go = None
    st.warning("⚠️ Plotly não instalado. Instale com: pip install plotly")

# Logger para depuração
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# 2. Estrutura para representar uma transação
# ============================================================
@dataclass
class Transaction:
    date: datetime
    description: str
    value: float
    balance: Optional[float] = None
    transaction_type: Optional[str] = None
    category: Optional[str] = None
    source_file: Optional[str] = None

# ============================================================
# 3. Classe base para padrões
# ============================================================
class BankPatternMatcher:
    def get_date_patterns(self) -> List[str]:
        raise NotImplementedError

    def get_value_patterns(self) -> List[str]:
        raise NotImplementedError

    def get_noise_patterns(self) -> List[str]:
        return []

    def parse_transaction_line(self, line: str, source_file: str) -> Optional[Transaction]:
        raise NotImplementedError

# ============================================================
# 4. Regras concretas de reconhecimento
# ============================================================
class EnhancedBankMatcher(BankPatternMatcher):
    def get_date_patterns(self) -> List[str]:
        return [
            r"\b\d{2}/\d{2}/\d{4}\b",
            r"\b\d{2}/\d{2}/\d{2}\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{2} [A-Za-z]{3} \d{4}\b"
        ]

    def get_value_patterns(self) -> List[str]:
        return [
            r"R?\$?\s*-?\d{1,3}(?:[.,]\d{3})*[.,]\d{2}",
            r"\d+[.,]\d{2}\s*[DC]",
            r"[+-]?\d+[.,]\d{2}",
            r"\d{1,3}(?:,\d{3})*\.\d{2}"
        ]

    def get_noise_patterns(self) -> List[str]:
        return [
            r"SALDO\s*(ANTERIOR|DO DIA|ATUAL|FINAL)",
            r"TOTAL\s*(DE CRÉDITOS|DE DÉBITOS|GERAL)",
            r"EXTRATO\s+DE\s+CONTA",
            r"BANCO\s+\d+",
            r"Página\s+\d+",
            r"INÍCIO\s+DO\s+EXTRATO",
            r"FIM\s+DO\s+EXTRATO",
            r"^Data\s+Histórico\s+Valor",
            r"^Descrição",
            r"^Crédito|^Débito"
        ]

    def parse_transaction_line(self, line: str, source_file: str) -> Optional[Transaction]:
        try:
            # Normalização
            clean_line = re.sub(r"\s+", " ", line.strip())
            if not clean_line:
                return None

            # Pega data
            date_match = None
            for pattern in self.get_date_patterns():
                date_match = re.search(pattern, clean_line)
                if date_match:
                    break
            if not date_match:
                return None
            date_str = date_match.group()
            for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%d %b %Y"):
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    date_obj = None
            if not date_obj:
                return None

            # Pega valor
            value_match = None
            for pattern in self.get_value_patterns():
                value_match = re.search(pattern, clean_line)
                if value_match:
                    break
            if not value_match:
                return None
            value_str = value_match.group()
            value_clean = re.sub(r"[^\d,.-]", "", value_str)
            value_clean = value_clean.replace(".", "").replace(",", ".")
            value = float(value_clean)
            if re.search(r"(D|DEBITO|DEBITO)", clean_line, re.IGNORECASE):
                value = -abs(value)
            elif re.search(r"(C|CREDITO|CREDITO)", clean_line, re.IGNORECASE):
                value = abs(value)

            # Descrição
            description = clean_line.replace(date_str, "").replace(value_str, "").strip()
            description = re.sub(r"\s+", " ", description)

            # Classificação básica
            transaction_type = "DESCONHECIDO"
            if re.search(r"pix", description, re.IGNORECASE):
                transaction_type = "PIX"
            elif re.search(r"(saque|atm)", description, re.IGNORECASE):
                transaction_type = "SAQUE"
            elif re.search(r"(deposito|depósito)", description, re.IGNORECASE):
                transaction_type = "DEPÓSITO"
            elif re.search(r"(transfer|ted|doc)", description, re.IGNORECASE):
                transaction_type = "TRANSFERÊNCIA"
            elif re.search(r"(compra|debito|débito)", description, re.IGNORECASE):
                transaction_type = "COMPRA_DÉBITO"

            # Categoria simples
            category = "OUTROS"
            if any(w in description.lower() for w in ["mercado", "supermercado", "padaria", "restaurante"]):
                category = "ALIMENTAÇÃO"
            elif any(w in description.lower() for w in ["uber", "99", "metrô", "ônibus", "posto"]):
                category = "TRANSPORTE"
            elif any(w in description.lower() for w in ["aluguel", "luz", "energia", "água", "internet"]):
                category = "CASA"

            return Transaction(
                date=date_obj,
                description=description,
                value=value,
                transaction_type=transaction_type,
                category=category,
                source_file=source_file
            )
        except Exception as e:
            logger.error(f"Erro ao processar linha: {clean_line} - {e}")
            return None

# ============================================================
# 5. Classe principal para extrair PDF
# ============================================================
class BankStatementExtractor:
    def __init__(self):
        self.matcher = EnhancedBankMatcher()

    def extract_from_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> pd.DataFrame:
        if not pdfplumber:
            raise ImportError("pdfplumber não instalado")
        transactions = []
        with pdfplumber.open(pdf_bytes) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    transactions.extend(self._process_lines(text.split("\n"), filename))
        return self._transactions_to_dataframe(transactions)

    def _process_lines(self, lines: List[str], filename: str) -> List[Transaction]:
        result = []
        noise_patterns = self.matcher.get_noise_patterns()
        for line in lines:
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in noise_patterns):
                continue
            tx = self.matcher.parse_transaction_line(line, filename)
            if tx and self._is_valid_transaction(tx):
                result.append(tx)
        return result

    def _is_valid_transaction(self, tx: Transaction) -> bool:
        return (
            tx.value is not None and
            tx.description and
            not any(re.search(pattern, tx.description, re.IGNORECASE) for pattern in self.matcher.get_noise_patterns())
        )

    def _transactions_to_dataframe(self, transactions: List[Transaction]) -> pd.DataFrame:
        df = pd.DataFrame([t.__dict__ for t in transactions])
        if not df.empty:
            df["month"] = df["date"].dt.to_period("M")
            df["year"] = df["date"].dt.year
            df["day_of_week"] = df["date"].dt.day_name()
            df["abs_value"] = df["value"].abs()
        return df

# ============================================================
# 6. Funções auxiliares
# ============================================================
def process_multiple_pdfs(uploaded_files) -> pd.DataFrame:
    extractor = BankStatementExtractor()
    dfs = []
    for i, uploaded_file in enumerate(uploaded_files):
        with st.spinner(f"Processando {uploaded_file.name}..."):
            df = extractor.extract_from_pdf_bytes(uploaded_file, uploaded_file.name)
            dfs.append(df)
        st.progress((i + 1) / len(uploaded_files))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def create_summary_charts(df: pd.DataFrame):
    if not px or df.empty:
        return
    st.subheader("📊 Análises Gráficas")
    # Fluxo de caixa mensal
    monthly_flow = df.groupby("month")["value"].sum().reset_index()
    st.plotly_chart(px.line(monthly_flow, x="month", y="value", title="Fluxo de Caixa Mensal"))
    # Gastos por categoria
    expenses = df[df["value"] < 0].groupby("category")["abs_value"].sum().reset_index()
    st.plotly_chart(px.bar(expenses, x="category", y="abs_value", title="Gastos por Categoria"))

# ============================================================
# 7. Interface principal Streamlit
# ============================================================
def main():
    st.title("📑 Extrator Inteligente de Extratos Bancários")
    st.markdown("Carregue seus extratos bancários em PDF e visualize transações, gráficos e métricas.")

    uploaded_files = st.sidebar.file_uploader(
        "Selecione arquivos PDF", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        if st.sidebar.button("🚀 Processar Extratos"):
            df = process_multiple_pdfs(uploaded_files)
            if not df.empty:
                # Métricas
                total = len(df)
                credits = df[df["value"] > 0]["value"].sum()
                debits = df[df["value"] < 0]["value"].sum()
                st.metric("Total Transações", total)
                st.metric("Créditos", f"R$ {credits:,.2f}")
                st.metric("Débitos", f"R$ {debits:,.2f}")
                st.metric("Fluxo Líquido", f"R$ {(credits + debits):,.2f}")

                create_summary_charts(df)

                # Tabela
                st.dataframe(df)

                # Download CSV
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("💾 Baixar CSV", csv, "transacoes.csv", "text/csv")

if __name__ == "__main__":
    main()
