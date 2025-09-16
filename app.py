import streamlit as st
import pandas as pd
import re
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
import logging
import io
import pdfplumber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Extrator de Extratos Bancários - Profissional", page_icon="🏦", layout="wide")

@dataclass
class Transaction:
    date: datetime
    description: str
    value: float
    balance: Optional[float] = None
    transaction_type: Optional[str] = None
    category: Optional[str] = None
    source_file: Optional[str] = None
    source_bank: Optional[str] = None
    confidence_score: float = 0.0

def parse_monetary_string(s: str) -> Optional[float]:
    if not s or not isinstance(s, str):
        return None
    original = s.strip()
    parentheses_negative = original.startswith("(") and original.endswith(")")
    if parentheses_negative:
        s = s[1:-1].strip()
    explicit_negative = s.startswith("-")
    explicit_positive = s.startswith("+")
    s = re.sub(r"[Rr]\$|\s+", "", s)
    s = s.lstrip("+-")
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) == 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    s = re.sub(r"[^\d.]", "", s)
    if not s:
        return None
    try:
        val = float(s)
        if parentheses_negative or explicit_negative:
            val = -abs(val)
        elif explicit_positive:
            val = abs(val)
        return val
    except ValueError:
        return None

class ProfessionalBankMatcher:
    def __init__(self):
        self.noise_patterns = [
            r'^(SALDO|TOTAL|SUBTOTAL|RESUMO|EXTRATO|PÁGINA|AGÊNCIA|CONTA|DATA|DESCRIÇÃO|PERÍODO|CLIENTE)',
            r'(DISPONÍVEL|BLOQUEADO|LIMITE|VENCIMENTO|TAXA|JUROS)',
            r'(CRÉDITO|CRÉDITOS|DÉBITO|DÉBITOS)$',
            r'(MOVIMENTAÇÃO|LANÇAMENTO|HISTÓRICO)\s+(TOTAL|DO\s+PERÍODO)',
            r'^\s*[-=_*]{3,}\s*$',
            r'^\s*\d+\s+of\s+\d+\s*$',
            r'(SAC|OUVIDORIA|TELEFONE|ATENDIMENTO)',
            r'(CNPJ|CPF):\s*[\d/.()-]+\s*$',
            r'^\s*(Períodos?|Data/Hora|Saldo\s+em|Opção\s+de)',
            r'(www\.|http|\.com|\.br)',
            r'^\s*\d{2}/\d{2}/\d{4}\s*$',
            r'^\s*R\$\s*[\d.,]+\s*$',
        ]

    def is_noise_line(self, line: str) -> bool:
        if not line or len(line.strip()) < 3:
            return True
        line_clean = line.strip()
        for pattern in self.noise_patterns:
            if re.search(pattern, line_clean, re.IGNORECASE):
                return True
        if len(line_clean) < 10 or re.match(r'^[\d\s.,/-]+$', line_clean):
            return True
        if any(word in line_clean.lower() for word in ['cabeçalho', 'rodapé', 'header', 'footer']):
            return True
        return False

    def detect_bank_format(self, content_lines: List[str]) -> str:
        content = '\n'.join(content_lines[:50])
        if re.search(r'BB\s+Rende\s+Fácil|Transferência\s+enviada|Folha\s+de\s+Pagamento', content, re.IGNORECASE):
            return 'BB'
        if re.search(r'RESGATE\s+INVEST\s+FACIL|GASTOS\s+CARTAO\s+DE\s+CREDITO', content, re.IGNORECASE):
            return 'BRADESCO'
        if re.search(r'Banco\s+Inter|Saldo\s+do\s+dia:\s+R\$|Pix\s+(enviado|recebido):', content, re.IGNORECASE):
            return 'INTER'
        if re.search(r'SAC\s+CAIXA|CRED\s+TED|PAG\s+BOLETO|ENVIO\s+PIX', content, re.IGNORECASE):
            return 'CAIXA'
        if re.search(r'XP\s+INVESTIMENTOS|RESGATE.*FIRF|TED\s+BCO\s+\d+', content, re.IGNORECASE):
            return 'XP'
        if re.search(r'BOLETO\s+PAGO|PIX\s+ENVIADO.*\d{2}\s*/\s*dez|SISPAG', content, re.IGNORECASE):
            return 'ITAU'
        if re.search(r'TED\s+RECEBIDA.*\d{11}|APLICACAO\s+CONTAMAX|PIX\s+ENVIADO.*ML', content, re.IGNORECASE):
            return 'SANTANDER'
        if re.search(r'SICOOB|PIX\s+RECEB\.OUTRA\s+IF|CRÉD\.TED-STR', content, re.IGNORECASE):
            return 'SICOOB'
        if re.search(r'Nu\s+Financeira|Nu\s+Pagamentos|VL\s+REPRESENTACAO', content, re.IGNORECASE):
            return 'NUBANK'
        return 'GENERIC'

class BankParser:
    def __init__(self):
        self.current_date = None

    def parse_bb(self, lines: List[str]) -> List[Transaction]:
        transactions = []
        for line in lines:
            if len(line.strip()) < 20:
                continue
            bb_pattern = re.compile(
                r'(\d{2}/\d{2}/\d{4})\s+'
                r'(\d{4})\s+'
                r'(\d{5})\s+'
                r'(.+?)\s+'
                r'([\d.,]+)\s+'
                r'([\d.,]+)\s*([CD])\s*'
                r'(?:([\d.,]+)\s*([CD]))?',
                re.IGNORECASE
            )
            match = bb_pattern.search(line)
            if not match:
                continue
            try:
                date_obj = datetime.strptime(match.group(1), '%d/%m/%Y')
                description = match.group(4).strip()
                valor_str = match.group(6) if match.group(6) else match.group(5)
                valor_dc = match.group(7).upper() if match.group(7) else 'D'
                value = parse_monetary_string(valor_str)
                if value is None:
                    continue
                if valor_dc == 'D':
                    value = -abs(value)
                else:
                    value = abs(value)
                balance = None
                if match.group(8):
                    balance = parse_monetary_string(match.group(8))
                    if balance:
                        balance = abs(balance)
                transaction = Transaction(
                    date=date_obj,
                    description=description,
                    value=value,
                    balance=balance,
                    source_bank='BB',
                    confidence_score=0.95
                )
                transactions.append(transaction)
            except Exception as e:
                continue
        return transactions

    def parse_bradesco(self, lines: List[str]) -> List[Transaction]:
        transactions = []
        current_date = None
        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 10:
                continue
            if re.search(r'SALDO\s+(ANTERIOR|ATUAL|FINAL|TOTAL)', line_clean, re.IGNORECASE):
                continue
            date_match = re.match(r'(\d{2}/\d{2}/\d{4})', line_clean)
            if date_match:
                try:
                    current_date = datetime.strptime(date_match.group(1), '%d/%m/%Y')
                except:
                    continue
            if not current_date:
                continue
            bradesco_pattern = re.compile(
                r'(?:\d{2}/\d{2}/\d{4}\s+)?'
                r'(.+?)\s+'
                r'(\d+)\s+'
                r'([\d.,]+|-[\d.,]+|\s*)\s*'
                r'([\d.,]+|-[\d.,]+|\s*)\s*'
                r'([\d.,]+|-[\d.,]+)',
                re.IGNORECASE
            )
            match = bradesco_pattern.search(line_clean)
            if not match:
                continue
            description = match.group(1).strip()
            credit_str = match.group(3).strip() if match.group(3) else ""
            debit_str = match.group(4).strip() if match.group(4) else ""
            saldo_str = match.group(5).strip()
            if 'SALDO' in description.upper():
                continue
            value = None
            if credit_str and credit_str not in ['', '-'] and not credit_str.startswith('-'):
                value = parse_monetary_string(credit_str)
                if value:
                    value = abs(value)
            elif debit_str and debit_str not in ['', '-']:
                value = parse_monetary_string(debit_str)
                if value:
                    value = -abs(value)
            if value is None or abs(value) < 0.01:
                continue
            saldo = parse_monetary_string(saldo_str)
            transaction = Transaction(
                date=current_date,
                description=description,
                value=value,
                balance=abs(saldo) if saldo else None,
                source_bank='BRADESCO',
                confidence_score=0.85
            )
            transactions.append(transaction)
        return transactions

    def parse_inter(self, lines: List[str]) -> List[Transaction]:
        transactions = []
        current_date = None
        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 15:
                continue
            date_pattern = r'(\d{1,2})\s+de\s+(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+de\s+(\d{4})'
            date_match = re.search(date_pattern, line_clean, re.IGNORECASE)
            if date_match:
                try:
                    day = int(date_match.group(1))
                    month_name = date_match.group(2).lower()
                    year = int(date_match.group(3))
                    months = {
                        'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4, 'maio': 5,
                        'junho': 6, 'julho': 7, 'agosto': 8, 'setembro': 9, 'outubro': 10,
                        'novembro': 11, 'dezembro': 12
                    }
                    month = months.get(month_name, 1)
                    current_date = datetime(year, month, day)
                except:
                    continue
            if not current_date:
                continue
            inter_pattern = re.compile(
                r'([^-+R$]+?)\s*'
                r'(-?R\$\s*[\d.,]+)\s+'
                r'R\$\s*([\d.,]+|-[\d.,]+)',
                re.IGNORECASE
            )
            match = inter_pattern.search(line_clean)
            if not match:
                continue
            description = match.group(1).strip()
            value_str = match.group(2)
            saldo_str = match.group(3)
            if re.search(r'\d+\s+de\s+\w+|saldo\s+do\s+dia', description, re.IGNORECASE):
                continue
            value = parse_monetary_string(value_str)
            if value is None or abs(value) < 0.01:
                continue
            saldo = parse_monetary_string(saldo_str)
            transaction = Transaction(
                date=current_date,
                description=description,
                value=value,
                balance=abs(saldo) if saldo else None,
                source_bank='INTER',
                confidence_score=0.88
            )
            transactions.append(transaction)
        return transactions

    def parse_caixa(self, lines: List[str]) -> List[Transaction]:
        transactions = []
        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 15:
                continue
            caixa_pattern = re.compile(
                r'(\d{2}/\d{2}/\d{4})\s+'
                r'(\d+)\s+'
                r'(.+?)\s+'
                r'([\d.,]+)\s*([CD])\s+'
                r'([\d.,]+)\s*([CD])',
                re.IGNORECASE
            )
            match = caixa_pattern.search(line_clean)
            if not match:
                continue
            try:
                date_obj = datetime.strptime(match.group(1), '%d/%m/%Y')
                description = match.group(3).strip()
                valor_str = match.group(4)
                valor_dc = match.group(5).upper()
                saldo_str = match.group(6)
                value = parse_monetary_string(valor_str)
                if value is None:
                    continue
                if valor_dc == 'D':
                    value = -abs(value)
                else:
                    value = abs(value)
                saldo = parse_monetary_string(saldo_str)
                transaction = Transaction(
                    date=date_obj,
                    description=description,
                    value=value,
                    balance=abs(saldo) if saldo else None,
                    source_bank='CAIXA',
                    confidence_score=0.92
                )
                transactions.append(transaction)
            except Exception:
                continue
        return transactions

    def parse_itau(self, lines: List[str]) -> List[Transaction]:
        transactions = []
        current_date = None
        current_year = datetime.now().year
        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 10:
                continue
            if re.search(r'SALDO\s+(ANTERIOR|TOTAL|DISPONÍVEL)', line_clean, re.IGNORECASE):
                continue
            date_match = re.search(r'(\d{1,2})\s*/\s*(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)', line_clean, re.IGNORECASE)
            if date_match:
                try:
                    day = int(date_match.group(1))
                    month_abbr = date_match.group(2).lower()
                    months = {
                        'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
                        'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12
                    }
                    month = months.get(month_abbr, 1)
                    current_date = datetime(current_year, month, day)
                except:
                    continue
            if not current_date:
                continue
            value_matches = re.findall(r'-?([\d.,]+)', line_clean)
            if not value_matches:
                continue
            values = [parse_monetary_string(v) for v in value_matches]
            values = [v for v in values if v and abs(v) > 0.01]
            if not values:
                continue
            value = max(values, key=abs)
            debit_keywords = ['BOLETO PAGO', 'PIX ENVIADO', 'PAGTO CDC', 'FIN VEIC', 'SISPAG TRIB']
            credit_keywords = ['PIX TRANSF', 'TED.*RECEBIDA', 'MOV TIT COB DISP', 'INT RESGATE']
            is_debit = any(kw in line_clean.upper() for kw in debit_keywords)
            is_credit = any(re.search(kw, line_clean.upper()) for kw in credit_keywords)
            if is_debit:
                value = -abs(value)
            elif is_credit:
                value = abs(value)
            elif line_clean.startswith('-'):
                value = -abs(value)
            description = re.sub(r'\d{1,2}\s*/\s*\w{3}|\d{2}/\d{2}/\d{4}', '', line_clean).strip()
            description = re.sub(r'-?[\d.,]+', '', description).strip()
            description = re.sub(r'\s+', ' ', description)[:100]
            if len(description) < 3:
                description = "Transação Itaú"
            transaction = Transaction(
                date=current_date,
                description=description,
                value=value,
                balance=None,
                source_bank='ITAU',
                confidence_score=0.80
            )
            transactions.append(transaction)
        return transactions

    def parse_generic(self, lines: List[str]) -> List[Transaction]:
        transactions = []
        for line in lines:
            pattern = re.compile(r'(\d{2}/\d{2}/\d{4})\s+(.+?)\s+(-?[\d.,]+)\s+(-?[\d.,]+)?', re.IGNORECASE)
            match = pattern.search(line)
            if not match:
                continue
            try:
                date_obj = datetime.strptime(match.group(1), '%d/%m/%Y')
                description = match.group(2).strip()
                value = parse_monetary_string(match.group(3))
                balance = parse_monetary_string(match.group(4)) if match.group(4) else None
                if value is None:
                    continue
                transaction = Transaction(
                    date=date_obj,
                    description=description,
                    value=value,
                    balance=balance,
                    source_bank='GENERIC',
                    confidence_score=0.70
                )
                transactions.append(transaction)
            except Exception:
                continue
        return transactions

    def parse_xp(self, lines: List[str]) -> List[Transaction]:
        st.warning("Parser para XP não implementado totalmente. Usando genérico.")
        return self.parse_generic(lines)

    def parse_santander(self, lines: List[str]) -> List[Transaction]:
        st.warning("Parser para Santander não implementado totalmente. Usando genérico.")
        return self.parse_generic(lines)

    def parse_sicoob(self, lines: List[str]) -> List[Transaction]:
        st.warning("Parser para Sicoob não implementado totalmente. Usando genérico.")
        return self.parse_generic(lines)

    def parse_nubank(self, lines: List[str]) -> List[Transaction]:
        st.warning("Parser para Nubank não implementado totalmente. Usando genérico.")
        return self.parse_generic(lines)

class BankStatementExtractor:
    def __init__(self):
        self.matcher = ProfessionalBankMatcher()
        self.parser = BankParser()

    def extract_from_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> pd.DataFrame:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                all_lines = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_lines.extend(text.splitlines())
            clean_lines = [line for line in all_lines if not self.matcher.is_noise_line(line)]
            if not clean_lines:
                return pd.DataFrame()
            bank_format = self.matcher.detect_bank_format(clean_lines)
            transactions = []
            if bank_format == 'BB':
                transactions = self.parser.parse_bb(clean_lines)
            elif bank_format == 'BRADESCO':
                transactions = self.parser.parse_bradesco(clean_lines)
            elif bank_format == 'INTER':
                transactions = self.parser.parse_inter(clean_lines)
            elif bank_format == 'CAIXA':
                transactions = self.parser.parse_caixa(clean_lines)
            elif bank_format == 'ITAU':
                transactions = self.parser.parse_itau(clean_lines)
            elif bank_format == 'XP':
                transactions = self.parser.parse_xp(clean_lines)
            elif bank_format == 'SANTANDER':
                transactions = self.parser.parse_santander(clean_lines)
            elif bank_format == 'SICOOB':
                transactions = self.parser.parse_sicoob(clean_lines)
            elif bank_format == 'NUBANK':
                transactions = self.parser.parse_nubank(clean_lines)
            elif bank_format == 'GENERIC':
                transactions = self.parser.parse_generic(clean_lines)
            if not transactions:
                return pd.DataFrame()
            df = self._transactions_to_dataframe(transactions)
            if not df.empty:
                df['arquivo'] = filename
                df['banco_detectado'] = bank_format
            return df
        except Exception as e:
            logger.exception(f"Erro ao processar {filename}: {e}")
            return pd.DataFrame()

    def _transactions_to_dataframe(self, transactions: List[Transaction]) -> pd.DataFrame:
        if not transactions:
            return pd.DataFrame()
        data = []
        for t in transactions:
            tipo = self._classify_transaction_type(t.description, t.value)
            categoria = self._categorize_transaction(t.description)
            data.append({
                'data': t.date,
                'descricao': t.description,
                'valor': t.value,
                'saldo': t.balance,
                'tipo': tipo,
                'categoria': categoria,
                'confianca': t.confidence_score,
                'banco_detectado': t.source_bank
            })
        df = pd.DataFrame(data)
        df['data'] = pd.to_datetime(df['data'])
        df = df.sort_values('data').reset_index(drop=True)
        df['mes'] = df['data'].dt.month
        df['ano'] = df['data'].dt.year
        df['dia_semana'] = df['data'].dt.day_name()
        df['valor_absoluto'] = df['valor'].abs()
        df['mes_ano'] = df['data'].dt.to_period('M').astype(str)
        df['eh_debito'] = df['valor'] < 0
        df['eh_credito'] = df['valor'] > 0
        # Validação simples de saldos
        for i in range(1, len(df)):
            if pd.notna(df['saldo'][i-1]) and pd.notna(df['valor'][i]) and pd.notna(df['saldo'][i]):
                expected_balance = df['saldo'][i-1] + df['valor'][i]
                if abs(expected_balance - df['saldo'][i]) > 0.01:
                    logger.warning(f"Inconsistência de saldo na linha {i}: esperado {expected_balance}, encontrado {df['saldo'][i]}")
        return df

    def _classify_transaction_type(self, description: str, value: float) -> str:
        desc = description.lower()
        if value < 0:
            if any(word in desc for word in ['pix', 'ted', 'doc', 'transferência']):
                return 'TRANSFERÊNCIA_SAÍDA'
            if any(word in desc for word in ['saque', 'atm', 'retirada']):
                return 'SAQUE'
            if any(word in desc for word in ['boleto', 'pagamento']):
                return 'PAGAMENTO'
            if any(word in desc for word in ['tarifa', 'taxa', 'juros']):
                return 'TARIFA'
            return 'DÉBITO'
        else:
            if any(word in desc for word in ['pix', 'ted', 'transferência']):
                return 'TRANSFERÊNCIA_ENTRADA'
            if any(word in desc for word in ['salário', 'remuneração']):
                return 'SALÁRIO'
            if any(word in desc for word in ['depósito']):
                return 'DEPÓSITO'
            return 'CRÉDITO'

    def _categorize_transaction(self, description: str) -> str:
        desc = description.lower()
        if any(word in desc for word in ['mercado', 'supermercado', 'restaurante', 'lanche']):
            return 'ALIMENTAÇÃO'
        if any(word in desc for word in ['combustível', 'posto', 'uber', 'taxi']):
            return 'TRANSPORTE'
        if any(word in desc for word in ['energia', 'água', 'telefone', 'internet']):
            return 'CASA'
        if any(word in desc for word in ['farmácia', 'hospital', 'médico']):
            return 'SAÚDE'
        return 'OUTROS'

def process_multiple_pdfs(uploaded_files):
    extractor = BankStatementExtractor()
    all_dfs = []
    progress = st.progress(0)
    status = st.empty()
    for i, uploaded_file in enumerate(uploaded_files):
        status.text(f'Processando {uploaded_file.name} ({i+1}/{len(uploaded_files)})')
        try:
            pdf_bytes = uploaded_file.read()
            df = extractor.extract_from_pdf_bytes(pdf_bytes, uploaded_file.name)
            if df.empty:
                st.warning(f"⚠️ {uploaded_file.name}: nenhuma transação identificada")
            else:
                all_dfs.append(df)
                banco = df['banco_detectado'].iloc[0] if not df.empty else 'N/A'
                confianca = df['confianca'].mean()
                st.success(f"✅ {uploaded_file.name}: {len(df)} transações | Banco: **{banco}** | Confiança: {confianca:.2f}")
        except Exception as e:
            st.error(f"❌ Erro ao processar {uploaded_file.name}: {e}")
        progress.progress((i + 1) / len(uploaded_files))
    status.text("Processamento concluído")
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values('data').reset_index(drop=True)
        return combined
    return pd.DataFrame()

def main():
    st.title("🏦 Extrator de Extratos Bancários - Versão Profissional")
    st.markdown("Algoritmo robusto para processamento de múltiplos formatos bancários com alta precisão.")
    uploaded_files = st.file_uploader(
        "Escolha arquivos PDF de extratos bancários",
        type="pdf",
        accept_multiple_files=True,
        help="Carregue extratos de BB, Bradesco, Inter, Caixa, Itaú, etc."
    )
    if uploaded_files:
        if st.button("🚀 Processar Extratos", type="primary"):
            with st.spinner("Processando com algoritmo profissional..."):
                df = process_multiple_pdfs(uploaded_files)
                if df.empty:
                    st.error("❌ Nenhuma transação extraída. Verifique se os PDFs são extratos bancários válidos.")
                else:
                    st.success(f"✅ Extraídas {len(df)} transações no total.")
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Baixar CSV",
                        data=csv,
                        file_name="extratos_processados.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
