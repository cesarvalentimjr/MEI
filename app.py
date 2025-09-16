"""
Extrator de Extratos Bancários - Versão Profissional
Algoritmo robusto para múltiplos formatos bancários
Correções principais:
- Filtros rigorosos contra linhas de ruído
- Parsers específicos por banco com validação
- Detecção precisa de débito/crédito
- Suporte a transações sem data repetida
- Validação de consistência de dados
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import logging
import io

try:
    import pdfplumber
    PDF_AVAILABLE = True
except Exception as e:
    pdfplumber = None
    PDF_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    px = go = None
    PLOTLY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not PDF_AVAILABLE:
    st.error("❌ pdfplumber não encontrado — instale `pdfplumber` para ativar leitura de PDFs.")
    st.stop()

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
    """Parse robusto de valores monetários"""
    if not s or not isinstance(s, str):
        return None
    
    original = s.strip()
    
    # Detecta parênteses (negativo)
    parentheses_negative = original.startswith("(") and original.endswith(")")
    if parentheses_negative:
        s = s[1:-1].strip()
    
    # Detecta sinal explícito
    explicit_negative = s.startswith("-")
    explicit_positive = s.startswith("+")
    
    # Remove símbolos monetários
    s = re.sub(r"[Rr]\$|\s+", "", s)
    s = s.lstrip("+-")
    
    # Trata separadores
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
    except:
        return None

class ProfessionalBankMatcher:
    def __init__(self):
        # Padrões de ruído rigorosos
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
            r'^\s*\d{2}/\d{2}/\d{4}\s*$',  # Só data
            r'^\s*R\$\s*[\d.,]+\s*$',      # Só valor
        ]
    
    def is_noise_line(self, line: str) -> bool:
        """Filtro rigoroso contra ruído"""
        if not line or len(line.strip()) < 3:
            return True
        
        line_clean = line.strip()
        
        # Padrões explícitos de ruído
        for pattern in self.noise_patterns:
            if re.search(pattern, line_clean, re.IGNORECASE):
                return True
        
        # Linhas muito curtas ou só números/símbolos
        if len(line_clean) < 10 or re.match(r'^[\d\s.,/-]+$', line_clean):
            return True
        
        # Linhas de cabeçalho/rodapé
        if any(word in line_clean.lower() for word in ['cabeçalho', 'rodapé', 'header', 'footer']):
            return True
        
        return False
    
    def detect_bank_format(self, content_lines: List[str]) -> str:
        """Detecção robusta baseada em múltiplas linhas"""
        content = '\n'.join(content_lines[:50])  # Analisa primeiras 50 linhas
        
        # Banco do Brasil
        if re.search(r'BB\s+Rende\s+Fácil|Transferência\s+enviada|Folha\s+de\s+Pagamento', content, re.IGNORECASE):
            return 'BB'
        
        # Bradesco
        if re.search(r'RESGATE\s+INVEST\s+FACIL|GASTOS\s+CARTAO\s+DE\s+CREDITO', content, re.IGNORECASE):
            return 'BRADESCO'
        
        # Banco Inter
        if re.search(r'Banco\s+Inter|Saldo\s+do\s+dia:\s+R\$|Pix\s+(enviado|recebido):', content, re.IGNORECASE):
            return 'INTER'
        
        # Caixa
        if re.search(r'SAC\s+CAIXA|CRED\s+TED|PAG\s+BOLETO|ENVIO\s+PIX', content, re.IGNORECASE):
            return 'CAIXA'
        
        # XP Investimentos
        if re.search(r'XP\s+INVESTIMENTOS|RESGATE.*FIRF|TED\s+BCO\s+\d+', content, re.IGNORECASE):
            return 'XP'
        
        # Itaú
        if re.search(r'BOLETO\s+PAGO|PIX\s+ENVIADO.*\d{2}\s*/\s*dez|SISPAG', content, re.IGNORECASE):
            return 'ITAU'
        
        # Santander
        if re.search(r'TED\s+RECEBIDA.*\d{11}|APLICACAO\s+CONTAMAX|PIX\s+ENVIADO.*ML', content, re.IGNORECASE):
            return 'SANTANDER'
        
        # Sicoob
        if re.search(r'SICOOB|PIX\s+RECEB\.OUTRA\s+IF|CRÉD\.TED-STR', content, re.IGNORECASE):
            return 'SICOOB'
        
        # Nubank
        if re.search(r'Nu\s+Financeira|Nu\s+Pagamentos|VL\s+REPRESENTACAO', content, re.IGNORECASE):
            return 'NUBANK'
        
        return 'GENERIC'

class BankParser:
    def __init__(self):
        self.current_date = None  # Para bancos que não repetem data
    
    def parse_bb(self, lines: List[str]) -> List[Transaction]:
        """Parser Banco do Brasil - formato estruturado"""
        transactions = []
        
        for line in lines:
            if len(line.strip()) < 20:
                continue
            
            # Padrão BB: DATA AGENCIA LOTE HISTORICO DOCUMENTO VALOR D/C [SALDO D/C]
            bb_pattern = re.compile(
                r'(\d{2}/\d{2}/\d{4})\s+'     # data
                r'(\d{4})\s+'                 # agencia
                r'(\d{5})\s+'                 # lote
                r'(.+?)\s+'                   # historico
                r'([\d.,]+)\s+'               # documento ou valor
                r'([\d.,]+)\s*([CD])\s*'      # valor final + D/C
                r'(?:([\d.,]+)\s*([CD]))?',   # saldo opcional
                re.IGNORECASE
            )
            
            match = bb_pattern.search(line)
            if not match:
                continue
            
            try:
                date_obj = datetime.strptime(match.group(1), '%d/%m/%Y')
                description = match.group(4).strip()
                
                # O valor correto pode estar no grupo 6 ou 7
                valor_str = match.group(6) if match.group(6) else match.group(5)
                valor_dc = match.group(7).upper() if match.group(7) else 'D'
                
                value = parse_monetary_string(valor_str)
                if value is None:
                    continue
                
                # Aplicar sinal correto
                if valor_dc == 'D':
                    value = -abs(value)
                else:
                    value = abs(value)
                
                # Saldo se presente
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
        """Parser Bradesco - colunas e datas não repetidas"""
        transactions = []
        current_date = None
        
        for line in lines:
            line_clean = line.strip()
            
            if len(line_clean) < 10:
                continue
            
            # Ignora linhas de saldo
            if re.search(r'SALDO\s+(ANTERIOR|ATUAL|FINAL|TOTAL)', line_clean, re.IGNORECASE):
                continue
            
            # Tenta extrair data no início da linha
            date_match = re.match(r'(\d{2}/\d{2}/\d{4})', line_clean)
            if date_match:
                try:
                    current_date = datetime.strptime(date_match.group(1), '%d/%m/%Y')
                except:
                    continue
            
            if not current_date:
                continue
            
            # Padrão Bradesco: [DATA] LANÇAMENTO DCTO. [CRÉDITO] [DÉBITO] SALDO
            bradesco_pattern = re.compile(
                r'(?:\d{2}/\d{2}/\d{4}\s+)?'    # data opcional
                r'(.+?)\s+'                      # descrição
                r'(\d+)\s+'                      # documento
                r'([\d.,]+|-[\d.,]+|\s*)\s*'     # crédito
                r'([\d.,]+|-[\d.,]+|\s*)\s*'     # débito
                r'([\d.,]+|-[\d.,]+)',           # saldo
                re.IGNORECASE
            )
            
            match = bradesco_pattern.search(line_clean)
            if not match:
                continue
            
            description = match.group(1).strip()
            credit_str = match.group(3).strip() if match.group(3) else ""
            debit_str = match.group(4).strip() if match.group(4) else ""
            saldo_str = match.group(5).strip()
            
            # Ignora se descrição contém "SALDO"
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
        """Parser Banco Inter - formato com datas por extenso"""
        transactions = []
        current_date = None
        
        for line in lines:
            line_clean = line.strip()
            
            if len(line_clean) < 15:
                continue
            
            # Detecta data por extenso
            date_pattern = r'(\d{1,2})\s+de\s+(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+de\s+(\d{4})'
            date_match = re.search(date_pattern, line_clean, re.IGNORECASE)
            
            if date_match:
                try:
                    day = int(date_match.group(1))
                    month_name = date_match.group(2).lower()
                    year = int(date_match.group(3))
                    
                    months = {
                        'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4,
                        'maio': 5, 'junho': 6, 'julho': 7, 'agosto': 8,
                        'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12
                    }
                    
                    month = months.get(month_name, 1)
                    current_date = datetime(year, month, day)
                except:
                    continue
            
            if not current_date:
                continue
            
            # Padrão Inter: descrição -R$ valor R$ saldo
            inter_pattern = re.compile(
                r'([^-+R$]+?)\s*'              # descrição
                r'(-?R\$\s*[\d.,]+)\s+'       # valor
                r'R\$\s*([\d.,]+|-[\d.,]+)',  # saldo
                re.IGNORECASE
            )
            
            match = inter_pattern.search(line_clean)
            if not match:
                continue
            
            description = match.group(1).strip()
            value_str = match.group(2)
            saldo_str = match.group(3)
            
            # Filtra descrições de data ou ruído
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
        """Parser Caixa - formato com D/C explícito"""
        transactions = []
        
        for line in lines:
            line_clean = line.strip()
            
            if len(line_clean) < 15:
                continue
            
            # Padrão Caixa: DATA Nr.Doc HISTÓRICO VALOR D/C SALDO D/C
            caixa_pattern = re.compile(
                r'(\d{2}/\d{2}/\d{4})\s+'      # data
                r'(\d+)\s+'                    # documento
                r'(.+?)\s+'                    # histórico
                r'([\d.,]+)\s*([CD])\s+'       # valor + D/C
                r'([\d.,]+)\s*([CD])',         # saldo + D/C
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
                
                # Aplicar sinal baseado em D/C
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
        """Parser Itaú - formato específico com datas dd/mmm"""
        transactions = []
        current_date = None
        current_year = 2024  # Inferir do contexto
        
        for line in lines:
            line_clean = line.strip()
            
            if len(line_clean) < 10:
                continue
            
            # Ignora linhas de saldo
            if re.search(r'SALDO\s+(ANTERIOR|TOTAL|DISPONÍVEL)', line_clean, re.IGNORECASE):
                continue
            
            # Detecta data dd / mmm
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
            
            # Extrai valor (negativo típico no Itaú)
            value_matches = re.findall(r'-?([\d.,]+)', line_clean)
            if not value_matches:
                continue
            
            # Pega o maior valor como transação
            values = [parse_monetary_string(v) for v in value_matches]
            values = [v for v in values if v and abs(v) > 0.01]
            
            if not values:
                continue
            
            value = max(values, key=abs)
            
            # Classificar baseado em palavras-chave
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
            
            # Limpar descrição
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

class BankStatementExtractor:
    def __init__(self):
        self.matcher = ProfessionalBankMatcher()
        self.parser = BankParser()
    
    def extract_from_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> pd.DataFrame:
        """Extrai transações com algoritmo robusto"""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                all_lines = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_lines.extend(text.splitlines())
            
            # Filtra ruído
            clean_lines = [line for line in all_lines if not self.matcher.is_noise_line(line)]
            
            if not clean_lines:
                return pd.DataFrame()
            
            # Detecta banco
            bank_format = self.matcher.detect_bank_format(clean_lines)
            
            # Parse específico por banco
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
            # Adicionar outros bancos...
            
            if not transactions:
                return pd.DataFrame()
            
            # Converter para DataFrame
            df = self._transactions_to_dataframe(transactions)
            
            if not df.empty:
                df['arquivo'] = filename
                df['banco_detectado'] = bank_format
            
            return df
            
        except Exception as e:
            logger.exception(f"Erro ao processar {filename}: {e}")
            return pd.DataFrame()
    
    def _transactions_to_dataframe(self, transactions: List[Transaction]) -> pd.DataFrame:
        """Converte transações para DataFrame"""
        if not transactions:
            return pd.DataFrame()
        
        data = []
        for t in transactions:
            # Classificar tipo e categoria
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
        
        # Colunas auxiliares
        df['mes'] = df['data'].dt.month
        df['ano'] = df['data'].dt.year
        df['dia_semana'] = df['data'].dt.day_name()
        df['valor_absoluto'] = df['valor'].abs()
        df['mes_ano'] = df['data'].dt.to_period('M').astype(str)
        df['eh_debito'] = df['valor'] < 0
        df['eh_credito'] = df['valor'] > 0
        
        return df
    
    def _classify_transaction_type(self, description: str, value: float) -> str:
        """Classifica tipo de transação"""
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
        """Categoriza transação"""
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
    """Processa múltiplos PDFs"""
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
                return

            st
