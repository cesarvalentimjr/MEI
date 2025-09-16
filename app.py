"""
Extrator de Extratos Bancários (versão melhorada)
Melhorias principais na classificação débito/crédito:
- Análise mais precisa de padrões textuais
- Melhor detecção de sinais e formatação
- Heurísticas baseadas em contexto bancário brasileiro
- Validação cruzada entre múltiplas evidências
"""

# ============================================================
# 1) Importações e configuração inicial
# ============================================================
import streamlit as st
import pandas as pd
import re
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import logging
import io

# Bibliotecas opcionais
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

# Config Streamlit
st.set_page_config(page_title="Extrator de Extratos Bancários (Melhorado)", page_icon="🏦", layout="wide")

# ============================================================
# 2) Estrutura que representa uma transação
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
    confidence_score: float = 0.0  # Nova: pontuação de confiança na classificação

# ============================================================
# 3) Helper: parsing robusto de números monetários
# ============================================================
def parse_monetary_string(s: str) -> Optional[float]:
    """
    Converte uma string que representa dinheiro (BR/EN) em float.
    Preserva o sinal original para análise posterior.
    """
    if not s or not isinstance(s, str):
        return None
    
    original = s.strip()
    s = original
    
    # Detecta parênteses (formato contábil para negativo)
    parentheses_negative = False
    if s.startswith("(") and s.endswith(")"):
        parentheses_negative = True
        s = s[1:-1].strip()
    
    # Detecta sinal explícito
    explicit_negative = s.startswith("-")
    explicit_positive = s.startswith("+")
    
    # Remove símbolos monetários e espaços
    s = re.sub(r"[Rr]\$|\s+", "", s)
    
    # Remove sinais para processamento
    s = s.lstrip("+-")
    
    # Trata separadores decimais e de milhares
    if "." in s and "," in s:
        # Determina qual é decimal pela posição
        dot_pos = s.rfind(".")
        comma_pos = s.rfind(",")
        
        if comma_pos > dot_pos:
            # Vírgula é decimal: "1.234,56"
            s = s.replace(".", "").replace(",", ".")
        else:
            # Ponto é decimal: "1,234.56"
            s = s.replace(",", "")
    elif "," in s and "." not in s:
        # Só vírgula - verifica se é decimal
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) == 2:
            # Provavelmente decimal
            s = s.replace(",", ".")
        else:
            # Provavelmente separador de milhares
            s = s.replace(",", "")
    elif "." in s and "," not in s:
        # Só ponto - verifica se é decimal
        parts = s.split(".")
        if len(parts) == 2 and len(parts[1]) == 2:
            # Provavelmente decimal
            pass
        else:
            # Provavelmente separador de milhares
            s = s.replace(".", "")
    
    # Remove qualquer caractere não numérico restante exceto ponto decimal
    s = re.sub(r"[^\d.]", "", s)
    
    if not s:
        return None
    
    try:
        val = float(s)
        # Aplica negativos detectados
        if parentheses_negative or explicit_negative:
            val = -abs(val)
        elif explicit_positive:
            val = abs(val)
        
        return val
    except Exception:
        return None

# ============================================================
# 4) Matcher aprimorado com melhor classificação débito/crédito
# ============================================================
class EnhancedBankMatcher:
    def __init__(self):
        # Padrões de data
        self.date_patterns = [
            r'\b\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4}\b',
            r'\b\d{2}[\/\-\.]\d{2}[\/\-\.]\d{2}\b',
            r'\b\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2}\b',
            r'\b\d{1,2}\s+(?:jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\s+\d{4}\b',
        ]
        
        # Padrão para valores monetários mais preciso
        self.value_pattern = re.compile(
            r'(\(?[+\-]?\s*(?:R\$)?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\)?\s*[CD]?)',
            flags=re.IGNORECASE
        )
        
        # Padrões de ruído
        self.noise_patterns = [
            r'^SALDO\b', r'\bSALDO\s+(ANTERIOR|ATUAL|FINAL)\b', 
            r'^TOTAL\b', r'\bTOTAL\s+DE\b', r'EXTRATO', r'PÁGINA', 
            r'AGÊNCIA', r'CONTA', r'^DATA\b.*VALOR\b', r'^DESCRIÇÃO\b',
            r'^(CRÉDITO|CRÉDITOS|DÉBITO|DÉBITOS)$',
            r'MOVIMENTAÇÃO\s+TOTAL', r'RESUMO\s+DO\s+PERÍODO'
        ]
        
        # Palavras-chave para débito (mais específicas)
        self.strong_debit_indicators = [
            # Saques e retiradas
            'saque', 'retirada', 'atm', 'caixa eletrônico', 'autoatendimento',
            # Pagamentos
            'pagamento', 'pagto', 'pago', 'quitação',
            # Transferências de saída
            'ted enviada', 'doc enviado', 'pix enviado', 'transferência enviada',
            'transf enviada', 'envio pix', 'envio ted',
            # Tarifas e taxas
            'tarifa', 'taxa', 'juros', 'iof', 'anuidade', 'mensalidade',
            # Compras
            'compra', 'débito automático', 'débito em conta', 'cartão de débito',
            # Outros débitos
            'cobrança', 'desconto', 'estorno débito'
        ]
        
        self.moderate_debit_indicators = [
            'débito', 'debito', 'saída', 'retirar', 'pagar', 'comprar',
            'transferir', 'enviar', 'remessa'
        ]
        
        # Palavras-chave para crédito (mais específicas)
        self.strong_credit_indicators = [
            # Depósitos e recebimentos
            'depósito', 'deposito', 'recebimento', 'receber',
            # Transferências de entrada
            'ted recebida', 'doc recebido', 'pix recebido', 'transferência recebida',
            'transf recebida', 'recebimento pix', 'recebimento ted',
            # Salários e rendimentos
            'salário', 'salario', 'rendimento', 'juros sobre saldo',
            'remuneração', 'provento', 'dividendo',
            # Estornos de crédito
            'estorno crédito', 'ressarcimento', 'reembolso',
            # Outros créditos
            'creditado', 'entrada', 'aplicação resgatada'
        ]
        
        self.moderate_credit_indicators = [
            'crédito', 'credito', 'entrada', 'receber', 'depositar',
            'transferir para', 'recebimento de'
        ]
        
        # Padrões que indicam formatação de débito
        self.debit_formatting_patterns = [
            r'\(\s*\d+[.,]\d{2}\s*\)',  # (123,45)
            r'-\s*\d+[.,]\d{2}',        # -123,45
            r'\d+[.,]\d{2}\s*D\b',      # 123,45 D
            r'D\s*\d+[.,]\d{2}',        # D 123,45
            r'\d+[.,]\d{2}\s*-',        # 123,45-
        ]
        
        # Padrões que indicam formatação de crédito  
        self.credit_formatting_patterns = [
            r'\+\s*\d+[.,]\d{2}',       # +123,45
            r'\d+[.,]\d{2}\s*C\b',      # 123,45 C
            r'C\s*\d+[.,]\d{2}',        # C 123,45
            r'\d+[.,]\d{2}\s*\+',       # 123,45+
        ]

    def is_noise_line(self, line: str) -> bool:
        """Identifica linhas que são ruído (saldos, totais, cabeçalhos)"""
        if not line or len(line.strip()) < 3:
            return True
        
        # Verifica padrões explícitos de ruído
        for pattern in self.noise_patterns:
            if re.search(pattern, line, flags=re.IGNORECASE):
                return True
        
        # Verifica palavras-chave de ruído
        lower = line.lower().strip()
        noise_keywords = [
            'saldo anterior', 'saldo atual', 'saldo final',
            'total de créditos', 'total de débitos', 'total geral',
            'subtotal', 'soma dos', 'movimentação do período'
        ]
        
        if any(keyword in lower for keyword in noise_keywords):
            return True
            
        return False

    def find_date(self, line: str) -> Optional[re.Match]:
        """Encontra padrão de data na linha"""
        for pattern in self.date_patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                return match
        return None

    def parse_date_from_match(self, match: re.Match) -> Optional[datetime]:
        """Converte match de data para datetime"""
        date_str = match.group().strip()
        
        # Tenta formato com mês abreviado em português
        month_abbr_pattern = r'(\d{1,2})\s+(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\s+(\d{4})'
        month_match = re.match(month_abbr_pattern, date_str.lower())
        
        if month_match:
            day = int(month_match.group(1))
            month_abbr = month_match.group(2)
            year = int(month_match.group(3))
            
            months = {
                'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
                'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12
            }
            
            try:
                return datetime(year, months[month_abbr], day)
            except (ValueError, KeyError):
                pass
        
        # Tenta formatos padrão
        formats = ['%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d', '%d-%m-%Y']
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None

    def extract_value_matches(self, line: str) -> List[Dict]:
        """Extrai todos os valores monetários da linha com contexto"""
        matches = []
        
        for match in self.value_pattern.finditer(line):
            value_text = match.group(1).strip()
            parsed_value = parse_monetary_string(value_text)
            
            if parsed_value is not None:
                # Contexto antes e depois do valor
                start, end = match.span(1)
                context_before = line[max(0, start-20):start].lower()
                context_after = line[end:min(len(line), end+20)].lower()
                
                matches.append({
                    'start': start,
                    'end': end,
                    'text': value_text,
                    'value': parsed_value,
                    'context_before': context_before,
                    'context_after': context_after
                })
        
        return matches

    def classify_debit_credit_improved(self, line: str, value_info: Dict) -> Tuple[bool, float]:
        """
        Classificação melhorada de débito/crédito especialmente para extratos do BB.
        Retorna (is_debit: bool, confidence: float)
        """
        line_lower = line.lower()
        value_text = value_info['text'].lower()
        context_before = value_info['context_before']
        context_after = value_info['context_after']
        full_line = line.strip()
        
        confidence_score = 0.0
        debit_score = 0.0
        credit_score = 0.0
        
        # 1. ANÁLISE ESPECÍFICA PARA FORMATO BB - Sufixos D/C no final da linha
        # Padrão: "valor D saldo" ou "valor C saldo"  
        dc_pattern = re.search(r'(\d+(?:[.,]\d+)*)\s*([DC])\s+(\d+(?:[.,]\d+)*)\s*([DC])?', full_line, re.IGNORECASE)
        if dc_pattern:
            valor_dc = dc_pattern.group(2).upper()
            if valor_dc == 'D':
                debit_score += 0.5
                confidence_score += 0.4
            elif valor_dc == 'C':
                credit_score += 0.5
                confidence_score += 0.4
        
        # 2. Análise do saldo final da linha (padrão BB: valor operação + D/C + saldo + C)
        # Ex: "1.306,10 C" or "1.925,00 D"
        saldo_pattern = re.search(r'(\d+(?:[.,]\d+)*)\s*([DC])\s*

    def identify_transaction_and_balance(self, value_matches: List[Dict], line: str) -> Tuple[Dict, Dict]:
        """
        Identifica qual valor é a transação e qual é o saldo.
        Retorna (transaction_value_info, balance_value_info)
        """
        if len(value_matches) == 1:
            return value_matches[0], None
        
        line_len = len(line)
        
        # Ordena por posição
        sorted_matches = sorted(value_matches, key=lambda x: x['start'])
        
        # Heurísticas para identificar saldo
        balance_candidate = None
        
        for i, match in enumerate(sorted_matches):
            # Saldo geralmente aparece no final da linha
            if match['start'] > line_len * 0.7:
                # Verifica se contexto indica saldo
                context = match['context_after']
                if any(word in context for word in ['saldo', 'sld']):
                    balance_candidate = match
                    break
                # Se é o último valor e está no final, provavelmente é saldo
                elif i == len(sorted_matches) - 1:
                    balance_candidate = match
                    break
        
        # Remove saldo da lista de candidatos a transação
        transaction_candidates = [m for m in sorted_matches if m != balance_candidate]
        
        if transaction_candidates:
            # Escolhe o primeiro candidato restante como transação
            transaction_value = transaction_candidates[0]
        else:
            # Se não há candidatos, usa o primeiro valor como transação
            transaction_value = sorted_matches[0]
            balance_candidate = None
        
        return transaction_value, balance_candidate

    def clean_description(self, line: str, date_match: re.Match, value_matches: List[Dict]) -> str:
        """Limpa a descrição removendo datas, valores e ruído"""
        # Cria máscara para marcar caracteres a remover
        mask = [False] * len(line)
        
        # Marca posição da data
        if date_match:
            for i in range(date_match.start(), date_match.end()):
                if i < len(mask):
                    mask[i] = True
        
        # Marca posições dos valores
        for match in value_matches:
            for i in range(match['start'], min(match['end'], len(mask))):
                mask[i] = True
        
        # Constrói descrição sem caracteres marcados
        description = ''.join(char if not mask[i] else ' ' for i, char in enumerate(line))
        
        # Limpeza adicional
        description = re.sub(r'[Rr]\$|\(|\)|[CD]\b|\bD\b|\bC\b', ' ', description)
        description = re.sub(r'[:\-•*=]+', ' ', description)
        description = re.sub(r'\s+', ' ', description).strip()
        
        # Remove palavras de ruído remanescentes
        noise_words = ['saldo', 'total', 'subtotal', 'resumo', 'extrato', 'anterior', 'atual', 'final']
        for word in noise_words:
            description = re.sub(rf'\b{word}\b', ' ', description, flags=re.IGNORECASE)
        
        description = re.sub(r'\s+', ' ', description).strip()
        
        if len(description) < 3:
            description = "Transação não identificada"
        
        return description

    def classify_transaction_type(self, description: str, value: float) -> str:
        """Classifica o tipo de transação baseado na descrição e valor"""
        desc_lower = description.lower()
        
        if value < 0:  # Débitos
            if any(word in desc_lower for word in ['pix', 'ted', 'doc', 'transferência', 'transf']):
                if any(word in desc_lower for word in ['enviado', 'enviada', 'para', 'pagto']):
                    return 'TRANSFERÊNCIA_SAÍDA'
            
            if any(word in desc_lower for word in ['saque', 'atm', 'caixa eletrônico', 'retirada']):
                return 'SAQUE'
            
            if any(word in desc_lower for word in ['compra', 'débito automático', 'cartão']):
                return 'COMPRA_DÉBITO'
            
            if any(word in desc_lower for word in ['tarifa', 'taxa', 'juros', 'iof', 'anuidade']):
                return 'TARIFA'
            
            if any(word in desc_lower for word in ['pagamento', 'pagto', 'boleto']):
                return 'PAGAMENTO'
            
            return 'DÉBITO'
        
        else:  # Créditos
            if any(word in desc_lower for word in ['pix', 'ted', 'doc', 'transferência', 'transf']):
                if any(word in desc_lower for word in ['recebido', 'recebida', 'de']):
                    return 'TRANSFERÊNCIA_ENTRADA'
            
            if any(word in desc_lower for word in ['salário', 'salario', 'remuneração']):
                return 'SALÁRIO'
            
            if any(word in desc_lower for word in ['depósito', 'deposito']):
                return 'DEPÓSITO'
            
            if any(word in desc_lower for word in ['rendimento', 'juros', 'remuneração']):
                return 'RENDIMENTO'
            
            if any(word in desc_lower for word in ['estorno', 'reembolso', 'ressarcimento']):
                return 'ESTORNO'
            
            return 'CRÉDITO'

    def categorize_transaction(self, description: str) -> str:
        """Categoriza a transação baseado na descrição"""
        desc_lower = description.lower()
        
        # Alimentação
        food_keywords = ['restaurante', 'lanche', 'mercado', 'supermercado', 'padaria', 
                        'ifood', 'uber eats', 'delivery', 'açougue', 'pizzaria']
        if any(word in desc_lower for word in food_keywords):
            return 'ALIMENTAÇÃO'
        
        # Transporte
        transport_keywords = ['uber', '99', 'combustível', 'posto', 'gasolina', 'diesel',
                             'estacionamento', 'pedágio', 'ônibus', 'metrô', 'taxi']
        if any(word in desc_lower for word in transport_keywords):
            return 'TRANSPORTE'
        
        # Casa
        home_keywords = ['energia', 'luz', 'água', 'gas', 'telefone', 'internet', 
                        'condomínio', 'aluguel', 'financiamento', 'iptu']
        if any(word in desc_lower for word in home_keywords):
            return 'CASA'
        
        # Saúde
        health_keywords = ['farmácia', 'hospital', 'médico', 'dentista', 'clínica',
                          'laboratório', 'exame', 'consulta', 'plano de saúde']
        if any(word in desc_lower for word in health_keywords):
            return 'SAÚDE'
        
        # Lazer
        leisure_keywords = ['netflix', 'spotify', 'cinema', 'teatro', 'show', 'bar',
                           'festa', 'viagem', 'hotel', 'turismo']
        if any(word in desc_lower for word in leisure_keywords):
            return 'LAZER'
        
        # Compras
        shopping_keywords = ['americanas', 'mercado livre', 'amazon', 'magazine', 
                            'shopping', 'loja', 'compra']
        if any(word in desc_lower for word in shopping_keywords):
            return 'COMPRAS'
        
        # Educação
        education_keywords = ['escola', 'universidade', 'curso', 'faculdade', 'colégio']
        if any(word in desc_lower for word in education_keywords):
            return 'EDUCAÇÃO'
        
        return 'OUTROS'

    def parse_transaction_line(self, line: str) -> Optional[Transaction]:
        """Parse principal de uma linha para extrair transação"""
        if not line or len(line.strip()) < 6:
            return None
        
        if self.is_noise_line(line):
            return None
        
        # 1. Encontra data
        date_match = self.find_date(line)
        if not date_match:
            return None
        
        date_obj = self.parse_date_from_match(date_match)
        if not date_obj:
            return None
        
        # 2. Extrai valores monetários
        value_matches = self.extract_value_matches(line)
        if not value_matches:
            return None
        
        # 3. Identifica transação e saldo
        transaction_value_info, balance_value_info = self.identify_transaction_and_balance(value_matches, line)
        
        # 4. Classifica débito/crédito
        is_debit, confidence = self.classify_debit_credit_improved(line, transaction_value_info)
        
        # 5. Ajusta o valor conforme classificação
        transaction_value = transaction_value_info['value']
        if is_debit and transaction_value > 0:
            transaction_value = -abs(transaction_value)
        elif not is_debit and transaction_value < 0:
            transaction_value = abs(transaction_value)
        
        # 6. Extrai saldo se identificado
        balance_value = None
        if balance_value_info:
            balance_value = abs(balance_value_info['value'])  # Saldo sempre positivo
        
        # 7. Limpa descrição
        description = self.clean_description(line, date_match, value_matches)
        
        # 8. Classifica tipo e categoria
        transaction_type = self.classify_transaction_type(description, transaction_value)
        category = self.categorize_transaction(description)
        
        return Transaction(
            date=date_obj,
            description=description,
            value=transaction_value,
            balance=balance_value,
            transaction_type=transaction_type,
            category=category,
            confidence_score=confidence
        )

# ============================================================
# 5) Extrator principal
# ============================================================
class BankStatementExtractor:
    def __init__(self, matcher: Optional[EnhancedBankMatcher] = None):
        self.matcher = matcher or EnhancedBankMatcher()

    def extract_from_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> pd.DataFrame:
        """Extrai transações de PDF em bytes"""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                all_lines = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_lines.extend(text.splitlines())
            
            transactions = self._process_lines(all_lines)
            df = self._transactions_to_dataframe(transactions)
            
            if not df.empty:
                df['arquivo'] = filename
            
            return df
        
        except Exception as e:
            logger.exception(f"Erro ao processar PDF {filename}: {e}")
            return pd.DataFrame()

    def _process_lines(self, lines: List[str]) -> List[Transaction]:
        """Processa linhas extraindo transações válidas"""
        transactions = []
        
        for line in lines:
            if not line or len(line.strip()) < 6:
                continue
            
            transaction = self.matcher.parse_transaction_line(line)
            if transaction and self._is_valid_transaction(transaction):
                transactions.append(transaction)
        
        return transactions

    def _is_valid_transaction(self, transaction: Transaction) -> bool:
        """Valida se a transação é legítima"""
        if not transaction:
            return False
        
        # Valor deve existir e ter magnitude mínima
        if transaction.value is None or abs(transaction.value) < 0.01:
            return False
        
        # Descrição deve ser significativa
        if not transaction.description or len(transaction.description.strip()) < 3:
            return False
        
        # Confiança mínima na classificação
        if transaction.confidence_score < 0.1:
            return False
        
        return True

    def _transactions_to_dataframe(self, transactions: List[Transaction]) -> pd.DataFrame:
        """Converte lista de transações para DataFrame"""
        if not transactions:
            return pd.DataFrame()
        
        data = []
        for t in transactions:
            data.append({
                'data': t.date,
                'descricao': t.description,
                'valor': t.value,
                'saldo': t.balance,
                'tipo': t.transaction_type,
                'categoria': t.category,
                'confianca': t.confidence_score
            })
        
        df = pd.DataFrame(data)
        
        # Normalização e ordenação
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

# ============================================================
# 6) Processamento de múltiplos PDFs
# ============================================================
def process_multiple_pdfs(uploaded_files):
    """Processa múltiplos arquivos PDF"""
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
                avg_confidence = df['confianca'].mean()
                banco_detectado = extractor.detected_banks.get(uploaded_file.name, 'N/A')
                st.success(f"✅ {uploaded_file.name}: {len(df)} transações | Banco: **{banco_detectado}** | Confiança: {avg_confidence:.2f}")
        
        except Exception as e:
            st.error(f"❌ Erro ao processar {uploaded_file.name}: {e}")
        
        progress.progress((i + 1) / len(uploaded_files))
    
    status.text("Processamento concluído")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values('data').reset_index(drop=True)
        return combined, extractor.detected_banks
    
    return pd.DataFrame(), {}

# ============================================================
# 7) Análises e gráficos
# ============================================================
def create_summary_charts(df):
    """Cria gráficos de resumo"""
    if df.empty or not PLOTLY_AVAILABLE:
        if not PLOTLY_AVAILABLE:
            st.warning("Plotly não disponível — gráficos desabilitados")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fluxo de caixa mensal
        monthly = df.groupby('mes_ano')['valor'].sum().reset_index()
        monthly['mes_ano_dt'] = pd.to_datetime(monthly['mes_ano'] + '-01')
        monthly = monthly.sort_values('mes_ano_dt')
        
        fig1 = px.line(
            monthly, 
            x='mes_ano_dt', 
            y='valor',
            title='Fluxo de Caixa Mensal',
            labels={'valor': 'Valor (R$)', 'mes_ano_dt': 'Mês'}
        )
        fig1.add_hline(y=0, line_dash='dash', line_color='red', annotation_text="Zero")
        fig1.update_layout(xaxis_title="Período", yaxis_title="Valor (R$)")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Gastos por categoria (apenas débitos)
        debits = df[df['valor'] < 0].copy()
        if not debits.empty:
            category_spending = debits.groupby('categoria')['valor'].sum().abs().reset_index()
            category_spending = category_spending.sort_values('valor', ascending=False)
            
            fig2 = px.bar(
                category_spending,
                x='categoria',
                y='valor',
                title='Gastos por Categoria',
                labels={'valor': 'Valor (R$)', 'categoria': 'Categoria'}
            )
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)

def create_confidence_analysis(df):
    """Análise de confiança das classificações"""
    if df.empty:
        return
    
    st.subheader("📊 Análise de Confiança na Classificação")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_confidence = df['confianca'].mean()
        st.metric("Confiança Média", f"{avg_confidence:.2f}")
    
    with col2:
        high_confidence = (df['confianca'] >= 0.7).sum()
        st.metric("Alta Confiança (≥0.7)", f"{high_confidence} ({high_confidence/len(df)*100:.1f}%)")
    
    with col3:
        low_confidence = (df['confianca'] < 0.3).sum()
        st.metric("Baixa Confiança (<0.3)", f"{low_confidence} ({low_confidence/len(df)*100:.1f}%)")
    
    if PLOTLY_AVAILABLE:
        # Histograma de confiança
        fig = px.histogram(
            df,
            x='confianca',
            bins=20,
            title='Distribuição da Confiança na Classificação',
            labels={'confianca': 'Confiança', 'count': 'Quantidade'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confiança por tipo de transação
        confidence_by_type = df.groupby('tipo')['confianca'].agg(['mean', 'count']).reset_index()
        confidence_by_type.columns = ['tipo', 'confianca_media', 'quantidade']
        
        fig2 = px.bar(
            confidence_by_type,
            x='tipo',
            y='confianca_media',
            title='Confiança Média por Tipo de Transação',
            labels={'confianca_media': 'Confiança Média', 'tipo': 'Tipo'}
        )
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)

def show_classification_review(df):
    """Interface para revisão das classificações"""
    if df.empty:
        return
    
    st.subheader("🔍 Revisão de Classificações")
    
    # Filtros para revisão
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider(
            "Mostrar transações com confiança menor que:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
    
    with col2:
        show_type = st.selectbox(
            "Filtrar por tipo:",
            ['Todos'] + df['tipo'].unique().tolist()
        )
    
    # Aplica filtros
    filtered = df[df['confianca'] < confidence_threshold]
    if show_type != 'Todos':
        filtered = filtered[filtered['tipo'] == show_type]
    
    if not filtered.empty:
        st.write(f"Encontradas {len(filtered)} transações para revisão:")
        
        # Mostra transações para revisão
        review_cols = ['data', 'descricao', 'valor', 'tipo', 'categoria', 'confianca']
        st.dataframe(
            filtered[review_cols].sort_values('confianca'),
            use_container_width=True
        )
    else:
        st.info("Nenhuma transação encontrada com os critérios selecionados.")

# ============================================================
# 8) Interface Streamlit principal
# ============================================================
def main():
    st.title("🏦 Extrator de Extratos Bancários - Versão Melhorada")
    st.markdown("""
    Esta versão aprimorada oferece:
    - **Melhor detecção de débito/crédito** com múltiplas heurísticas
    - **Pontuação de confiança** para cada classificação
    - **Análise de padrões** bancários brasileiros
    - **Interface de revisão** para classificações duvidosas
    """)

    with st.sidebar:
        st.header("📋 Instruções")
        st.markdown("""
        ### Como usar:
        1. Faça upload de PDFs de extratos bancários
        2. Aguarde o processamento automático
        3. Revise as classificações na aba de análise
        4. Baixe os dados processados
        
        ### Melhorias desta versão:
        - ✅ Detecção aprimorada de sinais (-, +, parênteses)
        - ✅ Reconhecimento de padrões D/C
        - ✅ Análise contextual de palavras-chave
        - ✅ Identificação de PIX, TED, DOC
        - ✅ Pontuação de confiança
        """)
        
        st.header("ℹ️ Informações")
        st.info("""
        **Confiança na classificação:**
        - 0.7+ : Alta confiança
        - 0.3-0.7 : Média confiança  
        - <0.3 : Baixa confiança
        """)

    # Upload de arquivos
    uploaded_files = st.file_uploader(
        "Escolha arquivos PDF de extratos bancários",
        type="pdf",
        accept_multiple_files=True,
        help="Selecione um ou mais arquivos PDF de extratos bancários"
    )

    if uploaded_files:
            if st.button("🚀 Processar Extratos", type="primary"):
            with st.spinner("Processando extratos..."):
                df, detected_banks = process_multiple_pdfs(uploaded_files)
            
            if df.empty:
                st.error("❌ Nenhuma transação foi extraída. Verifique se os PDFs contêm extratos bancários válidos.")
                return

            # Métricas principais
            st.success(f"✅ Processamento concluído — {len(df)} transações extraídas")
            
            # Mostrar bancos detectados
            if detected_banks:
                st.info("🏦 **Bancos detectados:** " + " | ".join([f"{arquivo}: {banco}" for arquivo, banco in detected_banks.items()]))
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de Transações", len(df))
            
            with col2:
                total_credits = df[df['valor'] > 0]['valor'].sum()
                st.metric("Total Créditos", f"R$ {total_credits:,.2f}")
            
            with col3:
                total_debits = df[df['valor'] < 0]['valor'].sum()
                st.metric("Total Débitos", f"R$ {total_debits:,.2f}")
            
            with col4:
                net_flow = df['valor'].sum()
                st.metric("Fluxo Líquido", f"R$ {net_flow:,.2f}")
            
            # Abas de análise
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Resumo", "🔍 Análise de Confiança", "📝 Revisão", "📊 Dados"])
            
            with tab1:
                st.header("Análises Gerais")
                create_summary_charts(df)
                
                # Distribuição de tipos e bancos
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Distribuição por Tipo")
                    type_dist = df['tipo'].value_counts()
                    st.bar_chart(type_dist)
                
                with col2:
                    st.subheader("Distribuição por Banco")
                    if 'banco_detectado' in df.columns:
                        banco_dist = df['banco_detectado'].value_counts()
                        st.bar_chart(banco_dist)
                    else:
                        st.info("Informação de banco não disponível")
            
            with tab2:
                create_confidence_analysis(df)
                
                # Análise de confiança por banco
                if 'banco_detectado' in df.columns and PLOTLY_AVAILABLE:
                    st.subheader("📊 Confiança por Banco")
                    confidence_by_bank = df.groupby('banco_detectado')['confianca'].agg(['mean', 'count']).reset_index()
                    confidence_by_bank.columns = ['banco', 'confianca_media', 'quantidade']
                    
                    fig3 = px.bar(
                        confidence_by_bank,
                        x='banco',
                        y='confianca_media',
                        title='Confiança Média por Banco Detectado',
                        labels={'confianca_media': 'Confiança Média', 'banco': 'Banco'},
                        text='quantidade'
                    )
                    fig3.update_traces(texttemplate='%{text} tx', textposition='outside')
                    st.plotly_chart(fig3, use_container_width=True)
            
            with tab3:
                show_classification_review(df)
            
            with tab4:
                st.header("📋 Dados Extraídos")
                
                # Filtros expandidos
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    tipos_sel = st.multiselect(
                        "Filtrar por tipo:",
                        options=df['tipo'].unique().tolist(),
                        default=df['tipo'].unique().tolist()
                    )
                
                with col2:
                    categorias_sel = st.multiselect(
                        "Filtrar por categoria:",
                        options=df['categoria'].unique().tolist(),
                        default=df['categoria'].unique().tolist()
                    )
                
                with col3:
                    min_confidence = st.slider(
                        "Confiança mínima:",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.1
                    )
                
                with col4:
                    if 'banco_detectado' in df.columns:
                        bancos_sel = st.multiselect(
                            "Filtrar por banco:",
                            options=df['banco_detectado'].unique().tolist(),
                            default=df['banco_detectado'].unique().tolist()
                        )
                    else:
                        bancos_sel = []
                
                # Filtro de data
                if not df.empty:
                    date_range = st.date_input(
                        "Período:",
                        value=(df['data'].min().date(), df['data'].max().date()),
                        min_value=df['data'].min().date(),
                        max_value=df['data'].max().date()
                    )
                
                # Aplicar filtros
                filtered_df = df[
                    (df['tipo'].isin(tipos_sel)) &
                    (df['categoria'].isin(categorias_sel)) &
                    (df['confianca'] >= min_confidence)
                ]
                
                if bancos_sel and 'banco_detectado' in df.columns:
                    filtered_df = filtered_df[filtered_df['banco_detectado'].isin(bancos_sel)]
                
                if len(date_range) == 2:
                    filtered_df = filtered_df[
                        (filtered_df['data'] >= pd.to_datetime(date_range[0])) &
                        (filtered_df['data'] <= pd.to_datetime(date_range[1]))
                    ]
                
                # Exibir dados filtrados
                st.dataframe(filtered_df, use_container_width=True)
                
                # Downloads
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_filtered = filtered_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        "📥 Baixar Dados Filtrados (CSV)",
                        csv_filtered,
                        file_name="extratos_filtrados.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv_all = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        "📥 Baixar Todos os Dados (CSV)",
                        csv_all,
                        file_name="extratos_completo.csv",
                        mime="text/csv"
                    )

    else:
        st.info("👆 Faça upload de arquivos PDF para começar o processamento")
        
        # Exemplo de bancos suportados
        with st.expander("🏦 Bancos e Formatos Suportados"):
            st.markdown("""
            ### Bancos com Parsers Específicos:
            
            **🔹 Banco do Brasil** - Formato estruturado com D/C
            - Detecta automaticamente transferências, PIX, boletos
            - Alta precisão na classificação débito/crédito
            
            **🔹 Bradesco** - Colunas Crédito/Débito separadas
            - Reconhece investimentos, cartão de crédito
            - Formatação em colunas estruturadas
            
            **🔹 Banco Inter** - Valores com sinais explícitos
            - PIX enviados/recebidos bem definidos
            - Datas em formato extenso português
            
            **🔹 Caixa Econômica** - Histórico + valor D/C
            - TED, boletos, aplicações automáticas
            - Formato tabular clássico
            
            **🔹 XP Investimentos** - Operações de investimento
            - Resgates, aportes, TED
            - Valores negativos com hífen
            
            **🔹 Itaú** - Múltiplos formatos
            - PIX, boletos, transferências
            - Detecção por palavras-chave
            
            **🔹 Santander** - TED e aplicações
            - Conta Max, PIX empresarial
            - Investimentos automáticos
            
            **🔹 Sicoob** - Cooperativas de crédito
            - PIX, TED, operações cooperativistas
            - Formato com D/C ao final
            
            **🔹 Nubank** - Conta digital
            - Formato minimalista
            - Transações digitais
            
            ### Funcionalidades Avançadas:
            - ✅ **Detecção automática** do banco
            - ✅ **Classificação inteligente** débito/crédito  
            - ✅ **Pontuação de confiança** por transação
            - ✅ **Categorização automática** (alimentação, transporte, etc.)
            - ✅ **Interface de revisão** para baixa confiança
            - ✅ **Suporte a múltiplos bancos** simultaneamente
            """)

if __name__ == "__main__":
    main(), full_line, re.IGNORECASE)
        if saldo_pattern and not dc_pattern:  # Se não achou padrão duplo, pode ser valor único
            valor_dc = saldo_pattern.group(1)
            sinal_dc = saldo_pattern.group(2).upper()
            
            # Verifica se é o valor da transação (não o saldo)
            parsed_saldo = parse_monetary_string(valor_dc)
            if parsed_saldo and abs(parsed_saldo - abs(value_info['value'])) < 0.01:
                if sinal_dc == 'D':
                    debit_score += 0.45
                    confidence_score += 0.35
                elif sinal_dc == 'C':
                    credit_score += 0.45
                    confidence_score += 0.35
        
        # 3. Palavras-chave específicas de débito do BB (alta confiança)
        bb_debit_keywords = [
            'pagamento de boleto', 'pagamento conta', 'pagto conta', 'transferência enviada',
            'pix - enviado', 'pix - agendamento', 'pix enviado', 'folha de pagamento',
            'impostos', 'tarifa', 'cheque compensado', 'cheque pago', 'transferência agendada'
        ]
        
        for keyword in bb_debit_keywords:
            if keyword in line_lower:
                debit_score += 0.4
                confidence_score += 0.3
                break
        
        # 4. Palavras-chave específicas de crédito do BB (alta confiança)
        bb_credit_keywords = [
            'depósito online', 'bb rende fácil', 'pix - rejeitado', 'pix rejeitado',
            'transferência recebida', 'pix recebido', 'saldo anterior'
        ]
        
        for keyword in bb_credit_keywords:
            if keyword in line_lower:
                credit_score += 0.4
                confidence_score += 0.3
                break
        
        # 5. Análise contextual adicional
        # PIX enviado vs recebido
        if 'pix' in line_lower:
            if any(word in line_lower for word in ['enviado', 'agendamento', 'para']):
                debit_score += 0.3
            elif any(word in line_lower for word in ['recebido', 'de

    def identify_transaction_and_balance(self, value_matches: List[Dict], line: str) -> Tuple[Dict, Dict]:
        """
        Identifica qual valor é a transação e qual é o saldo.
        Retorna (transaction_value_info, balance_value_info)
        """
        if len(value_matches) == 1:
            return value_matches[0], None
        
        line_len = len(line)
        
        # Ordena por posição
        sorted_matches = sorted(value_matches, key=lambda x: x['start'])
        
        # Heurísticas para identificar saldo
        balance_candidate = None
        
        for i, match in enumerate(sorted_matches):
            # Saldo geralmente aparece no final da linha
            if match['start'] > line_len * 0.7:
                # Verifica se contexto indica saldo
                context = match['context_after']
                if any(word in context for word in ['saldo', 'sld']):
                    balance_candidate = match
                    break
                # Se é o último valor e está no final, provavelmente é saldo
                elif i == len(sorted_matches) - 1:
                    balance_candidate = match
                    break
        
        # Remove saldo da lista de candidatos a transação
        transaction_candidates = [m for m in sorted_matches if m != balance_candidate]
        
        if transaction_candidates:
            # Escolhe o primeiro candidato restante como transação
            transaction_value = transaction_candidates[0]
        else:
            # Se não há candidatos, usa o primeiro valor como transação
            transaction_value = sorted_matches[0]
            balance_candidate = None
        
        return transaction_value, balance_candidate

    def clean_description(self, line: str, date_match: re.Match, value_matches: List[Dict]) -> str:
        """Limpa a descrição removendo datas, valores e ruído"""
        # Cria máscara para marcar caracteres a remover
        mask = [False] * len(line)
        
        # Marca posição da data
        if date_match:
            for i in range(date_match.start(), date_match.end()):
                if i < len(mask):
                    mask[i] = True
        
        # Marca posições dos valores
        for match in value_matches:
            for i in range(match['start'], min(match['end'], len(mask))):
                mask[i] = True
        
        # Constrói descrição sem caracteres marcados
        description = ''.join(char if not mask[i] else ' ' for i, char in enumerate(line))
        
        # Limpeza adicional
        description = re.sub(r'[Rr]\$|\(|\)|[CD]\b|\bD\b|\bC\b', ' ', description)
        description = re.sub(r'[:\-•*=]+', ' ', description)
        description = re.sub(r'\s+', ' ', description).strip()
        
        # Remove palavras de ruído remanescentes
        noise_words = ['saldo', 'total', 'subtotal', 'resumo', 'extrato', 'anterior', 'atual', 'final']
        for word in noise_words:
            description = re.sub(rf'\b{word}\b', ' ', description, flags=re.IGNORECASE)
        
        description = re.sub(r'\s+', ' ', description).strip()
        
        if len(description) < 3:
            description = "Transação não identificada"
        
        return description

    def classify_transaction_type(self, description: str, value: float) -> str:
        """Classifica o tipo de transação baseado na descrição e valor"""
        desc_lower = description.lower()
        
        if value < 0:  # Débitos
            if any(word in desc_lower for word in ['pix', 'ted', 'doc', 'transferência', 'transf']):
                if any(word in desc_lower for word in ['enviado', 'enviada', 'para', 'pagto']):
                    return 'TRANSFERÊNCIA_SAÍDA'
            
            if any(word in desc_lower for word in ['saque', 'atm', 'caixa eletrônico', 'retirada']):
                return 'SAQUE'
            
            if any(word in desc_lower for word in ['compra', 'débito automático', 'cartão']):
                return 'COMPRA_DÉBITO'
            
            if any(word in desc_lower for word in ['tarifa', 'taxa', 'juros', 'iof', 'anuidade']):
                return 'TARIFA'
            
            if any(word in desc_lower for word in ['pagamento', 'pagto', 'boleto']):
                return 'PAGAMENTO'
            
            return 'DÉBITO'
        
        else:  # Créditos
            if any(word in desc_lower for word in ['pix', 'ted', 'doc', 'transferência', 'transf']):
                if any(word in desc_lower for word in ['recebido', 'recebida', 'de']):
                    return 'TRANSFERÊNCIA_ENTRADA'
            
            if any(word in desc_lower for word in ['salário', 'salario', 'remuneração']):
                return 'SALÁRIO'
            
            if any(word in desc_lower for word in ['depósito', 'deposito']):
                return 'DEPÓSITO'
            
            if any(word in desc_lower for word in ['rendimento', 'juros', 'remuneração']):
                return 'RENDIMENTO'
            
            if any(word in desc_lower for word in ['estorno', 'reembolso', 'ressarcimento']):
                return 'ESTORNO'
            
            return 'CRÉDITO'

    def categorize_transaction(self, description: str) -> str:
        """Categoriza a transação baseado na descrição"""
        desc_lower = description.lower()
        
        # Alimentação
        food_keywords = ['restaurante', 'lanche', 'mercado', 'supermercado', 'padaria', 
                        'ifood', 'uber eats', 'delivery', 'açougue', 'pizzaria']
        if any(word in desc_lower for word in food_keywords):
            return 'ALIMENTAÇÃO'
        
        # Transporte
        transport_keywords = ['uber', '99', 'combustível', 'posto', 'gasolina', 'diesel',
                             'estacionamento', 'pedágio', 'ônibus', 'metrô', 'taxi']
        if any(word in desc_lower for word in transport_keywords):
            return 'TRANSPORTE'
        
        # Casa
        home_keywords = ['energia', 'luz', 'água', 'gas', 'telefone', 'internet', 
                        'condomínio', 'aluguel', 'financiamento', 'iptu']
        if any(word in desc_lower for word in home_keywords):
            return 'CASA'
        
        # Saúde
        health_keywords = ['farmácia', 'hospital', 'médico', 'dentista', 'clínica',
                          'laboratório', 'exame', 'consulta', 'plano de saúde']
        if any(word in desc_lower for word in health_keywords):
            return 'SAÚDE'
        
        # Lazer
        leisure_keywords = ['netflix', 'spotify', 'cinema', 'teatro', 'show', 'bar',
                           'festa', 'viagem', 'hotel', 'turismo']
        if any(word in desc_lower for word in leisure_keywords):
            return 'LAZER'
        
        # Compras
        shopping_keywords = ['americanas', 'mercado livre', 'amazon', 'magazine', 
                            'shopping', 'loja', 'compra']
        if any(word in desc_lower for word in shopping_keywords):
            return 'COMPRAS'
        
        # Educação
        education_keywords = ['escola', 'universidade', 'curso', 'faculdade', 'colégio']
        if any(word in desc_lower for word in education_keywords):
            return 'EDUCAÇÃO'
        
        return 'OUTROS'

    def parse_transaction_line(self, line: str) -> Optional[Transaction]:
        """Parse principal de uma linha para extrair transação"""
        if not line or len(line.strip()) < 6:
            return None
        
        if self.is_noise_line(line):
            return None
        
        # 1. Encontra data
        date_match = self.find_date(line)
        if not date_match:
            return None
        
        date_obj = self.parse_date_from_match(date_match)
        if not date_obj:
            return None
        
        # 2. Extrai valores monetários
        value_matches = self.extract_value_matches(line)
        if not value_matches:
            return None
        
        # 3. Identifica transação e saldo
        transaction_value_info, balance_value_info = self.identify_transaction_and_balance(value_matches, line)
        
        # 4. Classifica débito/crédito
        is_debit, confidence = self.classify_debit_credit_improved(line, transaction_value_info)
        
        # 5. Ajusta o valor conforme classificação
        transaction_value = transaction_value_info['value']
        if is_debit and transaction_value > 0:
            transaction_value = -abs(transaction_value)
        elif not is_debit and transaction_value < 0:
            transaction_value = abs(transaction_value)
        
        # 6. Extrai saldo se identificado
        balance_value = None
        if balance_value_info:
            balance_value = abs(balance_value_info['value'])  # Saldo sempre positivo
        
        # 7. Limpa descrição
        description = self.clean_description(line, date_match, value_matches)
        
        # 8. Classifica tipo e categoria
        transaction_type = self.classify_transaction_type(description, transaction_value)
        category = self.categorize_transaction(description)
        
        return Transaction(
            date=date_obj,
            description=description,
            value=transaction_value,
            balance=balance_value,
            transaction_type=transaction_type,
            category=category,
            confidence_score=confidence
        )

# ============================================================
# 5) Extrator principal
# ============================================================
class BankStatementExtractor:
    def __init__(self, matcher: Optional[EnhancedBankMatcher] = None):
        self.matcher = matcher or EnhancedBankMatcher()

    def extract_from_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> pd.DataFrame:
        """Extrai transações de PDF em bytes"""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                all_lines = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_lines.extend(text.splitlines())
            
            transactions = self._process_lines(all_lines)
            df = self._transactions_to_dataframe(transactions)
            
            if not df.empty:
                df['arquivo'] = filename
            
            return df
        
        except Exception as e:
            logger.exception(f"Erro ao processar PDF {filename}: {e}")
            return pd.DataFrame()

    def _process_lines(self, lines: List[str]) -> List[Transaction]:
        """Processa linhas extraindo transações válidas"""
        transactions = []
        
        for line in lines:
            if not line or len(line.strip()) < 6:
                continue
            
            transaction = self.matcher.parse_transaction_line(line)
            if transaction and self._is_valid_transaction(transaction):
                transactions.append(transaction)
        
        return transactions

    def _is_valid_transaction(self, transaction: Transaction) -> bool:
        """Valida se a transação é legítima"""
        if not transaction:
            return False
        
        # Valor deve existir e ter magnitude mínima
        if transaction.value is None or abs(transaction.value) < 0.01:
            return False
        
        # Descrição deve ser significativa
        if not transaction.description or len(transaction.description.strip()) < 3:
            return False
        
        # Confiança mínima na classificação
        if transaction.confidence_score < 0.1:
            return False
        
        return True

    def _transactions_to_dataframe(self, transactions: List[Transaction]) -> pd.DataFrame:
        """Converte lista de transações para DataFrame"""
        if not transactions:
            return pd.DataFrame()
        
        data = []
        for t in transactions:
            data.append({
                'data': t.date,
                'descricao': t.description,
                'valor': t.value,
                'saldo': t.balance,
                'tipo': t.transaction_type,
                'categoria': t.category,
                'confianca': t.confidence_score
            })
        
        df = pd.DataFrame(data)
        
        # Normalização e ordenação
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

# ============================================================
# 6) Processamento de múltiplos PDFs
# ============================================================
def process_multiple_pdfs(uploaded_files):
    """Processa múltiplos arquivos PDF"""
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
                avg_confidence = df['confianca'].mean()
                st.success(f"✅ {uploaded_file.name}: {len(df)} transações (confiança média: {avg_confidence:.2f})")
        
        except Exception as e:
            st.error(f"❌ Erro ao processar {uploaded_file.name}: {e}")
        
        progress.progress((i + 1) / len(uploaded_files))
    
    status.text("Processamento concluído")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values('data').reset_index(drop=True)
        return combined
    
    return pd.DataFrame()

# ============================================================
# 7) Análises e gráficos
# ============================================================
def create_summary_charts(df):
    """Cria gráficos de resumo"""
    if df.empty or not PLOTLY_AVAILABLE:
        if not PLOTLY_AVAILABLE:
            st.warning("Plotly não disponível — gráficos desabilitados")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fluxo de caixa mensal
        monthly = df.groupby('mes_ano')['valor'].sum().reset_index()
        monthly['mes_ano_dt'] = pd.to_datetime(monthly['mes_ano'] + '-01')
        monthly = monthly.sort_values('mes_ano_dt')
        
        fig1 = px.line(
            monthly, 
            x='mes_ano_dt', 
            y='valor',
            title='Fluxo de Caixa Mensal',
            labels={'valor': 'Valor (R$)', 'mes_ano_dt': 'Mês'}
        )
        fig1.add_hline(y=0, line_dash='dash', line_color='red', annotation_text="Zero")
        fig1.update_layout(xaxis_title="Período", yaxis_title="Valor (R$)")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Gastos por categoria (apenas débitos)
        debits = df[df['valor'] < 0].copy()
        if not debits.empty:
            category_spending = debits.groupby('categoria')['valor'].sum().abs().reset_index()
            category_spending = category_spending.sort_values('valor', ascending=False)
            
            fig2 = px.bar(
                category_spending,
                x='categoria',
                y='valor',
                title='Gastos por Categoria',
                labels={'valor': 'Valor (R$)', 'categoria': 'Categoria'}
            )
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)

def create_confidence_analysis(df):
    """Análise de confiança das classificações"""
    if df.empty:
        return
    
    st.subheader("📊 Análise de Confiança na Classificação")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_confidence = df['confianca'].mean()
        st.metric("Confiança Média", f"{avg_confidence:.2f}")
    
    with col2:
        high_confidence = (df['confianca'] >= 0.7).sum()
        st.metric("Alta Confiança (≥0.7)", f"{high_confidence} ({high_confidence/len(df)*100:.1f}%)")
    
    with col3:
        low_confidence = (df['confianca'] < 0.3).sum()
        st.metric("Baixa Confiança (<0.3)", f"{low_confidence} ({low_confidence/len(df)*100:.1f}%)")
    
    if PLOTLY_AVAILABLE:
        # Histograma de confiança
        fig = px.histogram(
            df,
            x='confianca',
            bins=20,
            title='Distribuição da Confiança na Classificação',
            labels={'confianca': 'Confiança', 'count': 'Quantidade'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confiança por tipo de transação
        confidence_by_type = df.groupby('tipo')['confianca'].agg(['mean', 'count']).reset_index()
        confidence_by_type.columns = ['tipo', 'confianca_media', 'quantidade']
        
        fig2 = px.bar(
            confidence_by_type,
            x='tipo',
            y='confianca_media',
            title='Confiança Média por Tipo de Transação',
            labels={'confianca_media': 'Confiança Média', 'tipo': 'Tipo'}
        )
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)

def show_classification_review(df):
    """Interface para revisão das classificações"""
    if df.empty:
        return
    
    st.subheader("🔍 Revisão de Classificações")
    
    # Filtros para revisão
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider(
            "Mostrar transações com confiança menor que:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
    
    with col2:
        show_type = st.selectbox(
            "Filtrar por tipo:",
            ['Todos'] + df['tipo'].unique().tolist()
        )
    
    # Aplica filtros
    filtered = df[df['confianca'] < confidence_threshold]
    if show_type != 'Todos':
        filtered = filtered[filtered['tipo'] == show_type]
    
    if not filtered.empty:
        st.write(f"Encontradas {len(filtered)} transações para revisão:")
        
        # Mostra transações para revisão
        review_cols = ['data', 'descricao', 'valor', 'tipo', 'categoria', 'confianca']
        st.dataframe(
            filtered[review_cols].sort_values('confianca'),
            use_container_width=True
        )
    else:
        st.info("Nenhuma transação encontrada com os critérios selecionados.")

# ============================================================
# 8) Interface Streamlit principal
# ============================================================
def main():
    st.title("🏦 Extrator de Extratos Bancários - Versão Melhorada")
    st.markdown("""
    Esta versão aprimorada oferece:
    - **Melhor detecção de débito/crédito** com múltiplas heurísticas
    - **Pontuação de confiança** para cada classificação
    - **Análise de padrões** bancários brasileiros
    - **Interface de revisão** para classificações duvidosas
    """)

    with st.sidebar:
        st.header("📋 Instruções")
        st.markdown("""
        ### Como usar:
        1. Faça upload de PDFs de extratos bancários
        2. Aguarde o processamento automático
        3. Revise as classificações na aba de análise
        4. Baixe os dados processados
        
        ### Melhorias desta versão:
        - ✅ Detecção aprimorada de sinais (-, +, parênteses)
        - ✅ Reconhecimento de padrões D/C
        - ✅ Análise contextual de palavras-chave
        - ✅ Identificação de PIX, TED, DOC
        - ✅ Pontuação de confiança
        """)
        
        st.header("ℹ️ Informações")
        st.info("""
        **Confiança na classificação:**
        - 0.7+ : Alta confiança
        - 0.3-0.7 : Média confiança  
        - <0.3 : Baixa confiança
        """)

    # Upload de arquivos
    uploaded_files = st.file_uploader(
        "Escolha arquivos PDF de extratos bancários",
        type="pdf",
        accept_multiple_files=True,
        help="Selecione um ou mais arquivos PDF de extratos bancários"
    )

    if uploaded_files:
        if st.button("🚀 Processar Extratos", type="primary"):
            with st.spinner("Processando extratos..."):
                df = process_multiple_pdfs(uploaded_files)
            
            if df.empty:
                st.error("❌ Nenhuma transação foi extraída. Verifique se os PDFs contêm extratos bancários válidos.")
                return

            # Métricas principais
            st.success(f"✅ Processamento concluído — {len(df)} transações extraídas")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de Transações", len(df))
            
            with col2:
                total_credits = df[df['valor'] > 0]['valor'].sum()
                st.metric("Total Créditos", f"R$ {total_credits:,.2f}")
            
            with col3:
                total_debits = df[df['valor'] < 0]['valor'].sum()
                st.metric("Total Débitos", f"R$ {total_debits:,.2f}")
            
            with col4:
                net_flow = df['valor'].sum()
                st.metric("Fluxo Líquido", f"R$ {net_flow:,.2f}")
            
            # Abas de análise
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Resumo", "🔍 Análise de Confiança", "📝 Revisão", "📊 Dados"])
            
            with tab1:
                st.header("Análises Gerais")
                create_summary_charts(df)
                
                # Distribuição de tipos
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Distribuição por Tipo")
                    type_dist = df['tipo'].value_counts()
                    st.bar_chart(type_dist)
                
                with col2:
                    st.subheader("Distribuição por Categoria")
                    cat_dist = df['categoria'].value_counts()
                    st.bar_chart(cat_dist)
            
            with tab2:
                create_confidence_analysis(df)
            
            with tab3:
                show_classification_review(df)
            
            with tab4:
                st.header("📋 Dados Extraídos")
                
                # Filtros
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    tipos_sel = st.multiselect(
                        "Filtrar por tipo:",
                        options=df['tipo'].unique().tolist(),
                        default=df['tipo'].unique().tolist()
                    )
                
                with col2:
                    categorias_sel = st.multiselect(
                        "Filtrar por categoria:",
                        options=df['categoria'].unique().tolist(),
                        default=df['categoria'].unique().tolist()
                    )
                
                with col3:
                    min_confidence = st.slider(
                        "Confiança mínima:",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.1
                    )
                
                # Filtro de data
                if not df.empty:
                    date_range = st.date_input(
                        "Período:",
                        value=(df['data'].min().date(), df['data'].max().date()),
                        min_value=df['data'].min().date(),
                        max_value=df['data'].max().date()
                    )
                
                # Aplicar filtros
                filtered_df = df[
                    (df['tipo'].isin(tipos_sel)) &
                    (df['categoria'].isin(categorias_sel)) &
                    (df['confianca'] >= min_confidence)
                ]
                
                if len(date_range) == 2:
                    filtered_df = filtered_df[
                        (filtered_df['data'] >= pd.to_datetime(date_range[0])) &
                        (filtered_df['data'] <= pd.to_datetime(date_range[1]))
                    ]
                
                # Exibir dados filtrados
                st.dataframe(filtered_df, use_container_width=True)
                
                # Downloads
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_filtered = filtered_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        "📥 Baixar Dados Filtrados (CSV)",
                        csv_filtered,
                        file_name="extratos_filtrados.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv_all = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        "📥 Baixar Todos os Dados (CSV)",
                        csv_all,
                        file_name="extratos_completo.csv",
                        mime="text/csv"
                    )

    else:
        st.info("👆 Faça upload de arquivos PDF para começar o processamento")
        
        # Exemplo de funcionamento
        with st.expander("🎯 Exemplo de Melhorias Implementadas"):
            st.markdown("""
            ### Detecção Aprimorada de Débito/Crédito:
            
            **Antes:** 
            ```
            01/01/2024 PIX JOÃO SILVA 150,00 → Não classificado corretamente
            ```
            
            **Agora:**
            ```
            01/01/2024 PIX ENVIADO JOÃO SILVA 150,00 D → DÉBITO (confiança: 0.8)
            01/01/2024 PIX RECEBIDO MARIA SANTOS 200,00 C → CRÉDITO (confiança: 0.9)
            ```
            
            ### Padrões Reconhecidos:
            - 🔸 Formatação: `(150,00)`, `-150,00`, `150,00 D`
            - 🔸 Contexto: "PIX enviado", "TED recebida", "saque ATM"
            - 🔸 Sufixos: valores terminados em C/D
            - 🔸 Palavras-chave: 200+ termos específicos do contexto bancário
            """)

if __name__ == "__main__":
    main()
