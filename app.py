
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
        if s.rfind(",") > s.rfind("."): # Brazilian format
            s = s.replace(".", "").replace(",", ".")
        else: # US format
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
        if re.search(r'Nenhuma movimentação realizada', content, re.IGNORECASE):
            return 'NUBANK_EMPTY'
        return 'GENERIC'

class BankParser:
    def __init__(self):
        pass

    def parse_bb(self, lines: List[str]) -> List[Transaction]:
        transactions = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if len(line) < 10 or "SALDO ANTERIOR" in line.upper() or "S A L D O" in line.upper() or "Dt. balancete Dt. movimento" in line.upper():
                i += 1
                continue

            if "BB RENDE FÁCIL" in line.upper():
                i+=1
                continue

            bb_pattern = re.compile(r'(\d{2}/\d{2}/\d{4})\s+(.+)')
            match = bb_pattern.match(line)

            if match:
                try:
                    date_obj = datetime.strptime(match.group(1), '%d/%m/%Y')
                    remaining_line = match.group(2).strip()

                    parts = remaining_line.split()
                    value_str = "0"
                    balance_str = "0"
                    description = ""

                    if len(parts) >= 2:
                        if parts[-1] in ['C', 'D'] and len(parts) >=3:
                            balance_str = parts[-2]
                            value_candidate = ""
                            for k in range(len(parts) - 3, -1, -1):
                                if parse_monetary_string(parts[k]):
                                    value_candidate = parts[k]
                                    description = " ".join(parts[:k])
                                    break
                            value_str = value_candidate
                        else:
                            balance_str = parts[-1]
                            value_str = parts[-2]
                            description = " ".join(parts[:-2])

                    j = i + 1
                    while j < len(lines) and not re.match(r'\d{2}/\d{2}/\d{4}', lines[j]):
                        description += " " + lines[j].strip()
                        j += 1

                    value = parse_monetary_string(value_str)
                    if value is None:
                        i += 1
                        continue

                    if "D" in parts[-1].upper() or (value < 0):
                        value = -abs(value)
                    else:
                        value = abs(value)

                    balance = parse_monetary_string(balance_str)

                    transaction = Transaction(
                        date=date_obj,
                        description=description.strip(),
                        value=value,
                        balance=balance,
                        source_bank='BB',
                        confidence_score=0.95
                    )
                    transactions.append(transaction)
                    i = j
                except Exception as e:
                    logger.warning(f"Erro ao processar linha BB: {line} - {e}")
                    i += 1
                    continue
            else:
                i += 1
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
            except Exception as e:
                logger.warning(f"Erro ao processar linha CAIXA: {line_clean} - {e}")
        return transactions

    def parse_xp(self, lines: List[str]) -> List[Transaction]:
        transactions = []
        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 15:
                continue
            xp_pattern = re.compile(
                r'(\d{2}/\d{2}/\d{4})\s+'
                r'(.+?)\s+'
                r'([\d.,]+|--)\s*'
                r'([\d.,]+|--)\s*'
                r'([\d.,]+)',
                re.IGNORECASE
            )
            match = xp_pattern.search(line_clean)
            if not match:
                continue
            try:
                date_obj = datetime.strptime(match.group(1), '%d/%m/%Y')
                description = match.group(2).strip()
                debit_str = match.group(3).strip()
                credit_str = match.group(4).strip()
                saldo_str = match.group(5).strip()
                value = None
                if credit_str and credit_str != '--':
                    value = parse_monetary_string(credit_str)
                    if value:
                        value = abs(value)
                elif debit_str and debit_str != '--':
                    value = parse_monetary_string(debit_str)
                    if value:
                        value = -abs(value)
                if value is None:
                    continue
                saldo = parse_monetary_string(saldo_str)
                transaction = Transaction(
                    date=date_obj,
                    description=description,
                    value=value,
                    balance=saldo,
                    source_bank='XP',
                    confidence_score=0.70
                )
                transactions.append(transaction)
            except Exception as e:
                logger.warning(f"Erro ao processar linha XP: {line_clean} - {e}")
        return transactions

    def parse_itau(self, lines: List[str]) -> List[Transaction]:
        transactions = []
        current_year = datetime.now().year
        month_mapping = {
            'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
            'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12
        }

        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 15:
                continue

            # Itaú pattern: DD / MMM DESCRIPTION VALUE
            itau_pattern = re.compile(
                r'(\d{2})\s+/\s+(\w{3})\s+'  # Day and abbreviated month (e.g., 02 / dez)
                r'(.+?)\s+'  # Description (non-greedy)
                r'(-?[\d.,]+)'  # Value (can be negative)
            )
            match = itau_pattern.search(line_clean)

            if match:
                try:
                    day = int(match.group(1))
                    month_abbr = match.group(2).lower()
                    month = month_mapping.get(month_abbr)
                    if not month:
                        raise ValueError(f"Mês abreviado desconhecido: {month_abbr}")

                    date_obj = datetime(current_year, month, day)
                    description = match.group(3).strip()
                    value_str = match.group(4).strip()

                    if 'SALDO' in description.upper():
                        continue

                    value = parse_monetary_string(value_str)
                    if value is None:
                        continue

                    transaction = Transaction(
                        date=date_obj,
                        description=description,
                        value=value,
                        balance=None,  # Itaú statements often don't have a running balance per transaction
                        source_bank='ITAU',
                        confidence_score=0.80
                    )
                    transactions.append(transaction)
                except Exception as e:
                    logger.warning(f"Erro ao processar linha ITAU: {line_clean} - {e}")
        return transactions

    def parse_santander(self, lines: List[str]) -> List[Transaction]:
        transactions = []
        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 15:
                continue
            santander_pattern = re.compile(
                r'(\d{2}/\d{2}/\d{4})\s+'
                r'(.+?)\s+'
                r'([\d.,]+)'
            )
            match = santander_pattern.search(line_clean)
            if not match:
                continue
            try:
                date_obj = datetime.strptime(match.group(1), '%d/%m/%Y')
                description = match.group(2).strip()
                value_str = match.group(3).strip()
                if 'SALDO' in description.upper():
                    continue
                value = parse_monetary_string(value_str)
                if value is None:
                    continue
                transaction = Transaction(
                    date=date_obj,
                    description=description,
                    value=value,
                    balance=None, # Santander statements often don't have a running balance per transaction
                    source_bank='SANTANDER',
                    confidence_score=0.70
                )
                transactions.append(transaction)
            except Exception as e:
                logger.warning(f"Erro ao processar linha SANTANDER: {line_clean} - {e}")
        return transactions

    def parse_sicoob(self, lines: List[str]) -> List[Transaction]:
        transactions = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if len(line) < 15 or "SALDO ANTERIOR" in line.upper() or "SALDO DO DIA" in line.upper() or "SALDO BLOQ.ANTERIOR" in line.upper():
                i += 1
                continue

            # Sicoob pattern: DATE HISTÓRICO VALOR C/D
            sicoob_pattern = re.compile(
                r'(\d{2}/\d{2})\s+'  # Date (DD/MM)
                r'(.+?)\s+'  # Description (non-greedy)
                r'([\d.,]+)\s*([CD])'  # Value and C/D indicator
            )
            match = sicoob_pattern.search(line)

            if match:
                try:
                    # Assuming the year is the current year, which might need adjustment
                    date_str = f"{match.group(1)}/{datetime.now().year}"
                    date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                    description_part = match.group(2).strip()
                    value_str = match.group(3)
                    value_dc = match.group(4).upper()

                    # Handle multi-line descriptions
                    full_description = description_part
                    next_line_idx = i + 1
                    while next_line_idx < len(lines) and not re.match(r'\d{2}/\d{2}', lines[next_line_idx]):
                        if not re.search(r'SALDO (ANTERIOR|DO DIA|BLOQ.ANTERIOR)', lines[next_line_idx].upper()):
                            full_description += " " + lines[next_line_idx].strip()
                        next_line_idx += 1

                    value = parse_monetary_string(value_str)
                    if value is None:
                        i = next_line_idx
                        continue

                    if value_dc == 'D':
                        value = -abs(value)
                    else:
                        value = abs(value)

                    transaction = Transaction(
                        date=date_obj,
                        description=full_description.strip(),
                        value=value,
                        balance=None,  # Sicoob statements often don't have a running balance per transaction
                        source_bank='SICOOB',
                        confidence_score=0.85
                    )
                    transactions.append(transaction)
                    i = next_line_idx
                except Exception as e:
                    logger.warning(f"Erro ao processar linha SICOOB: {line} - {e}")
                    i += 1
                    continue
            else:
                i += 1
        return transactions

    def parse_nubank(self, lines: List[str]) -> List[Transaction]:
        transactions = []
        current_year = datetime.now().year
        month_mapping = {
            'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
            'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12
        }

        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 10:
                continue

            if "Nenhuma movimentação realizada." in line_clean:
                return []

            # Nubank format: DD MMM DESCRIPTION VALUE
            # Example: 01 MAR PIX ENVIADO -100,00
            nubank_pattern = re.compile(
                r'(\d{2})\s+(\w{3})\s+'  # Day and abbreviated month (e.g., 01 MAR)
                r'(.+?)\s+'  # Description (non-greedy)
                r'(-?[\d.,]+)'  # Value (can be negative)
            )
            match = nubank_pattern.match(line_clean)

            if match:
                try:
                    day = int(match.group(1))
                    month_abbr = match.group(2).lower()
                    month = month_mapping.get(month_abbr)
                    if not month:
                        raise ValueError(f"Mês abreviado desconhecido: {month_abbr}")

                    date_obj = datetime(current_year, month, day)
                    description = match.group(3).strip()
                    value_str = match.group(4)

                    value = parse_monetary_string(value_str)
                    if value is None:
                        continue

                    transaction = Transaction(
                        date=date_obj,
                        description=description,
                        value=value,
                        balance=None,  # Nubank statements don't have a running balance
                        source_bank='NUBANK',
                        confidence_score=0.90
                    )
                    transactions.append(transaction)
                except Exception as e:
                    logger.warning(f"Erro ao processar linha NUBANK: {line} - {e}")
        return transactions

    def parse_generic(self, lines: List[str]) -> List[Transaction]:
        transactions = []
        # A very basic generic parser, tries to find lines with date, description and value
        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 15:
                continue
            # Generic pattern: Date (DD/MM/YYYY), Description (anything), Value (monetary)
            generic_pattern = re.compile(r'(\d{2}/\d{2}/\d{4})\s+(.+?)\s+([\d.,-]+)C?\s*$')
            match = generic_pattern.search(line_clean)
            if match:
                try:
                    date_obj = datetime.strptime(match.group(1), '%d/%m/%Y')
                    description = match.group(2).strip()
                    value_str = match.group(3).strip()
                    value = parse_monetary_string(value_str)
                    if value is None:
                        continue
                    transaction = Transaction(
                        date=date_obj,
                        description=description,
                        value=value,
                        balance=None,
                        source_bank='GENERIC',
                        confidence_score=0.70
                    )
                    transactions.append(transaction)
                except Exception as e:
                    logger.warning(f"Erro ao processar linha GENERIC: {line_clean} - {e}")
        return transactions

class BankStatementExtractor:
    def __init__(self):
        self.matcher = ProfessionalBankMatcher()
        self.parser = BankParser()

    def _get_parser(self, bank_format: str):
        return {
            'BB': self.parser.parse_bb,
            'BRADESCO': self.parser.parse_bradesco,
            'INTER': self.parser.parse_inter,
            'CAIXA': self.parser.parse_caixa,
            'XP': self.parser.parse_xp,
            'ITAU': self.parser.parse_itau,
            'SANTANDER': self.parser.parse_santander,
            'SICOOB': self.parser.parse_sicoob,
            'NUBANK': self.parser.parse_nubank,
            'NUBANK_EMPTY': lambda lines: [],
        }.get(bank_format, self.parser.parse_generic)

    def extract_from_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> List[Transaction]:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text_lines = []
                for page in pdf.pages:
                    text_lines.extend(page.extract_text().split('\n'))

            if not text_lines:
                logger.warning(f"Nenhum texto extraído de {filename}")
                return []

            bank_format = self.matcher.detect_bank_format(text_lines)
            logger.info(f"Banco detectado para {filename}: {bank_format}")

            parser_func = self._get_parser(bank_format)
            transactions = parser_func(text_lines)

            for t in transactions:
                t.source_file = filename

            logger.info(f"Processado {filename}: {len(transactions)} transações encontradas.")
            
            if transactions:
                df = pd.DataFrame([t.__dict__ for t in transactions])
                # Ensure date column is datetime and sort
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                # Rename columns for clarity
                df = df.rename(columns={
                    "date": "data",
                    "description": "descricao",
                    "value": "valor",
                    "balance": "saldo",
                    "transaction_type": "tipo_transacao",
                    "category": "categoria",
                    "source_file": "arquivo_origem",
                    "source_bank": "banco_detectado",
                    "confidence_score": "confianca"
                })
                return df
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Falha ao processar {filename}: {e}")
            return []

