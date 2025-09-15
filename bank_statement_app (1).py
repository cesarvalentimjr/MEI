import streamlit as st
import pandas as pd
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor

# Imports com tratamento de erro
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError as e:
    st.error("""
    ❌ **Erro de Dependência**: pdfplumber não está instalado.
    
    **Para corrigir:**
    1. Se executando localmente: `pip install pdfplumber`
    2. Se no Streamlit Cloud: Verifique se o requirements.txt inclui:
       - pdfplumber==0.9.0
       - Pillow (dependência do pdfplumber)
    3. No Streamlit Cloud, também pode ser necessário criar packages.txt com:
       - python3-dev
       - gcc
       - g++
       - libpoppler-cpp-dev
       - pkg-config
    
    **Erro específico:** {e}
    """)
    PDF_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError as e:
    st.warning(f"Plotly não disponível: {e}. Gráficos serão desabilitados.")
    PLOTLY_AVAILABLE = False

# Para quando as dependências não estão disponíveis
if not PDF_AVAILABLE:
    st.info("💡 **Modo de Demonstração**: Sem processamento de PDF disponível.")
    st.stop()

# Configuração da página
st.set_page_config(
    page_title="Extrator de Extratos Bancários",
    page_icon="🏦",
    layout="wide"
)

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Transaction:
    """Classe para representar uma transação bancária"""
    date: datetime
    description: str
    value: float
    balance: Optional[float] = None
    transaction_type: Optional[str] = None
    category: Optional[str] = None
    source_file: Optional[str] = None

class BankPatternMatcher(ABC):
    """Classe abstrata para definir padrões de diferentes bancos"""
    
    @abstractmethod
    def get_date_patterns(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_value_patterns(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_noise_patterns(self) -> List[str]:
        pass
    
    @abstractmethod
    def parse_transaction_line(self, line: str) -> Optional[Transaction]:
        pass

class EnhancedBankMatcher(BankPatternMatcher):
    """Matcher aprimorado para diversos bancos brasileiros"""
    
    def get_date_patterns(self) -> List[str]:
        return [
            r'\b(\d{2})[\/\-\.](\d{2})[\/\-\.](\d{4})\b',  # dd/mm/yyyy ou dd-mm-yyyy ou dd.mm.yyyy
            r'\b(\d{2})[\/\-\.](\d{2})[\/\-\.](\d{2})\b',   # dd/mm/yy ou dd-mm-yy ou dd.mm.yy
            r'\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})\b', # d/m/yyyy
            r'\b(\d{4})[\/\-\.](\d{2})[\/\-\.](\d{2})\b',  # yyyy/mm/dd
            r'\b(\d{2})\s+(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\s+(\d{4})\b',  # dd mmm yyyy
        ]
    
    def get_value_patterns(self) -> List[str]:
        return [
            r'(?:R\$\s*)?([+-]?\s*\d{1,3}(?:\.\d{3})*(?:,\d{2}))',  # R$ 1.234,56 com sinal opcional
            r'([+-]?\s*\d{1,3}(?:\.\d{3})*,\d{2})',                 # 1.234,56 com sinal opcional
            r'([+-]?\s*\d+,\d{2})',                                  # 123,45 com sinal opcional
            r'([+-]?\s*\d{1,3}(?:\.\d{3})*\.\d{2})',                # Formato americano 1,234.56
            r'(\d+\s*[CD])',                                         # Valores com C (crédito) ou D (débito)
        ]
    
    def get_noise_patterns(self) -> List[str]:
        return [
            r'^BANCO\s+.*$',
            r'^AGÊNCIA\s+.*$',
            r'^CONTA\s+.*$',
            r'^SALDO\s+ANTERIOR\s*.*$',
            r'^SALDO\s+ATUAL\s*.*$',
            r'^EXTRATO\s+.*$',
            r'^PERÍODO\s+.*$',
            r'^PÁGINA\s+\d+.*$',
            r'^CPF\s*:?\s*\d+.*$',
            r'^CNPJ\s*:?\s*\d+.*$',
            r'^\s*$',
            r'^-+$',
            r'^=+$',
            r'^\*+$',
            r'^TOTAL\s+.*$',
            r'^SUBTOTAL\s+.*$',
            r'^ITAÚ\s+.*$',
            r'^BANCO\s+DO\s+BRASIL.*$',
            r'^BRADESCO.*$',
            r'^CAIXA\s+ECONÔMICA.*$',
            r'^SANTANDER.*$',
            r'^NUBANK.*$',
            r'^INTER.*$',
            r'^C6\s+BANK.*$',
            r'^BTG\s+PACTUAL.*$',
            r'^ORIGINAL.*$',
            r'^PagBank.*$',
            r'^Picpay.*$',
            r'^Stone.*$',
        ]
    
    def parse_transaction_line(self, line: str) -> Optional[Transaction]:
        """Tenta extrair uma transação de uma linha com algoritmo aprimorado"""
        line = line.strip()
        
        if len(line) < 10:  # Linhas muito curtas provavelmente não são transações
            return None
        
        # Busca por data com múltiplos formatos
        date_obj = None
        date_match = None
        
        for pattern in self.get_date_patterns():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 3:
                        if 'jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez' in pattern:
                            # Formato dd mmm yyyy
                            day, month_str, year = groups
                            months = {
                                'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
                                'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12
                            }
                            month = months.get(month_str.lower(), 1)
                            date_obj = datetime(int(year), month, int(day))
                        else:
                            g1, g2, g3 = groups
                            # Tenta diferentes combinações
                            if len(g1) == 4:  # yyyy/mm/dd
                                year, month, day = int(g1), int(g2), int(g3)
                            else:  # dd/mm/yyyy ou dd/mm/yy
                                day, month, year = int(g1), int(g2), int(g3)
                                if year < 50:  # Assumir 20xx para anos < 50
                                    year += 2000
                                elif year < 100:  # Assumir 19xx para anos >= 50 e < 100
                                    year += 1900
                            
                            date_obj = datetime(year, month, day)
                        date_match = match
                        break
                except (ValueError, KeyError):
                    continue
        
        if not date_obj:
            return None
        
        # Busca por valores monetários com lógica aprimorada
        values = []
        value_matches = []
        
        for pattern in self.get_value_patterns():
            matches = list(re.finditer(pattern, line))
            for match in matches:
                try:
                    value_str = match.group(1).strip()
                    
                    # Processa diferentes formatos
                    if value_str.endswith('C') or value_str.endswith('D'):
                        is_debit = value_str.endswith('D')
                        value_str = value_str[:-1].strip()
                    else:
                        is_debit = value_str.startswith('-') or 'débito' in line.lower() or 'saque' in line.lower()
                    
                    # Remove sinais e espaços
                    value_str = re.sub(r'[+\-\s]', '', value_str)
                    
                    # Converte formato brasileiro para float
                    if ',' in value_str and '.' in value_str:
                        # Formato 1.234,56
                        value_str = value_str.replace('.', '').replace(',', '.')
                    elif ',' in value_str and value_str.count(',') == 1:
                        # Formato 1234,56
                        value_str = value_str.replace(',', '.')
                    # Se só tem pontos, assume formato americano ou milhares brasileiros
                    elif '.' in value_str:
                        parts = value_str.split('.')
                        if len(parts[-1]) == 2:  # Último grupo tem 2 dígitos - decimal
                            value_str = ''.join(parts[:-1]) + '.' + parts[-1]
                        else:  # Separadores de milhares
                            value_str = value_str.replace('.', '')
                    
                    value = float(value_str)
                    if is_debit:
                        value = -abs(value)
                    
                    values.append(value)
                    value_matches.append(match)
                except (ValueError, AttributeError):
                    continue
        
        if not values:
            return None
        
        # Extrai descrição removendo data e valores
        description = line
        if date_match:
            description = description[:date_match.start()] + description[date_match.end():]
        
        for match in value_matches:
            start = match.start() - (date_match.end() - date_match.start() if date_match and match.start() > date_match.start() else 0)
            end = match.end() - (date_match.end() - date_match.start() if date_match and match.start() > date_match.start() else 0)
            description = description[:max(0, start)] + description[end:]
        
        # Limpa a descrição
        description = re.sub(r'[R\$\-\+\s]+', ' ', description)
        description = re.sub(r'\s+', ' ', description).strip()
        
        if not description or len(description) < 3:
            description = "Transação não identificada"
        
        # Determina valor principal e saldo
        transaction_value = values[0]
        balance = values[1] if len(values) > 1 else None
        
        # Se há múltiplos valores, tenta identificar qual é o saldo
        if len(values) > 1:
            # O saldo geralmente é o maior valor absoluto
            max_value = max(values, key=abs)
            if abs(max_value) > abs(transaction_value) * 2:  # Heurística simples
                balance = max_value
                transaction_value = [v for v in values if v != max_value][0]
        
        return Transaction(
            date=date_obj,
            description=description,
            value=transaction_value,
            balance=balance,
            transaction_type=self._classify_transaction(description, transaction_value),
            category=self._categorize_transaction(description)
        )
    
    def _classify_transaction(self, description: str, value: float) -> str:
        """Classifica o tipo de transação baseado na descrição e valor"""
        description_lower = description.lower()
        
        if value < 0:
            if any(word in description_lower for word in ['pix', 'transferencia', 'ted', 'doc', 'transf']):
                return 'TRANSFERÊNCIA_SAÍDA'
            elif any(word in description_lower for word in ['saque', 'atm', '24h']):
                return 'SAQUE'
            elif any(word in description_lower for word in ['compra', 'débito', 'cartão', 'visa', 'master']):
                return 'COMPRA_DÉBITO'
            elif any(word in description_lower for word in ['tarifa', 'taxa', 'juros', 'iof']):
                return 'TARIFA'
            else:
                return 'DÉBITO'
        else:
            if any(word in description_lower for word in ['pix', 'transferencia', 'ted', 'doc', 'transf']):
                return 'TRANSFERÊNCIA_ENTRADA'
            elif any(word in description_lower for word in ['depósito', 'deposito']):
                return 'DEPÓSITO'
            elif any(word in description_lower for word in ['salário', 'salario']):
                return 'SALÁRIO'
            elif any(word in description_lower for word in ['rendimento', 'juros']):
                return 'RENDIMENTO'
            else:
                return 'CRÉDITO'
    
    def _categorize_transaction(self, description: str) -> str:
        """Categoriza a transação para análise"""
        description_lower = description.lower()
        
        # Categorias alimentação
        if any(word in description_lower for word in ['restaurante', 'lanche', 'mercado', 'supermercado', 'açougue', 'padaria', 'ifood', 'uber eats']):
            return 'ALIMENTAÇÃO'
        
        # Categorias transporte
        elif any(word in description_lower for word in ['uber', '99', 'combustível', 'posto', 'gasolina', 'etanol', 'diesel', 'estacionamento']):
            return 'TRANSPORTE'
        
        # Categorias casa
        elif any(word in description_lower for word in ['energia', 'água', 'gás', 'telefone', 'internet', 'condomínio', 'aluguel']):
            return 'CASA'
        
        # Categorias saúde
        elif any(word in description_lower for word in ['farmácia', 'hospital', 'médico', 'dentista', 'clínica']):
            return 'SAÚDE'
        
        # Categorias lazer
        elif any(word in description_lower for word in ['cinema', 'teatro', 'show', 'netflix', 'spotify', 'streaming']):
            return 'LAZER'
        
        # Categorias compras
        elif any(word in description_lower for word in ['shopping', 'loja', 'magazine', 'americanas', 'mercado livre', 'amazon']):
            return 'COMPRAS'
        
        else:
            return 'OUTROS'

class BankStatementExtractor:
    """Classe principal para extração de dados de extratos bancários"""
    
    def __init__(self, matcher: BankPatternMatcher = None):
        self.matcher = matcher or EnhancedBankMatcher()
        self.transactions: List[Transaction] = []
    
    def extract_from_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> pd.DataFrame:
        """Extrai dados de um arquivo PDF em bytes"""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                all_text = []
                
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        all_text.extend(text.split('\n'))
                
                transactions = self._process_lines(all_text, filename)
                return self._transactions_to_dataframe(transactions)
                
        except Exception as e:
            st.error(f"Erro ao processar {filename}: {str(e)}")
            return pd.DataFrame()
    
    def _process_lines(self, lines: List[str], filename: str) -> List[Transaction]:
        """Processa as linhas extraídas do PDF"""
        transactions = []
        
        # Limpeza e identificação de candidatos
        cleaned_lines = self._remove_noise(lines)
        candidate_lines = self._identify_transaction_candidates(cleaned_lines)
        
        # Parse das transações
        for line in candidate_lines:
            transaction = self.matcher.parse_transaction_line(line)
            if transaction:
                transaction.source_file = filename
                transactions.append(transaction)
        
        return transactions
    
    def _remove_noise(self, lines: List[str]) -> List[str]:
        """Remove linhas que são claramente ruído"""
        cleaned_lines = []
        noise_patterns = self.matcher.get_noise_patterns()
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
                
            is_noise = any(re.match(pattern, line, re.IGNORECASE) for pattern in noise_patterns)
            
            if not is_noise:
                cleaned_lines.append(line)
        
        return cleaned_lines
    
    def _identify_transaction_candidates(self, lines: List[str]) -> List[str]:
        """Identifica linhas que podem conter transações usando heurísticas"""
        candidates = []
        date_patterns = self.matcher.get_date_patterns()
        value_patterns = self.matcher.get_value_patterns()
        
        for line in lines:
            # Verifica se tem data
            has_date = any(re.search(pattern, line, re.IGNORECASE) for pattern in date_patterns)
            # Verifica se tem valor monetário
            has_value = any(re.search(pattern, line) for pattern in value_patterns)
            # Verifica se tem palavras-chave de transação
            has_transaction_keywords = any(keyword in line.lower() for keyword in [
                'pix', 'ted', 'doc', 'saque', 'depósito', 'transferência', 'compra', 'débito', 'crédito'
            ])
            
            # Critérios para ser candidato
            if (has_date and has_value) or \
               (has_date and len(line.strip()) > 15) or \
               (has_value and has_transaction_keywords) or \
               (has_date and has_transaction_keywords):
                candidates.append(line)
        
        return candidates
    
    def _transactions_to_dataframe(self, transactions: List[Transaction]) -> pd.DataFrame:
        """Converte transações para DataFrame"""
        if not transactions:
            return pd.DataFrame()
        
        data = []
        for transaction in transactions:
            data.append({
                'data': transaction.date,
                'descricao': transaction.description,
                'valor': transaction.value,
                'saldo': transaction.balance,
                'tipo': transaction.transaction_type,
                'categoria': transaction.category,
                'arquivo': transaction.source_file
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('data').reset_index(drop=True)
        
        # Adiciona colunas calculadas
        df['mes'] = df['data'].dt.month
        df['ano'] = df['data'].dt.year
        df['dia_semana'] = df['data'].dt.day_name()
        df['valor_absoluto'] = df['valor'].abs()
        df['mes_ano'] = df['data'].dt.to_period('M').astype(str)
        
        return df

def process_multiple_pdfs(uploaded_files):
    """Processa múltiplos PDFs de forma eficiente"""
    extractor = BankStatementExtractor()
    all_dataframes = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f'Processando {uploaded_file.name}...')
        
        try:
            # Lê o arquivo
            pdf_bytes = uploaded_file.read()
            
            # Extrai as transações
            df = extractor.extract_from_pdf_bytes(pdf_bytes, uploaded_file.name)
            
            if not df.empty:
                all_dataframes.append(df)
                st.success(f"✅ {uploaded_file.name}: {len(df)} transações extraídas")
            else:
                st.warning(f"⚠️ {uploaded_file.name}: Nenhuma transação encontrada")
                
        except Exception as e:
            st.error(f"❌ Erro ao processar {uploaded_file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text('Processamento concluído!')
    
    if all_dataframes:
        # Combina todos os DataFrames
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df = combined_df.sort_values('data').reset_index(drop=True)
        return combined_df
    else:
        return pd.DataFrame()

def create_summary_charts(df):
    """Cria gráficos de resumo dos dados"""
    if df.empty:
        return
    
    if not PLOTLY_AVAILABLE:
        st.warning("📊 Gráficos não disponíveis (Plotly não instalado)")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de fluxo de caixa mensal
        monthly_flow = df.groupby('mes_ano')['valor'].sum().reset_index()
        monthly_flow['mes_ano'] = pd.to_datetime(monthly_flow['mes_ano'])
        
        fig_flow = px.line(
            monthly_flow, 
            x='mes_ano', 
            y='valor',
            title='Fluxo de Caixa Mensal',
            labels={'valor': 'Valor (R$)', 'mes_ano': 'Mês/Ano'}
        )
        fig_flow.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_flow, use_container_width=True)
    
    with col2:
        # Gráfico de categorias
        category_summary = df.groupby('categoria')['valor'].sum().abs().reset_index()
        category_summary = category_summary.sort_values('valor', ascending=False)
        
        fig_cat = px.bar(
            category_summary,
            x='categoria',
            y='valor',
            title='Gastos por Categoria',
            labels={'valor': 'Valor (R$)', 'categoria': 'Categoria'}
        )
        fig_cat.update_xaxes(tickangle=45)
        st.plotly_chart(fig_cat, use_container_width=True)

def main():
    """Interface principal do Streamlit"""
    
    # Título e descrição
    st.title("🏦 Extrator de Extratos Bancários")
    st.markdown("""
    Esta aplicação extrai automaticamente transações de extratos bancários em PDF.
    Suporta múltiplos bancos e formatos diferentes.
    """)
    
    # Sidebar com instruções
    with st.sidebar:
        st.header("📋 Instruções")
        st.markdown("""
        1. **Faça upload** dos seus extratos em PDF
        2. **Aguarde** o processamento
        3. **Visualize** os dados extraídos
        4. **Baixe** o resultado em CSV
        
        ### 🏛️ Bancos Suportados
        - Banco do Brasil
        - Itaú
        - Bradesco
        - Santander
        - Caixa Econômica
        - Nubank
        - Inter
        - C6 Bank
        - BTG Pactual
        - E muitos outros...
        
        ### ⚠️ Dicas
        - Use PDFs de boa qualidade
        - Evite PDFs escaneados de baixa resolução
        - Múltiplos arquivos são processados automaticamente
        """)
    
    # Upload de arquivos
    uploaded_files = st.file_uploader(
        "Escolha os arquivos PDF dos extratos",
        type="pdf",
        accept_multiple_files=True,
        help="Você pode selecionar múltiplos arquivos de uma vez"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} arquivo(s) carregado(s)")
        
        if st.button("🚀 Processar Extratos", type="primary"):
            with st.spinner("Processando arquivos..."):
                df = process_multiple_pdfs(uploaded_files)
            
            if not df.empty:
                st.success(f"✅ Processamento concluído! {len(df)} transações extraídas")
                
                # Resumo dos dados
                st.header("📊 Resumo dos Dados")
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
                
                # Gráficos
                st.header("📈 Análises")
                create_summary_charts(df)
                
                # Tabela de dados
                st.header("📋 Dados Extraídos")
                
                # Filtros
                col1, col2, col3 = st.columns(3)
                with col1:
                    selected_types = st.multiselect(
                        "Filtrar por Tipo",
                        options=df['tipo'].unique(),
                        default=df['tipo'].unique()
                    )
                
                with col2:
                    selected_categories = st.multiselect(
                        "Filtrar por Categoria",
                        options=df['categoria'].unique(),
                        default=df['categoria'].unique()
                    )
                
                with col3:
                    date_range = st.date_input(
                        "Período",
                        value=(df['data'].min(), df['data'].max()),
                        min_value=df['data'].min(),
                        max_value=df['data'].max()
                    )
                
                # Aplica filtros
                filtered_df = df[
                    (df['tipo'].isin(selected_types)) &
                    (df['categoria'].isin(selected_categories)) &
                    (df['data'] >= pd.to_datetime(date_range[0])) &
                    (df['data'] <= pd.to_datetime(date_range[1]))
                ]
                
                # Exibe tabela filtrada
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    column_config={
                        'data': st.column_config.DateColumn('Data'),
                        'valor': st.column_config.NumberColumn('Valor', format='R$ %.2f'),
                        'saldo': st.column_config.NumberColumn('Saldo', format='R$ %.2f'),
                        'valor_absoluto': st.column_config.NumberColumn('Valor Absoluto', format='R$ %.2f'),
                    }
                )
                
                # Download dos dados
                st.header("💾 Download")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Baixar CSV Filtrado",
                        data=csv,
                        file_name=f"extratos_bancarios_filtrado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv_full = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Baixar CSV Completo",
                        data=csv_full,
                        file_name=f"extratos_bancarios_completo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.error("❌ Nenhuma transação foi extraída dos arquivos fornecidos.")
    
    else:
        st.info("👆 Faça upload de arquivos PDF para começar")

if __name__ == "__main__":
    main()