# 🤖 Agente Six Sigma – Streamlit + LangChain

Este projeto cria um agente inteligente baseado em **Six Sigma**, que responde apenas com base no conteúdo de apostilas em PDF fornecidas na pasta `DADOS/`.

Ele utiliza:
- **LangChain** (retrieval + chain)
- **OpenAI Embeddings + Chat Models**
- **FAISS** para busca vetorial
- **Streamlit** para interface web

---

## 🚀 Como rodar

### 1. Clone o repositório
```bash
git clone <seu-repo>
cd <seu-repo>
```

### 2. Crie um ambiente virtual (opcional mas recomendado)
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate   # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Configure a chave da OpenAI
Crie um arquivo `.env` na raiz do projeto:
```env
OPENAI_API_KEY=coloque_sua_chave_aqui
```

### 5. Adicione os PDFs
Coloque suas apostilas na pasta `DADOS/` com os seguintes nomes:
```
DADOS/
  ├── six_sigma_apostila_agente.pdf
  ├── six_sigma_black_belt.pdf
  ├── six_sigma_green_belt.pdf
  ├── six_sigma_guia.pdf
  ├── six_sigma_introducao.pdf
  ├── six_sigma_resumo.pdf
  ├── six_sigma_white_belt.pdf
  └── six_sigma_yellow_belt.pdf
```

### 6. Execute o app no Streamlit
```bash
streamlit run app.py
```

---

## 📂 Estrutura do Projeto
```
├── agente.py          # Lógica principal do agente (carregamento, embeddings e chain)
├── app.py             # Interface Streamlit
├── requirements.txt   # Dependências do projeto
├── DADOS/             # Apostilas em PDF
└── README.md
```

---

## ⚙️ Funcionamento
1. O agente carrega os PDFs em `DADOS/`
2. Cria embeddings e indexa no FAISS
3. Recebe a pergunta do usuário via Streamlit
4. Retorna resposta baseada **somente** na apostila
   - Se não encontrar: `"Não encontrei essa resposta na apostila."`

---

## ✨ Regras do Agente
- Responde apenas com base na apostila.
- Mantém o idioma da pergunta.
- É objetivo e direto.
- Usa fórmulas práticas para:
  - Identificação do nível Belt
  - Início de mapeamento de processo
  - Benefícios em tratar um processo
  - Cálculo de ganhos financeiros, erros ou ROI

---

### 📌 Aviso Legal

Este agente utiliza como base de conhecimento o material **“Six Sigma” (Graeme Knowles, Ventus Publishing ApS, 2011), disponível gratuitamente em [Bookboon.com](http://bookboon.com/)**.  
O conteúdo é fornecido apenas para **fins educacionais e explicativos**, sem qualquer finalidade comercial.  
Todos os direitos autorais do material original permanecem com o autor e a editora.
---

