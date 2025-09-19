# 📋 README - Sistema de Autenticação Facial para Fiap Invest+

## 🎯 Visão Geral
O Sistema de Autenticação Facial é um módulo de segurança desenvolvido para o aplicativo Fiap Invest+, uma plataforma completa de controle financeiro, sistemas de gestão e dicas de investimentos. Esta solução substitui métodos tradicionais de login por reconhecimento biométrico facial, oferecendo acesso rápido e seguro ao aplicativo.

<br>

## ✨ Contexto do Aplicativo
Fiap Invest+ é um aplicativo mobile que oferece:

- 📊 Controle Financeiro Pessoal

- 💰 Sistemas de Gestão de Investimentos
- 📈 Dicas e Recomendações Personalizadas
- 🔄 Acompanhamento de Carteira em Tempo Real
- 🎯 Simulador de Estratégias de Investimento

<br>

## 🔐 Funcionalidades de Autenticação
### Reconhecimento Facial
- Acesso biométrico sem necessidade de senhas
- Processamento em tempo real via câmera do dispositivo
- Múltiplos usuários com perfis individuais
- Registro simplificado com poucos passos

<br>

## Níveis de Segurança Adaptáveis
- 🔓 Baixo: Acesso rápido para uso diário
- ⚖️ Médio: Balanceado entre segurança e conveniência (recomendado)
- 🔒 Alto: Máxima segurança para operações críticas

<br>

## 🛠️ Tecnologias Utilizadas
- Python 3.x - Linguagem principal
- OpenCV - Processamento de imagem e visão computacional
- Haar Cascade - Algoritmo para detecção facial
- NumPy - Manipulação de arrays numéricos
- Webcam/API Câmera - Captura de imagens em tempo real

<br>

## 📦 Instalação e Configuração

### Instalar dependências
```bash
pip install opencv-python numpy
```
<br>

## 🤝 Suporte e Contato
### Problemas Comuns
- Câmera não detectada: Verifique permissões do sistema
- Baixa precisão: Melhore a iluminação do ambiente
- Múltiplos rostos: Certifique-se de estar sozinho na frente da câmera
<br>

## 🧾 Nota Ética sobre Uso de Dados Faciais

Este projeto utiliza reconhecimento facial com Haarcascade **exclusivamente para fins educacionais e de demonstração técnica**.  

- Nenhuma imagem ou dado biométrico é armazenado, compartilhado ou utilizado para identificar pessoas reais.  
- O reconhecimento facial aqui implementado não deve ser usado em ambientes de produção que envolvam segurança, autenticação sensível ou monitoramento de indivíduos.  
- Qualquer aplicação prática desta tecnologia deve respeitar princípios de **privacidade**, **consentimento informado** e estar em conformidade com legislações como a **LGPD** (Lei Geral de Proteção de Dados) e o **GDPR** (Regulamento Europeu de Proteção de Dados).  
- Este projeto **não endossa** o uso indevido de dados biométricos e reforça a importância de considerar os impactos éticos e sociais relacionados ao reconhecimento facial.  
