# PROTOCOLO LLM LOCKED — Personas sintéticas × crédito à habitação (PT)

**STATUS: LOCKED** — protocolo de prompts para aprovação antes da recolha.  
**Tipo de dados:** *sintético / experimental* (personas geradas + estímulos). **Não** confundir com dados empíricos de respondentes humanos.  
**Recolha N=800:** **NÃO iniciada.**

| Camada | Versão LOCKED |
|--------|----------------|
| **Stimuli** (títulos EC neutros; corpos EC/CH/VAL; Miguel) | **v1.1** — inalterados desde v1.1 |
| **Protocolo LLM** (dois turnos + ops + API + MC_FRAME approach/avoidance) | **v1.6 FINAL** (`gpt-5.6-luna`) |

Ver `stimuli_locked.md` / `stimuli_locked.json`. **Não** rotular stimuli como «v1.3».

---

## Changelog

| Versão | Nota |
|--------|------|
| **LOCKED v1.6 FINAL** | **Protocolo — MC_FRAME approach/avoidance:** item «Pensando apenas na frase publicitária apresentada pela instituição bancária, como classificaria a forma como está formulada?»; escala 1=Enfatiza sobretudo evitar resultados indesejáveis … 7=Enfatiza sobretudo alcançar resultados desejáveis. Chave JSON `MC_FRAME` (substitui `MC_VAL`). **Porquê:** v1.5 fixou a contaminação EC (referente estreito) mas o MC afetivo (negativa/positiva) falhou a discriminar VAL_NEG. O constructo passou de polaridade afetiva para framing obtenção vs evitação (operacionalização à Dobson & Poels). Interpretação: VAL_POS→MC_FRAME alto; VAL_NEG→MC_FRAME baixo. Nota tese: «positivo/negativo» = obtenção vs evitação, não agradável/desagradável. Modelo/params = luna. **Corpos EC/CH/VAL/BI/CE = stimuli v1.1 inalterados.** |
| **LOCKED v1.5 FINAL** | **Protocolo — MC_VAL referent-specific:** item «Pensando apenas na frase publicitária apresentada pela instituição bancária, e não na informação económica apresentada anteriormente, como classificaria a forma como essa frase está formulada?»; escala 1=Claramente negativa … 4=Neutra … 7=Claramente positiva. Chave JSON `MC_VAL` mantida. **Jury:** o referente foi estreitado porque «a mensagem» (v1.4) era ambígua entre o slogan bancário e o bloco EC prévio — o MC deve isolar a formulação da frase publicitária. Modelo/params = luna. **Corpos EC/CH/VAL/BI/CE = stimuli v1.1 inalterados.** |
| LOCKED v1.0 | Textos EC/CH/VAL literais; títulos EC com prefixo Expansão/Contração. |
| LOCKED v1.1 | **Stimuli:** títulos EC neutros idênticos; corpos/Miguel/CH/VAL inalterados. (Apresentação ainda era um único user message.) |
| LOCKED v1.2 | **Protocolo — apresentação:** **mesmo chat, dois turnos:** Turno 1 = DV (BI); Turno 2 = MC. Motivo: com um LLM a mensagem completa é visível antes de qualquer token — MC no mesmo user message contaminam a BI. Persistência do JSON do Turno 1 **antes** de enviar o Turno 2. **Stimuli inalterados** (permanecem v1.1). |
| LOCKED v1.2.1 (ops) | **Protocolo — polish operacional (arquitectura inalterada):** (1) separação explícita Stimuli v1.1 vs Protocolo v1.2.1; (2) mensagens de retry de formato **literais congeladas**; (3) política LOCKED: se Turno 1 inválido após 1 retry → **ainda assim** enviar Turno 2 (se execução normal); `technical_failure` separado; (4) cosméticos de rendimento no render (`build_prompts_N800.py`). Stimuli (corpos, título neutro, Miguel, CH, VAL) **inalterados** desde v1.1. |
| **LOCKED v1.4 FINAL** | **Protocolo — MC_VAL bipolar:** item «Como classificaria a forma como a mensagem está formulada?»; escala 1=Claramente negativa … 4=Neutra … 7=Claramente positiva. Chave JSON `MC_VAL` mantida. Modelo/params = luna (v1.3.1). **Corpos EC/CH/VAL/BI/CE = stimuli v1.1 inalterados.** Pré-teste N=40 `MCPROTO14_*`: criteria_pass false (NEG não discriminou). |
| **LOCKED v1.3.1 (modelo)** | **Protocolo — troca de modelo:** `gpt-5.6-sol` → **`gpt-5.6-luna`**. Restantes params (temp=1.0; reasoning.effort=none; max_output_tokens=128; SO strict; previous_response_id + instructions) inalterados. **Stimuli v1.1 inalterados.** Observações já recolhidas com `sol` (piloto N=8, P001 smoke, pré-testes) **não** são misturáveis com recolha `luna` sob o mesmo instrumento — amostra final N=800 deve ser uniforme em modelo. |
| **LOCKED v1.3 FINAL** | **Protocolo — lock de API/params + Responses API + auditoria:** (1) modelo/params **congelados** (gpt-5.6-luna; temperature=1,0; reasoning.effort=none; max_output_tokens=128; Structured Outputs strict); (2) Responses API: **reenviar** o mesmo system `instructions` no Turno 2 com `previous_response_id`, porque as `instructions` **não** são herdadas; (3) campos de auditoria extra nos logs raw (`*_response_id`, `api_model_returned`, `service_tier`, finish status, tokens, `schema_version`/`schema_hash`). **Stimuli inalterados** (permanecem v1.1). |

---

## Porque dois turnos? (documentação)

Com **humanos**, a ordem das páginas (ou ecrãs) esconde os manipulation checks até **depois** da DV (intenção comportamental).  
Com um **LLM**, o *user message* completo fica visível **antes** de qualquer token ser gerado. Colocar itens MC depois da BI **na mesma mensagem** ainda assim contamina a BI (o modelo «vê» MC_FRAME e CE1–CE3 ao gerar BI1–BI4).

**Correção (protocolo v1.2+):** mesmo chat, **dois turnos**; gravar / validar o JSON do Turno 1 **antes** de enviar a mensagem do Turno 2.

---

## 1. Arquitetura (visão geral)

```
personas LOCKED
    (personas_v3_1_N800_seed20260921800.csv)
            │
            ▼
randomização LOCKED  ──seed 20260921801──►  base_experimental_N800_seed20260921801.csv
    (randomizacao_N800_seed20260921801.csv)
            │
            ▼
stimuli LOCKED v1.1  (stimuli_locked.md / .json)  — títulos neutros; corpos fixos
            │
            ▼
gerador de prompts  (build_prompts_N800.py)  — protocolo v1.6 FINAL
            │
            ├── system (mesmo texto literal em ambos os turnos; reenviado nas instructions)
            ├── turn1_user  (PERFIL + EC + CH + VAL + BI → JSON BI)
            └── turn2_user  (opener + MC_FRAME → JSON MC; sem reenviar estímulos)
            │
            ▼
         OpenAI Responses API
            modelo gpt-5.6-luna | temperature=1.0 | reasoning.effort=none
            Structured Outputs (schema_v1) | max_output_tokens=128
            Turno 1 → persistir → Turno 2 (previous_response_id + instructions)
            │
            ▼
    raw_responses.csv  (turn1_raw, turn2_raw, parse/retry, IDs, tokens, schema_*; imutável após recolha)
            │
            ▼
       validação JSON (+ 1 retry de formato por turno, se necessário)
            │
            ▼
    analysis_dataset.csv  (derivado, transforms reproduzíveis)
```

Between-subjects: **um chat independente por persona**; **uma única condição** (C1–C4) por persona; **dois turnos** nesse chat.

---

## 2. Ficheiros LOCKED (inputs)

| Papel | Ficheiro | Notas |
|-------|----------|-------|
| Personas | `personas_v3_1_N800_seed20260921800.csv` | **Não modificar.** MD5 `14c8839535451b47be14267697069182` |
| Randomização | `randomizacao_N800_seed20260921801.csv` | Seed `20260921801` **congelada.** MD5 `4814fd05c030987a1365efda9aeb323d` |
| Base mestre | `base_experimental_N800_seed20260921801.csv` | Join personas+randomização; ver `BASE_EXPERIMENTAL_N800.md` |
| Stimuli (humano) | `stimuli_locked.md` | **Stimuli v1.1** (EC/CH/VAL); protocolo **v1.6 FINAL** (MC_FRAME approach/avoidance) |
| Stimuli (máquina) | `stimuli_locked.json` | Corpos EC/CH/VAL v1.1 + MC_FRAME approach/avoidance v1.6 + protocolo (API, schemas, turnos, retries) |
| Exemplos | `exemplos_prompts_C1_C4.md` | 4 exemplos com **ambos** os turnos |
| Gerador | `build_prompts_N800.py` | Emite system + turn1_user + turn2_user; schemas/retries; **sem** API |

---

## 3. Ordem conceptual vs apresentação ao LLM

### Ordem conceptual (inalterada)

1. Bloco **PERFIL** (persona)  
2. Estímulo de ciclo económico → `EC_EXP` **ou** `EC_CON` (título neutro idêntico)  
3. `CH_CONTEXT`  
4. Slogan de valência entre aspas → `VAL_POS` **ou** `VAL_NEG`  
5. Intenção comportamental → **BI1–BI4**  
6. Manipulation checks → **MC_FRAME**, depois **CE1–CE3**

### Apresentação ao LLM (protocolo v1.2 / v1.5 FINAL)

| Turno | Conteúdo da *user message* | Output JSON |
|-------|----------------------------|-------------|
| **1 (DV)** | PERFIL + EC + CH + VAL + bloco QUESTÕES (BI1–BI4) + instrução JSON | `BI1`–`BI4` apenas |
| **2 (MC)** | Opener neutro + MC_FRAME + CE1–CE3 + instrução JSON (**não** reenviar persona/EC/CH/VAL) | `MC_FRAME`, `CE1`–`CE3` |

Persistir a resposta raw do Turno 1 **antes** de enviar o Turno 2.

---

## 4. System prompt (LOCKED — texto exacto)

O texto de system é o **mesmo** em **ambos** os turnos:

```
És um respondente sintético num estudo de investigação.
Responde de acordo com o perfil atribuído e apenas com a informação apresentada.
Não procures inferir os objetivos ou hipóteses do estudo.
Quando uma característica não estiver definida, não inventes novos factos biográficos.
Responde apenas no formato solicitado.
```

Na implementação através da Responses API, o system prompt será enviado explicitamente em ambos os pedidos. No Turno 2, será utilizado o mesmo texto literal do Turno 1 juntamente com o `previous_response_id`, uma vez que as `instructions` não são automaticamente herdadas entre respostas encadeadas.

### Forma das chamadas (Responses API)

**Turno 1:**
- `instructions` = `SYSTEM_PROMPT`
- `input` = `TURN1_USER`
- (sem `previous_response_id`)

**Turno 2:**
- `instructions` = `SYSTEM_PROMPT` (mesmo texto literal)
- `previous_response_id` = `ID_TURNO_1`
- `input` = `TURN2_USER`

**Não** reenviar persona / EC / CH / VAL no `input` do Turno 2 (esses conteúdos permanecem via encadeamento do histórico).

### Proibições no system prompt (CRÍTICO)

O system prompt **nunca** deve mencionar: H1/H2/H3; «expansão»/«contração» como rótulos de condição; framing; valência; efeito esperado; C1–C4; ou quaisquer labels de condição.  
(Os corpos EC e os itens CE podem conter linguagem económica — isso é estímulo/medida, **não** instrução de sistema.)

### Proibições gerais nos prompts (metadados)

Labels de metadados (`condicao`, `ciclo`, `valencia`, `ciclo_id`, `valencia_id`, C1–C4, H1/H2/H3, framing) **nunca** aparecem nas mensagens enviadas ao modelo.

---

## 5. Turno 1 — mensagem de utilizador (DV)

A *user message* do Turno 1 contém **apenas**:

### 5.1 Bloco PERFIL

```
PERFIL DO RESPONDENTE
Idade: {idade}
Situação profissional: {situacao_profissional_label}
Composição do agregado: {composicao_label}
{rendimento_prompt_pt_normalizado}
Situação habitacional: {situacao_habitacional_label}
```

**Nota (v1.2.1+):** a linha de rendimento é o conteúdo de `rendimento_prompt_pt` **sem** prefixo adicional `Rendimento: ` (o CSV já começa por `Rendimento bruto anual do agregado: …`). Cosméticos aplicados **só no render** por `build_prompts_N800.py` (CSV personas **intocado**): `ate` → `até`; milhares com espaço (`12 200`); símbolo € mantido.

#### Tabela de mapeamento (códigos CSV → rótulos PT-PT)

**Situação profissional**

| Código CSV | Rótulo no prompt |
|------------|------------------|
| `empregado` | Empregado |
| `desempregado` | Desempregado |
| `inativo` | Inativo |

**Composição do agregado** (níveis observados no CSV N=800)

| Código CSV | Rótulo no prompt |
|------------|------------------|
| `unipessoal` | Agregado unipessoal |
| `casal_sem_filhos` | Casal sem filhos |
| `casal_com_filhos` | Casal com filhos |
| `monoparental` | Agregado monoparental |
| `outro` | Outra composição do agregado |

**Situação habitacional** (níveis observados no CSV N=800)

| Código CSV | Rótulo no prompt |
|------------|------------------|
| `prop_com_hipoteca_RP` | Proprietário com hipoteca sobre residência principal |
| `prop_sem_hipoteca` | Proprietário sem hipoteca |
| `nao_proprietario` | Não proprietário (inclui arrendatário e outras situações sem propriedade) |

### 5.2 Estímulo EC (`EC_EXP` ou `EC_CON`)

Inserir o **texto completo** (título + corpo) conforme `stimuli_locked.json` (**stimuli v1.1**).

- **Título (ambos, idêntico, v1.1):** `Previsões Económicas para os Próximos Anos`  
- Corpos paralelos; narrativa **Miguel** mantida; sem inventar texto de estímulo.

### 5.3 `CH_CONTEXT` (inalterado)

```
Crédito à Habitação

Imagine que se encontra a considerar a contratação de um crédito à habitação para aquisição de habitação própria permanente.

Neste contexto, uma instituição bancária apresenta-lhe uma proposta de crédito à habitação e comunica a seguinte mensagem:
```

### 5.4 Slogan de valência (entre aspas)

- `VAL_POS`: `"Faça ambos: pague menos e viva mais."`  
- `VAL_NEG`: `"Não pague mais nem viva menos."`

### 5.5 Bloco de questões (BI) — sem título de constructo

**Rótulo suave (LOCKED):** `QUESTÕES`  
(**Não** usar «INTENÇÃO COMPORTAMENTAL».)

**Instrução (LOCKED = anexos):**

`Com base na informação apresentada sobre este crédito à habitação, indique a probabilidade de realizar cada uma das seguintes ações.`

**Escala:** 1 = Muito improvável … 7 = Muito provável

| Código | Item LOCKED (anexos) |
|--------|----------------------|
| BI1 | Solicitar este crédito à habitação. |
| BI2 | Colocar este crédito à habitação entre as minhas primeiras opções. |
| BI3 | Fazer um esforço considerável para obter este crédito à habitação. |
| BI4 | Recomendar este crédito à habitação a um amigo ou familiar. |

**Nota:** existem variantes ligeiras de wording no §3.3 do rascunho da dissertação; a versão **LOCKED** para recolha é a dos **anexos** acima.

### 5.6 Output Turno 1 (JSON apenas BI)

Exigir **exactamente** o objeto abaixo e **nada mais** (Structured Outputs, `schema_v1`, modo strict):

```json
{
  "BI1": <int 1-7>,
  "BI2": <int 1-7>,
  "BI3": <int 1-7>,
  "BI4": <int 1-7>
}
```

**Turno 1 NÃO contém** MC_FRAME, CE1, CE2, CE3, nem títulos «VERIFICAÇÕES» / «MANIPULATION».

---

## 6. Após o Turno 1 (obrigatório antes do Turno 2)

1. Parse/validar o JSON BI (chaves `BI1`–`BI4`; inteiros ∈ [1, 7]).  
2. Aplicar a regra de inválido do Turno 1 (secção 8): **um** retry só de formato.  
3. **Persistir** `turn1_raw` (e estado de parse/retry, `turn1_response_id`, tokens) **antes** de enviar o Turno 2.

---

## 7. Turno 2 — mensagem de utilizador (manipulation checks)

Nova *user message* no **mesmo chat** (`previous_response_id` = ID do Turno 1), com o **mesmo** `instructions` (system) reenviado. **Não** reenviar persona, EC, CH nem valência (já estão no histórico encadeado).

### 7.1 Opener neutro

Exemplo LOCKED:

`Responda agora às seguintes afirmações.`

**Proibido:** títulos «MANIPULATION CHECKS», «VERIFICAÇÕES», ou nomes de constructo.

### 7.2 Itens

**MC_FRAME** (mesmo item de framing em **todas** as condições — não ramificar wording; chave JSON `MC_FRAME`):

`Pensando apenas na frase publicitária apresentada pela instituição bancária, como classificaria a forma como está formulada?`  
Escala: 1 = Enfatiza sobretudo evitar resultados indesejáveis … 7 = Enfatiza sobretudo alcançar resultados desejáveis  

*(v1.6 FINAL — framing approach/avoidance (Dobson & Poels); substitui MC_VAL afetivo v1.5; sem âncora média explícita.)*

**CE** (Sirola & Pitesa, 2017 — adaptação PT):

| Código | Item |
|--------|------|
| CE1 | A economia está má. |
| CE2 | A economia encontra-se numa fase de contração. |
| CE3 | É provável ocorrer uma recessão económica. |

Escala CE: 1 = Discordo totalmente … 5 = Concordo totalmente

### 7.3 Output Turno 2 (JSON apenas MC)

Structured Outputs, `schema_v1`, modo strict:

```json
{
  "MC_FRAME": <int 1-7>,
  "CE1": <int 1-5>,
  "CE2": <int 1-5>,
  "CE3": <int 1-5>
}
```

**Turno 2 NÃO reinclui** texto EC, slogan de valência, nem itens BI.

---

## 8. Regra de resposta inválida (por turno, independente)

Aplicar **separadamente** ao Turno 1 e ao Turno 2:

1. Fazer parse do JSON; exigir as chaves do turno; inteiros nos intervalos correctos  
   (Turno 1: BI* ∈ [1,7]; Turno 2: MC_FRAME ∈ [1,7], CE* ∈ [1,5]).  
2. Se inválido: **UM** retry de correção de formato **apenas** — usar a mensagem literal LOCKED abaixo; **não** reapresentar nem modificar estímulos; **não** alterar system / persona / EC / VAL; **não** pedir para reconsiderar / repensar a resposta.  
3. Registar: raw original do turno + tentativa de correção + estado final desse turno.  
4. Se continuar inválido → `parse_status`/`status` inválido para esse turno; campos de análise em missing; **manter** o raw.

### 8.1 Mensagens de retry LOCKED (literais exactas)

**Turno 1 — retry (exacto):**

```
A resposta anterior não respeitou o formato solicitado. Responde novamente apenas com um objeto JSON válido contendo exatamente as chaves BI1, BI2, BI3 e BI4, com valores inteiros entre 1 e 7. Não acrescentes qualquer outro texto.
```

**Turno 2 — retry (exacto):**

```
A resposta anterior não respeitou o formato solicitado. Responde novamente apenas com um objeto JSON válido contendo exatamente as chaves MC_FRAME, CE1, CE2 e CE3, respeitando os intervalos indicados. Não acrescentes qualquer outro texto.
```

Documentado: o retry corrige **apenas o FORMATO** — nunca pedir rethink/reconsider; nunca reenviar estímulos. Strings também em `stimuli_locked.json` (`turnos.turno1.retry_message` / `turnos.turno2.retry_message`) e expostas por `build_prompts_N800.py`.

### 8.2 Política Turno 2 se Turno 1 inválido (LOCKED v1.2.1+)

Se o Turno 1 permanecer inválido após um retry de formato: **ainda assim enviar o Turno 2**, desde que a execução da conversa tenha sido **normal** (não falha de API/transporte/modelo). Campos BI → missing/invalid; raw do Turno 1 **preservado**; MCs ainda assim recolhidos quando possível.

### 8.3 `technical_failure` (categoria separada)

Se erro de API / transporte / modelo: marcar `technical_failure` (categoria **separada**); **não** pretender que é um caso de format-invalid. Em falha técnica, não aplicar a lógica de «retry de formato» como se fosse resposta malformada do respondente.

---

## 9. Mapeamento condições (metadados — não no prompt)

| Cond | Ciclo (meta) | Valência (meta) | ciclo_id | valencia_id |
|------|--------------|-----------------|----------|-------------|
| C1 | Expansão | positiva | EC_EXP | VAL_POS |
| C2 | Expansão | negativa | EC_EXP | VAL_NEG |
| C3 | Contração | positiva | EC_CON | VAL_POS |
| C4 | Contração | negativa | EC_CON | VAL_NEG |

Estes rótulos servem para atribuição/análise. **Não** devem aparecer no system prompt nem como labels explícitos no user prompt.

---

## 10. Campos a armazenar por execução

Campos base (inalterados na lógica):

- `model`, `version`, `datetime`  
- `params` (temperature, top_p, reasoning.effort, max_output_tokens, etc. — ver secção 11)  
- `persona_id`, `condicao`, `ciclo_id`, `valencia_id`  
- `system_prompt` completo  
- `turn1_user` completo, `turn2_user` completo  
- `turn1_raw`, `turn2_raw`  
- `turn1_parse_status`, `turn2_parse_status`  
- `turn1_retry_flag`, `turn2_retry_flag`  
- `technical_failure` (se aplicável)  
- campos parseados: BI1–BI4 (do Turno 1); MC_FRAME, CE1–CE3 (do Turno 2)

Campos de auditoria **adicionais (v1.3 FINAL):**

- `turn1_response_id`, `turn2_response_id`  
- `api_model_returned`  
- `service_tier`  
- `turn1_finish_status` / `turn2_finish_status` (ou `finish_reason` equivalente)  
- `input_tokens_turn1`, `output_tokens_turn1`, `input_tokens_turn2`, `output_tokens_turn2`  
- `schema_version` (ex.: `schema_v1`) e/ou `schema_hash`

### Outputs planeados (nomes scaffold — sem inventar dados)

- `raw_responses.csv` — imutável após recolha  
- `analysis_dataset.csv` — derivado, transforms reproduzíveis  

---

## 11. Modelo e parâmetros (LOCKED v1.5 FINAL — iguais a v1.3.1/v1.3/v1.4)

| Configuração | Valor |
|--------------|-------|
| API | OpenAI Responses API |
| Modelo | gpt-5.6-luna |
| Reasoning mode | standard |
| Reasoning effort | none |
| Temperature | 1.0 |
| top_p | 1.0 |
| Tools | nenhum |
| max_output_tokens | 128 |
| Structured Output | JSON Schema strict |
| Conversação | previous_response_id |
| store | true |
| System instructions | reenviadas nos dois turnos |

A recolha será realizada através da OpenAI Responses API, utilizando o modelo gpt-5.6-luna, em modo de raciocínio standard e com reasoning.effort = none. A temperatura será fixada em 1,0, mantendo-se top_p = 1,0. Não serão disponibilizadas ferramentas externas ao modelo. A saída será restringida através de Structured Outputs com JSON Schema em modo strict e um limite máximo de 128 tokens de saída por turno. O Turno 2 será encadeado ao Turno 1 através de previous_response_id, sendo o mesmo system prompt enviado explicitamente em ambos os pedidos.

**Nota:** alterar temperature **ou** top_p, não ambos — neste lock, `temperature = 1.0` e `top_p = 1.0` (default).

Configuração também em `stimuli_locked.json` → `api_config`.

---

## 11b. Structured Output schemas (`schema_v1`)

Dois JSON Schemas em modo **strict**, versão `schema_v1` (ver `stimuli_locked.json` → `schemas`):

**Turno 1** — chaves `BI1`–`BI4`, inteiros 1–7; `additionalProperties: false`.

**Turno 2** — `MC_FRAME` inteiro 1–7; `CE1`–`CE3` inteiros 1–5; `additionalProperties: false`.

Em cada execução raw: armazenar `schema_version` (ex. `schema_v1`) e/ou `schema_hash` (SHA-256 canónico dos dois schemas). Helpers em `build_prompts_N800.py` (`get_schemas`, `schema_hash`).

---

## 12. Gerador de prompts (uso)

```bash
cd /workspace/personas-sinteticas
python3 build_prompts_N800.py --persona P037          # dry-run: system + turn1 + turn2
python3 build_prompts_N800.py --persona P037 --out prompts_out
python3 build_prompts_N800.py --all --out prompts_out  # N=800; sem API
python3 build_prompts_N800.py --examples               # 1º de C1–C4 (ambos turnos)
python3 build_prompts_N800.py --show-config            # api_config + schema_v1 (sem API)
```

Exemplos documentais renderizados: `exemplos_prompts_C1_C4.md`.

---

## 13. Distinção sintético vs empírico

| Camada | Natureza |
|--------|----------|
| Personas N=800 | **Sintéticas** (calibradas a margens oficiais; não são inquérito humano) |
| Randomização / estímulos / prompts | **Desenho experimental** |
| Respostas LLM (futuras) | **Sintéticas** — proxy experimental; **não** substituem amostra empírica humana |
| Fontes de calibração (Eurostat, ISFF, HFCS, etc.) | **Empíricas** (apenas para construção das margens das personas) |

---

## 14. Próximos passos (fora deste lock)

1. Aprovar este `PROTOCOLO_LLM_LOCKED.md` (**v1.6 FINAL** — Stimuli v1.1 EC/CH/VAL + MC_FRAME approach/avoidance + modelo luna + Responses API)  
2. **Piloto técnico** com ~8 personas **fictícias fora** do N=800 (validar JSON Turno 1 → persistir → Turno 2 com `previous_response_id` + reenvio de `instructions`, Structured Outputs, retries literais, política T2 se T1 inválido, `technical_failure`, campos de auditoria)  
3. Só então: recolher N=800  

Ver também: `PIPELINE.md`, `BASE_EXPERIMENTAL_N800.md`, `RANDOMIZACAO_N800.md`.
