# Prompts da dissertação: enquadramento da mensagem e ciclo económico no crédito à habitação

Ricardo Franco, Mestrado em Gestão de Marketing, IPAM Lisboa (2026).

Material para consulta e replicação do estudo experimental 2×2 (valência da mensagem × fase do ciclo económico), com N = 800 personas sintéticas e o modelo gpt-5.6-luna (OpenAI Responses API). Protocolo LLM v1.6 FINAL; estímulos v1.1.

| Ficheiro | Conteúdo |
|---|---|
| `resultado.csv` | Registo original da recolha (21/09/2026). Para cada uma das 800 personas: data e hora da chamada, parâmetros, prompt de sistema (`system_prompt`), texto integral do Turno 1 (`turn1_user`) e do Turno 2 (`turn2_user`) e respostas do modelo. |
| `PROTOCOLO_LLM_LOCKED.md` | Protocolo fechado antes da recolha: prompt de sistema, estrutura dos dois turnos, parâmetros da API e regras de nova tentativa. |
| `stimuli_locked.json` | Estímulos (notícias, contexto do crédito, slogans), itens BI, MC_FRAME e CE, esquemas JSON de resposta e mensagens literais de nova tentativa. |
| `run_recolha_N800.py` | Script de execução da recolha (requer `OPENAI_API_KEY` no ambiente). |

Parâmetros: temperature 1.0, top_p 1.0, reasoning effort none, max_output_tokens 128, Structured Outputs (JSON Schema strict). Turno 1 mede a intenção (BI); Turno 2, na mesma conversa (`previous_response_id`) e sem reenviar os estímulos, mede MC_FRAME e CE.

Todos os dados são sintéticos; não há dados de respondentes humanos.
