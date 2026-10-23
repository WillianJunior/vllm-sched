#!/usr/bin/env python3
"""
Utilitário para converter métricas de execução do AIPerf Multi-Turn
para o formato JSON padronizado do projeto (compatível com os scripts de análise e gráficos).
"""

import argparse
import json
import os
import sys

def convert_aiperf_results(artifact_dir, output_file=None, model_id_fallback=None, concurrency_fallback=None, request_rate_fallback=None):
    aiperf_file = os.path.join(artifact_dir, 'profile_export_aiperf.json')
    if not os.path.exists(aiperf_file):
        # Procura recursivamente dentro de artifact_dir caso o AIPerf tenha salvo em subdiretorio
        for root, dirs, files in os.walk(artifact_dir):
            if 'profile_export_aiperf.json' in files:
                aiperf_file = os.path.join(root, 'profile_export_aiperf.json')
                artifact_dir = root
                break

    server_file = os.path.join(artifact_dir, 'server_metrics_export.json')
    profile_jsonl = os.path.join(artifact_dir, 'profile_export.jsonl')

    if not os.path.exists(aiperf_file):
        raise FileNotFoundError(f"Arquivo de perfil do AIPerf não encontrado em: {aiperf_file}")

    with open(aiperf_file, 'r', encoding='utf-8') as f:
        d = json.load(f)

    server_metrics = {}
    if os.path.exists(server_file):
        try:
            with open(server_file, 'r', encoding='utf-8') as f:
                s_data = json.load(f)
                server_metrics = s_data.get('metrics', {})
        except Exception as e:
            print(f"Aviso ao ler server_metrics_export.json: {e}", file=sys.stderr)

    def get_server_total(metric_name):
        m = server_metrics.get(metric_name, {})
        series = m.get('series', [])
        if series and 'stats' in series[0]:
            return series[0]['stats'].get('total', 0.0)
        return 0.0

    completed = int(d.get('completed_request_count', {}).get('avg', 0))
    failed = int(d.get('error_request_count', {}).get('avg', 0))
    model_id = d.get('input_config', {}).get('model', model_id_fallback or 'unknown_model')
    if model_id == 'unknown_model':
        items = d.get('input_config', {}).get('models', {}).get('items', [])
        if items and 'name' in items[0]:
            model_id = items[0]['name']

    concurrency = d.get('input_config', {}).get('concurrency', concurrency_fallback or 0)
    request_rate = d.get('input_config', {}).get('request_rate', request_rate_fallback or 0.0)
    if not request_rate:
        phases = d.get('input_config', {}).get('phases', [])
        if phases and 'rate' in phases[0]:
            request_rate = float(phases[0]['rate'])

    itl = d.get('inter_token_latency', {})
    ttft = d.get('time_to_first_token', {})
    req_lat = d.get('request_latency', {})
    dec_dur = d.get('decode_duration', {})
    tt2t = d.get('time_to_second_token', {})

    cached_tokens = get_server_total('vllm:prompt_tokens_cached')
    total_prompt_tokens = get_server_total('vllm:prompt_tokens')
    preemptions = get_server_total('vllm:num_preemptions')
    cache_hit_rate = (cached_tokens / total_prompt_tokens * 100.0) if total_prompt_tokens > 0 else 0.0

    # Cálculo do Goodput Estrito baseado no max(ITL) <= 60ms por turno/requisição
    good_strict_count = 0
    total_strict_reqs = 0

    if os.path.exists(profile_jsonl):
        try:
            with open(profile_jsonl, 'r', encoding='utf-8') as f_jsonl:
                for line in f_jsonl:
                    if not line.strip():
                        continue
                    req_obj = json.loads(line)
                    itls = req_obj.get('metrics', {}).get('inter_chunk_latency', {}).get('value', [])
                    if itls:
                        total_strict_reqs += 1
                        # Critério estrito: se 1 único token passar de 60ms, a requisição inteira perde o Goodput
                        if max(itls) <= 60.0:
                            good_strict_count += 1
        except Exception as e:
            print(f"Aviso ao processar profile_export.jsonl para goodput estrito: {e}", file=sys.stderr)

    duration = d.get('benchmark_duration', {}).get('avg', 0.0)
    strict_good_fraction = (good_strict_count / total_strict_reqs) if total_strict_reqs > 0 else 0.0
    strict_goodput = (good_strict_count / duration) if duration > 0 else 0.0

    aiperf_fraction = d.get('good_request_fraction', {}).get('avg', 0.0)
    aiperf_goodput = d.get('goodput', {}).get('avg', 0.0)

    print(f"\n[GOODPUT] Estrito (MAX ITL <= 60ms): {good_strict_count}/{total_strict_reqs} ({strict_good_fraction*100:.2f}%) | {strict_goodput:.3f} req/s")
    print(f"[GOODPUT] AIPerf Padrao (MEDIA ITL <= 60ms): {aiperf_fraction*100:.2f}% | {aiperf_goodput:.3f} req/s\n")

    legacy_result = {
        'date': d.get('start_time', ''),
        'endpoint_type': 'vllm_aiperf_multiturn',
        'backend': 'vllm',
        'label': None,
        'model_id': model_id,
        'tokenizer_id': model_id,
        'num_prompts': completed + failed,
        'concurrency': concurrency,
        'request_rate': request_rate,
        'duration': duration,
        'completed': completed,
        'failed': failed,
        'total_input_tokens': int(d.get('total_usage_prompt_tokens', {}).get('avg', 0)),
        'total_output_tokens': int(d.get('total_usage_completion_tokens', {}).get('avg', 0)),
        'request_throughput': d.get('request_throughput', {}).get('avg', 0.0),
        'request_goodput': strict_goodput,
        'good_request_fraction': strict_good_fraction,
        'request_goodput_strict': strict_goodput,
        'good_request_fraction_strict': strict_good_fraction,
        'aiperf_request_goodput_mean': aiperf_goodput,
        'aiperf_good_request_fraction_mean': aiperf_fraction,
        'output_throughput': d.get('output_token_throughput', {}).get('avg', 0.0),
        'total_token_throughput': d.get('total_token_throughput', {}).get('avg', 0.0),

        # Métricas estendidas Multi-Turn e GPU
        'tokens_in_flight': d.get('tokens_in_flight', {}).get('avg', 0.0),
        'effective_concurrency': d.get('effective_concurrency', {}).get('avg', 0.0),
        'effective_decode_concurrency': d.get('effective_decode_concurrency', {}).get('avg', 0.0),
        'effective_prefill_concurrency': d.get('effective_prefill_concurrency', {}).get('avg', 0.0),
        'prefix_cache_hit_rate_perc': cache_hit_rate,
        'prompt_tokens_cached': int(cached_tokens),
        'num_preemptions': int(preemptions),

        # TTFT
        'mean_ttft_ms': ttft.get('avg', 0.0),
        'median_ttft_ms': ttft.get('p50', 0.0),
        'std_ttft_ms': ttft.get('std', 0.0),
        'p50_ttft_ms': ttft.get('p50', 0.0),
        'p75_ttft_ms': ttft.get('p75', 0.0),
        'p90_ttft_ms': ttft.get('p90', 0.0),
        'p99_ttft_ms': ttft.get('p99', 0.0),

        # TPOT / ITL
        'mean_tpot_ms': itl.get('avg', 0.0),
        'median_tpot_ms': itl.get('p50', 0.0),
        'std_tpot_ms': itl.get('std', 0.0),
        'p50_tpot_ms': itl.get('p50', 0.0),
        'p75_tpot_ms': itl.get('p75', 0.0),
        'p90_tpot_ms': itl.get('p90', 0.0),
        'p99_tpot_ms': itl.get('p99', 0.0),

        'mean_itl_ms': itl.get('avg', 0.0),
        'median_itl_ms': itl.get('p50', 0.0),
        'std_itl_ms': itl.get('std', 0.0),
        'p50_itl_ms': itl.get('p50', 0.0),
        'p75_itl_ms': itl.get('p75', 0.0),
        'p90_itl_ms': itl.get('p90', 0.0),
        'p99_itl_ms': itl.get('p99', 0.0),

        # Latência E2E
        'mean_e2el_ms': req_lat.get('avg', 0.0),
        'median_e2el_ms': req_lat.get('p50', 0.0),
        'std_e2el_ms': req_lat.get('std', 0.0),
        'p50_e2el_ms': req_lat.get('p50', 0.0),
        'p75_e2el_ms': req_lat.get('p75', 0.0),
        'p90_e2el_ms': req_lat.get('p90', 0.0),
        'p99_e2el_ms': req_lat.get('p99', 0.0),

        # Decoding
        'mean_decode_duration_ms': dec_dur.get('avg', 0.0),
        'p50_decode_duration_ms': dec_dur.get('p50', 0.0),
        'p90_decode_duration_ms': dec_dur.get('p90', 0.0),
        'p99_decode_duration_ms': dec_dur.get('p99', 0.0),
        'mean_tt2t_ms': tt2t.get('avg', 0.0)
    }

    if not output_file:
        dir_name = os.path.basename(os.path.normpath(artifact_dir))
        output_file = os.path.join(os.path.dirname(os.path.normpath(artifact_dir)) or '.', f"{dir_name}_normalized.json")

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(legacy_result, f, indent=4)

    print(f"Métricas salvas com sucesso no formato padrão em {output_file}")
    return legacy_result

def main():
    parser = argparse.ArgumentParser(description="Converte artefatos do AIPerf para formato padrão do benchmark vLLM.")
    parser.add_argument("--artifact-dir", required=True, help="Diretório onde estão os artefatos do AIPerf (profile_export_aiperf.json, etc.)")
    parser.add_argument("--output-file", default=None, help="Caminho do arquivo JSON de saída")
    parser.add_argument("--model", default=None, help="Fallback para nome/caminho do modelo")
    parser.add_argument("--concurrency", type=int, default=None, help="Fallback para concorrência")
    parser.add_argument("--request-rate", type=float, default=None, help="Fallback para taxa de requisições")

    args = parser.parse_args()
    convert_aiperf_results(args.artifact_dir, args.output_file, args.model, args.concurrency, args.request_rate)

if __name__ == '__main__':
    main()
