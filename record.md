baseline:
step 1
(RolloutManager pid=310428) [2026-06-29 12:28:30] rollout.py:1244 - perf 0: {'rollout/segments_per_trajectory_mean': 2.0625, 'rollout/segments_per_trajectory_max': 15.0, 'rollout/segments_per_trajectory_min': 1.0, 'rollout/num_trajectories': 512.0, 'rollout/num_segments': 1056.0, 'rollout/dropped_tail_groups': 0.0, 'rollout/round_type': 0.0, 'rollout/long_prompt_queue_size': 16.0, 'rollout/raw_reward_trajectory_mean': 0.677734375, 'rollout/response_len/mean': 2069.723484848485, 'rollout/response_len/median': 1727.5, 'rollout/response_len/max': 10353, 'rollout/response_len/min': 34, 'rollout/zero_std/count_0.0': 2, 'rollout/zero_std/count_1.0': 4, 'rollout/spec_accept_rate': 0.0, 'rollout/spec_accept_length': 0.0, 'rollout/prefix_cache_hit_rate': 0.0, 'rollout/avg_cached_tokens_per_sample': 0.0, 'rollout/repetition_frac': 0.0, 'rollout/truncated_ratio': 0.00946969696969697, 'perf/rollout_time': 636.0984523296356, 'perf/tokens_per_gpu_per_sec': 938.2262255383184, 'perf/longest_sample_tokens_per_sec': 19.814542786323933, 'perf/effective_tokens_per_gpu_per_sec': 858.997530962147, 'perf/longest_effective_sample_tokens_per_sec': 16.275782407712764}

(RolloutManager pid=310428) [2026-06-29 12:43:03] rollout.py:1244 - perf 1: {'rollout/segments_per_trajectory_mean': 2.24609375, 'rollout/segments_per_trajectory_max': 19.0, 'rollout/segments_per_trajectory_min': 1.0, 'rollout/num_trajectories': 512.0, 'rollout/num_segments': 1150.0, 'rollout/dropped_tail_groups': 0.0, 'rollout/round_type': 0.0, 'rollout/long_prompt_queue_size': 32.0, 'rollout/raw_reward_trajectory_mean': 0.658203125, 'rollout/response_len/mean': 2395.8017391304347, 'rollout/response_len/median': 1860.5, 'rollout/response_len/max': 13349, 'rollout/response_len/min': 25, 'rollout/zero_std/count_1.0': 4, 'rollout/zero_std/count_0.0': 1, 'rollout/spec_accept_rate': 0.0, 'rollout/spec_accept_length': 0.0, 'rollout/prefix_cache_hit_rate': 0.0, 'rollout/avg_cached_tokens_per_sample': 0.0, 'rollout/repetition_frac': 0.0, 'rollout/truncated_ratio': 0.029565217391304348, 'perf/rollout_time': 872.2481963634491, 'perf/tokens_per_gpu_per_sec': 841.3966954128216, 'perf/longest_sample_tokens_per_sec': 15.494443045392734, 'perf/effective_tokens_per_gpu_per_sec': 789.6754649326821, 'perf/longest_effective_sample_tokens_per_sec': 15.304130241431565}

fp8+consistent_hashing:
step1:
(RolloutManager pid=34645) [2026-06-29 13:35:02] rollout.py:1244 - perf 0: {'rollout/segments_per_trajectory_mean': 2.27734375, 'rollout/segments_per_trajectory_max': 47.0, 'rollout/segments_per_trajectory_min': 1.0, 'rollout/num_trajectories': 512.0, 'rollout/num_segments': 1166.0, 'rollout/dropped_tail_groups': 0.0, 'rollout/round_type': 0.0, 'rollout/long_prompt_queue_size': 16.0, 'rollout/raw_reward_trajectory_mean': 0.6484375, 'rollout/response_len/mean': 2164.4150943396226, 'rollout/response_len/median': 1749.0, 'rollout/response_len/max': 12644, 'rollout/response_len/min': 36, 'rollout/zero_std/count_1.0': 3, 'rollout/zero_std/count_0.0': 3, 'rollout/spec_accept_rate': 0.0, 'rollout/spec_accept_length': 0.0, 'rollout/prefix_cache_hit_rate': 0.0, 'rollout/avg_cached_tokens_per_sample': 0.0, 'rollout/repetition_frac': 0.0, 'rollout/truncated_ratio': 0.012006861063464836, 'perf/rollout_time': 637.1364121437073, 'perf/tokens_per_gpu_per_sec': 1110.3136416576986, 'perf/longest_sample_tokens_per_sec': 24.48769165068679, 'perf/effective_tokens_per_gpu_per_sec': 990.2541872896338, 'perf/longest_effective_sample_tokens_per_sec': 19.845043791262903}

step2
(RolloutManager pid=34645) [2026-06-29 13:47:56] rollout.py:1244 - perf 1: {'rollout/segments_per_trajectory_mean': 2.39453125, 'rollout/segments_per_trajectory_max': 25.0, 'rollout/segments_per_trajectory_min': 1.0, 'rollout/num_trajectories': 512.0, 'rollout/num_segments': 1226.0, 'rollout/dropped_tail_groups': 0.0, 'rollout/round_type': 0.0, 'rollout/long_prompt_queue_size': 32.0, 'rollout/raw_reward_trajectory_mean': 0.6796875, 'rollout/response_len/mean': 2522.300163132137, 'rollout/response_len/median': 1885.0, 'rollout/response_len/max': 10572, 'rollout/response_len/min': 22, 'rollout/zero_std/count_1.0': 8, 'rollout/zero_std/count_0.0': 2, 'rollout/spec_accept_rate': 0.0, 'rollout/spec_accept_length': 0.0, 'rollout/prefix_cache_hit_rate': 0.0, 'rollout/avg_cached_tokens_per_sample': 0.0, 'rollout/repetition_frac': 0.0, 'rollout/truncated_ratio': 0.033442088091353996, 'perf/rollout_time': 773.3105375766754, 'perf/tokens_per_gpu_per_sec': 1059.0502911888705, 'perf/longest_sample_tokens_per_sec': 19.948001805769366, 'perf/effective_tokens_per_gpu_per_sec': 999.7083479847796, 'perf/longest_effective_sample_tokens_per_sec': 13.671092641682467}

fp8+--router-policy cache_aware
step 1:
(RolloutManager pid=221836) [2026-06-29 13:05:09] rollout.py:1244 - perf 0: {'rollout/segments_per_trajectory_mean': 2.21484375, 'rollout/segments_per_trajectory_max': 25.0, 'rollout/segments_per_trajectory_min': 1.0, 'rollout/num_trajectories': 512.0, 'rollout/num_segments': 1134.0, 'rollout/dropped_tail_groups': 0.0, 'rollout/round_type': 0.0, 'rollout/long_prompt_queue_size': 16.0, 'rollout/raw_reward_trajectory_mean': 0.603515625, 'rollout/response_len/mean': 2259.543209876543, 'rollout/response_len/median': 1764.5, 'rollout/response_len/max': 11818, 'rollout/response_len/min': 35, 'rollout/zero_std/count_1.0': 4, 'rollout/zero_std/count_0.0': 3, 'rollout/spec_accept_rate': 0.0, 'rollout/spec_accept_length': 0.0, 'rollout/prefix_cache_hit_rate': 0.0, 'rollout/avg_cached_tokens_per_sample': 0.0, 'rollout/repetition_frac': 0.0, 'rollout/truncated_ratio': 0.014991181657848324, 'perf/rollout_time': 623.7341170310974, 'perf/tokens_per_gpu_per_sec': 1109.9829736674199, 'perf/longest_sample_tokens_per_sec': 24.343706052627187, 'perf/effective_tokens_per_gpu_per_sec': 1027.0089169550151, 'perf/longest_effective_sample_tokens_per_sec': 18.94717585155085}


(RolloutManager pid=221836) [2026-06-29 13:17:37] rollout.py:1244 - perf 1: {'rollout/segments_per_trajectory_mean': 2.31640625, 'rollout/segments_per_trajectory_max': 17.0, 'rollout/segments_per_trajectory_min': 1.0, 'rollout/num_trajectories': 512.0, 'rollout/num_segments': 1186.0, 'rollout/dropped_tail_groups': 0.0, 'rollout/round_type': 0.0, 'rollout/long_prompt_queue_size': 32.0, 'rollout/raw_reward_trajectory_mean': 0.59765625, 'rollout/response_len/mean': 2391.5, 'rollout/response_len/median': 1840.0, 'rollout/response_len/max': 11111, 'rollout/response_len/min': 26, 'rollout/zero_std/count_1.0': 3, 'rollout/zero_std/count_0.0': 2, 'rollout/spec_accept_rate': 0.0, 'rollout/spec_accept_length': 0.0, 'rollout/prefix_cache_hit_rate': 0.0, 'rollout/avg_cached_tokens_per_sample': 0.0, 'rollout/repetition_frac': 0.0, 'rollout/truncated_ratio': 0.02782462057335582, 'perf/rollout_time': 748.1215217113495, 'perf/tokens_per_gpu_per_sec': 1009.2320941026414, 'perf/longest_sample_tokens_per_sec': 22.583229475008554, 'perf/effective_tokens_per_gpu_per_sec': 947.813596349908, 'perf/longest_effective_sample_tokens_per_sec': 14.851865208453392}

2026-07-01 H200x4 Qwen3.5-35B-A3B fp8 + deep_gemm profiling

config:
- script: /root/Dressage/examples/scripts/run_blackbox_qwen3.5_35b_a3b_sync_local_h200x4.sh
- SGLang: fp8, --sglang-moe-runner-backend deep_gemm, EAGLE, 4 engines x 1 H200, local_bwrap 256 slots
- important change: --sglang-mem-fraction-static 0.75
- job: raysubmit_QJuXcqR6bzjBKm1f
- trace dir: /root/Dressage/profiles/sglang_075_abort1s/20260701_050448

stability:
- DeepGEMM cuda graph capture passed. Previous silu_and_mul_masked_post_quant.cuh:319 did not reproduce with mem fraction 0.75.
- Capture started around bs=104 with avail_mem about 38 GB, then completed down to bs=1 on all 4 workers.
- 4 SGLang worker ports 15000/15002/15004/15006 and router 8000 became healthy.

kernel/component trace:
- DECODE trace main buckets:
  - eagle/target verify: about 0.91-1.21 s per sampled trace, largest explicit bucket.
  - mamba/deltarule: about 0.02-0.55 s depending on trace.
  - DeepGEMM fp8 gemm: about 0.11-0.19 s.
  - cudaGraphLaunch: about 0.12-0.21 s.
  - cudaLaunchKernel overhead is still visible: about 0.05-0.21 s.
  - attention and moe reorder/silu/router are smaller than EAGLE verify and Mamba/DeltaRule in these samples.
- EXTEND trace main buckets:
  - long step[EXTEND toks=4096] dominates the top events.
  - mamba/deltarule about 0.29-0.51 s.
  - cudaLaunchKernel about 0.31-0.46 s.
  - DeepGEMM fp8 gemm about 0.11-0.14 s.
- Conclusion: DeepGEMM is active, but current rollout bottleneck is not only MoE GEMM. EAGLE target verify, long EXTEND/prefill, Mamba/DeltaRule, and launch/graph overhead are all first-class contributors.

blackbox/sandbox component result:
- local_bwrap pool started cleanly: total_capacity=256, total_ready=256, total_leased=0.
- Previous 30s sandbox release RPC timeout disappeared in this run: count 0.
- blackbox rollout failed count 0; ServerState.ERROR count 0; 502 count 0.
- perf 0 after full short rollout:
  - rollout_time 731.305 s.
  - tokens_per_gpu_per_sec 868.408.
  - effective_tokens_per_gpu_per_sec 772.356.
  - response_len mean 2080.40, median 1666.5, max 10240.
  - num_trajectories 512, num_segments 1086, long_prompt_queue_size 16.
- perf 0 component profile:
  - blackbox_call_agent mean 172.33 s, p50 152.95 s, p95 362.62 s, max 600.05 s.
  - blackbox_paddock_init mean 170.70 s, p50 206.83 s, p95 410.56 s, max 506.07 s.
  - sandbox_provider_acquire mean 129.35 s, p50 64.58 s, p95 384.96 s, max 451.83 s.
  - sglang_generate_http_s mean 16.28 s, p50 6.74 s, p95 66.36 s, max 172.69 s.
  - blackbox_register_agent mean 23.06 s, p50 20.07 s, p95 60.59 s, max 128.64 s.
- abort_session timeout patch worked:
  - slow blackbox abort_session count 174, avg 1.020 s, p50 1.015 s, max 1.223 s.
  - terminate total count 174, avg 1.596 s, p50 1.031 s, max 8.905 s.
  - provider terminate avg 0.576 s, p50 0.007 s, max 7.889 s.
- After perf 0, one new blackbox rollout failed with 502 on /v1/rollout/register at port 31034. That happened after the measured short rollout and should be investigated separately.
- Conclusion: the old 10s abort bottleneck is fixed, but the current throughput is dominated by blackbox/sandbox lifecycle and SGLang HTTP tail latency, not raw DeepGEMM MoE speed. Provider terminate still has tail latency under concurrency.

open items:
- This run is slower than previous fp8 cache_aware/consistent_hashing records:
  - previous fp8+cache_aware effective perf 0: 1027.01 tokens/GPU/s.
  - previous fp8+consistent_hashing effective perf 0: 990.25 tokens/GPU/s.
  - current fp8+deep_gemm mem 0.75 effective perf 0: 772.36 tokens/GPU/s.
- If final throughput still under expectation, next experiments:
  - reduce session teardown pressure or make blackbox abort fire-and-forget after local release is queued.
  - tune EAGLE verify/draft settings, because DECODE is target-verify heavy.
  - inspect Mamba/DeltaRule kernels separately, because EXTEND is not MoE-bound.

2026-07-01 H200x4 rollback without fast_finalize/reset-throttle, keep HTTP pool

code/config:
- Restored blackbox HTTP client pool limits: max_connections=1024, max_keepalive_connections=256.
- fast_finalize remained disabled/absent: mode=fast_finalize count 0.
- DRESSAGE_LOCAL_BWRAP_RESET_CONCURRENCY was removed from the run script/runtime env.
- DRESSAGE_LOCAL_BWRAP_ACQUIRE_CONCURRENCY=32 and DRESSAGE_BLACKBOX_REGISTER_CONCURRENCY=32 remained in the script; these are acquire/register limits, not reset cleanup throttling.
- script: /root/Dressage/examples/scripts/run_blackbox_qwen3.5_35b_a3b_sync_local_h200x4.sh
- launcher log: /root/Dressage/rollback_keep_pool_20260701_123048.log
- saved ray log: /root/Dressage/rollback_keep_pool_raylogs_20260701_123048.log
- job: raysubmit_q5GfepsUP73uykqV

unit checks before E2E:
- pytest -q tests/test_blackbox_node_supervisor.py tests/test_new_paddock_layers.py tests/blackbox_server/test_server.py -k "abort_clean_session or abort_desynced_session or abort_inflight_session or abort_active_request_or_command or blackbox_terminate_backgrounds_abort_and_release or node_supervisor" => 8 passed, 40 deselected.

perf 0:
- rollout_time 649.309 s.
- tokens_per_gpu_per_sec 987.656.
- effective_tokens_per_gpu_per_sec 899.474.
- reward mean 0.7266.
- truncated_ratio 0.01025.
- response_len mean 2177.21, median 1744, max 13422, min 33.
- num_trajectories 512, num_segments 1073.
- segments_per_trajectory_mean 2.096, max 11.

error/stability stats from saved ray log:
- PoolTimeout 0. This confirms the HTTP pool rollback was the source of the earlier PoolTimeout flood.
- mode=fast_finalize 0.
- silu_and_mul_masked_post_quant 0; Capture cuda graph failed 0.
- 502 Bad Gateway 0.
- blackbox rollout failed 5, all ReadError.
- ReadTimeout 24.
- 404 Not Found 5, corresponding to abort on sessions already gone after the ReadError failures.
- context_overflow 2.

comparison:
- vs best bgterminate_noprofile_20260701_071842: rollout_time 588.627 -> 649.309, slower by 60.682 s (+10.31%); effective 1168.740 -> 899.474 (-23.04%).
- vs fastfinalize_off_20260701_114857: rollout_time 640.588 -> 649.309, slower by 8.721 s (+1.36%); effective 959.060 -> 899.474 (-6.21%).
- vs rollback attempt without HTTP pool: PoolTimeout flood disappeared after restoring 1024/256 pool.

interpretation:
- Restoring the HTTP pool fixed the connection-pool failure mode, but did not restore the previous best throughput.
- The remaining failures are not DeepGEMM/CUDA graph related and not PoolTimeout. They look like blackbox server/session lifecycle races: a rollout request sees ReadError, then abort sees 404 because the session/server side is already gone.
- The run kept running after perf0 and continued occupying GPUs, so it was manually stopped after saving the perf0 log.
- Next target should be session lifecycle hardening around server death/restart and idempotent abort/finalize behavior, not more HTTP pool tuning.

2026-07-01 H200x4 blackbox /abort fast_finalize + httpx poolfix

config:
- script: /root/Dressage/examples/scripts/run_blackbox_qwen3.5_35b_a3b_sync_local_h200x4.sh
- log: /root/Dressage/fastfinalize_poolfix_20260701_083533.log
- report: /root/Dressage/reports/h200_fast_finalize_poolfix_20260701.md
- change: /abort clean inactive session fast path returns mode=fast_finalize; dirty/inflight/desynced/active command still best_effort. Also increase blackbox httpx client pool to avoid PoolTimeout under concurrent cleanup.

result:
- perf 0 rollout_time 677.382 s, tokens_per_gpu_per_sec 1030.322, effective_tokens_per_gpu_per_sec 942.230.
- reward mean 0.7422, truncated_ratio 0.01898, response_len mean 2202.76, median 1715, max 10936.
- abort stats: fast_finalize 758, best_effort 17, PoolTimeout 0, slow abort 50.
- remaining terminate/provider tail: slow terminate 308, total avg 5.025 s, p95 12.924 s, max 14.620 s.
- Compared with bgterminate_noprofile_20260701_071842, this run is slower by 88.755 s (+15.08%) and effective throughput is lower by 19.38%. Fast path works functionally, but this E2E run is dominated by provider reset/release tail and sample/run variance, so it is not a throughput win yet.

next:
- profile provider release/reset in detail.
- split slow abort into client wait/server lock/adapter abort/command terminate.
- gate verbose abort mode logs before longer benchmark.

2026-07-01 H200x4 fast_finalize quiet-log perf0 check

config:
- script: /root/Dressage/examples/scripts/run_blackbox_qwen3.5_35b_a3b_sync_local_h200x4.sh
- ray job: raysubmit_BqLdLgNPvZ8DgfuM
- launcher log: /root/Dressage/fastfinalize_quiet_20260701_090253.log
- saved ray log: /root/Dressage/fastfinalize_quiet_raylogs_20260701_090253.log
- change: keep fast_finalize logic, downgrade per-session abort completion and server abort decision logs to debug.

perf 0:
- rollout_time 597.076 s.
- tokens_per_gpu_per_sec 1023.637.
- effective_tokens_per_gpu_per_sec 940.366.
- reward mean 0.7559.
- truncated_ratio 0.01246.
- response_len mean 2153.29, median 1665, max 12497, min 10.
- num_trajectories 512, num_segments 1043.

log/error stats:
- saved ray log size 55 KB, 367 lines.
- blackbox abort_session completed 0, mode=fast_finalize 0, slow abort 0, slow terminate 0.
- PoolTimeout 0.
- blackbox rollout failed 14, retry 5.
- ReadError 10, 502 Bad Gateway 2, ConnectTimeout 4, ReadTimeout 27, RemoteProtocolError 2, 404 Not Found 7.

comparison:
- vs noisy fast_finalize_poolfix_20260701_083533: rollout_time 677.382 -> 597.076, faster by 80.306 s (-11.86%). This supports that the previous 677 s run had large overhead/variance outside fast_finalize itself.
- vs bgterminate_noprofile_20260701_071842: rollout_time 588.627 -> 597.076, slower by 8.449 s (+1.44%). Given this quiet run had 14 rollout failures and 5 group retries, it is close enough to say the earlier +15.08% slowdown was not caused by fast_finalize logic alone.
- This run is not a clean A/B because of ReadError/502/desync/finalize-404/retries. It is useful for disproving “fast_finalize necessarily causes 15% slowdown”, but not enough to estimate exact speedup.

2026-07-01 H200x4 fast_finalize reset-concurrency validation

root cause hypothesis:
- fast_finalize itself does not touch proxy /session/finalize and does not directly delete proxy sessions.
- It removes the previous ~1s abort latency, so hard reset/provider terminate can burst much faster across many slots.
- This can amplify connection churn around blackbox servers / SGLang router, causing early /messages 502/ReadError and cleanup /abort timeout clusters.

code changes:
- dressage/sandbox/local/bwrap/supervisor.py
  - added DRESSAGE_LOCAL_BWRAP_RESET_CONCURRENCY, default 32.
  - wraps _reset_slot with an asyncio.Semaphore so hard/soft slot reset/restart is throttled.
  - health() now reports reset_concurrency.
- examples/scripts/run_blackbox_qwen3.5_35b_a3b_sync_local_h200x4.sh
  - exports DRESSAGE_LOCAL_BWRAP_RESET_CONCURRENCY and passes it through Ray runtime env.
- tests/test_blackbox_node_supervisor.py
  - added concurrent hard reset throttle test.
  - fixed soft reset test to mock blackbox abort and wait async reset semantics.

unit tests:
- pytest -q tests/test_blackbox_node_supervisor.py => 8 passed.
- pytest -q tests/test_new_paddock_layers.py tests/blackbox_server/test_server.py -k "abort_clean_session or abort_desynced_session or abort_inflight_session or abort_active_request_or_command or blackbox_terminate_backgrounds_abort_and_release" => 5 passed.
- py_compile supervisor/paddock/server and bash -n h200 script passed.

E2E:
- script: /root/Dressage/examples/scripts/run_blackbox_qwen3.5_35b_a3b_sync_local_h200x4.sh
- env: DRESSAGE_LOCAL_BWRAP_RESET_CONCURRENCY=32, DRESSAGE_BLACKBOX_TERMINATE_WARN_SEC=999999
- job: raysubmit_FCtmu3yQVRdxexxf
- launcher log: /root/Dressage/fastfinalize_resetlimit_20260701_112654.log
- saved ray log: /root/Dressage/fastfinalize_resetlimit_raylogs_20260701_112654.log

perf 0:
- rollout_time 615.107 s.
- tokens_per_gpu_per_sec 963.542.
- effective_tokens_per_gpu_per_sec 883.110.
- reward mean 0.7637.
- truncated_ratio 0.00766.
- response_len mean 2081.25, median 1649.5, max 10240, min 27.
- num_trajectories 512, num_segments 1044.

error stats:
- blackbox rollout failed 4.
- resubmitting rollout group for retry 1.
- ReadError 2.
- 502 Bad Gateway 2.
- PoolTimeout 0.
- ConnectTimeout 0.
- ReadTimeout 18, mostly after perf0 during final cleanup.
- RemoteProtocolError 1.
- 404 Not Found 2.

comparison with previous quiet fast_finalize:
- quiet before reset throttle: rollout_time 597.076 s, rollout failed 14, retry 5, ReadError 10, 502 2, ReadTimeout 27.
- reset throttle: rollout_time 615.107 s, rollout failed 4, retry 1, ReadError 2, 502 2, ReadTimeout 18.
- The patch reduced trajectory errors/retries substantially, but did not eliminate early /messages 502. The remaining failures likely come from SGLang/router/blackbox readiness or early worker connection churn, not from proxy finalize or fast_finalize directly.

next:
- Add a warmup/readiness barrier before first rollout requests: do not start blackbox rollout until all 4 SGLang engines pass /health and one small /generate or router health probe succeeds.
- Add bounded retry for /messages 502/transport errors only when no committed proxy turns exist for the session.
- Consider a separate abort cleanup concurrency limit; reset throttle reduced final cleanup timeout count but did not fully eliminate batched abort ReadTimeout after perf0.

2026-07-01 H200x4 fast_finalize default-off validation

code/config:
- DRESSAGE_BLACKBOX_ABORT_FAST_FINALIZE defaults to off; this run did not set it.
- /abort therefore used backend best_effort path instead of server-side fast_finalize.
- DRESSAGE_LOCAL_BWRAP_RESET_CONCURRENCY=32 remained enabled from the reset-throttle patch.
- script: /root/Dressage/examples/scripts/run_blackbox_qwen3.5_35b_a3b_sync_local_h200x4.sh
- launcher log: /root/Dressage/fastfinalize_off_20260701_114857.log
- saved ray log: /root/Dressage/fastfinalize_off_raylogs_20260701_114857.log
- job: raysubmit_qNn2qAkJuSrBQAZe

unit checks before E2E:
- pytest -q tests/test_new_paddock_layers.py tests/blackbox_server/test_server.py -k "abort_clean_session or abort_desynced_session or abort_inflight_session or abort_active_request_or_command or blackbox_terminate_backgrounds_abort_and_release" => 6 passed.
- pytest -q tests/test_blackbox_node_supervisor.py => 8 passed.

perf 0:
- rollout_time 640.588 s.
- tokens_per_gpu_per_sec 1052.140.
- effective_tokens_per_gpu_per_sec 959.060.
- reward mean 0.7227.
- truncated_ratio 0.01775.
- response_len mean 2180.522, median 1712, max 12993, min 25.
- num_trajectories 512, num_segments 1127.
- segments_per_trajectory_mean 2.201, max 11.

pre-perf error/crash stats:
- mode=fast_finalize 0; fast_finalize 0.
- silu_and_mul_masked_post_quant 0; Capture cuda graph failed 0.
- blackbox rollout failed 2.
- resubmitting rollout group for retry 1.
- 502 Bad Gateway 1.
- 404 Not Found 1.
- RemoteProtocolError 1.
- ReadError 0; ReadTimeout 0; ConnectTimeout 0; PoolTimeout 0.
- abort_session best-effort failed 1.

post-perf cleanup stats:
- abort_session best-effort failed 27.
- ReadTimeout 24, ConnectTimeout 1, ReadError 2, 404 Not Found 2.
- These are after perf0 and do not affect perf/rollout_time, but show the default backend abort cleanup is still expensive/noisy.

comparison:
- vs record best bgterminate_noprofile_20260701_071842: rollout_time 588.627 -> 640.588, slower by 51.961 s (+8.83%); effective 1168.740 -> 959.060 (-17.94%).
- vs fastfinalize_quiet_20260701_090253: rollout_time 597.076 -> 640.588, slower by 43.512 s (+7.29%).
- vs fastfinalize_resetlimit_20260701_112654: rollout_time 615.107 -> 640.588, slower by 25.481 s (+4.14%).

interpretation:
- Default-off did not return to the 588.6s best run in this sample.
- It did confirm fast_finalize was fully disabled and the previous DeepGEMM CUDA graph crash did not recur.
- The run was not clean: one /messages 502 caused one group retry, and one /session/finalize 404 occurred before perf0. That retry likely accounts for a meaningful part of the +52s gap to the best record.
- The remaining optimization target is not fast_finalize alone; it is the blackbox/proxy session lifecycle around /messages 502, /session/finalize 404, and backend abort cleanup pressure.

2026-07-02 H200x4 LLM proxy profiling refinement

goal:
- The previous profile showed opencode_post_message_s mean 205.66s and proxy_llm_http_s mean 182.48s, but sglang_generate_http_s mean 21.61s was not the same scope.
- Direct subtraction was invalid because one opencode turn contains multiple streamed chat-completion steps.

code changes:
- Kept blackbox server @profile_response for outer /messages timing; it is still useful as an envelope metric.
- Added Dressage proxy chat-completion profile fields returned in streaming usage.profile:
  - chat.total_s, chat.session_lock_wait_s, chat.generation_s, chat.reasoning_parse_s, chat.tool_parse_s, chat.record_step_s, chat.stream_emit_s, chat.stream_chunk_count.
- Passed the same profile dict into GenerationController and SGLangRouterClient.
- Split SGLang client generate into:
  - sglang.client_generate_request_build_s
  - sglang.client_generate_post_s
  - sglang.client_generate_json_parse_s
  - sglang.client_generate_coerce_s
- Blackbox rollout LLM proxy now parses streaming SSE usage.profile and aggregates fields under proxy.llm_*; step_timeline_json also carries these fields.

validation:
- python3 -m py_compile dressage/proxy/server.py dressage/proxy/generation_controller.py dressage/proxy/sglang_client.py blackbox_server/proxy/rollout_llm_proxy.py
- pytest -q tests/blackbox_server/test_rollout_llm_proxy.py tests/blackbox_server/test_opencode_adapter.py => 54 passed.
- pytest -q tests/blackbox_server => 114 passed.
- pytest -q tests/test_blackbox_dispatch.py => 58 passed.
- pytest -q tests/test_proxy.py -k "not non_partial_first_step_waiting_for_resume_binds_generated_epoch" => 115 passed, 1 deselected.

next run should inspect:
- profile/blackbox_adapter_proxy_llm_chat_total_s/*
- profile/blackbox_adapter_proxy_llm_chat_generation_s/*
- profile/blackbox_adapter_proxy_llm_sglang_generate_http_s/*
- profile/blackbox_adapter_proxy_llm_sglang_client_generate_post_s/*
- profile/blackbox_adapter_proxy_llm_chat_reasoning_parse_s/*
- profile/blackbox_adapter_proxy_llm_chat_tool_parse_s/*
- profile/blackbox_adapter_proxy_llm_chat_stream_emit_s/*

interpretation plan:
- If chat.generation_s and sglang.client_generate_post_s dominate, optimize SGLang/MoE/CUDA graph/router.
- If stream_emit_s dominates, optimize OpenAI-compatible streaming and chunk emission/consumption.
- If reasoning_parse_s or tool_parse_s dominates, cache workers and reduce separate_reasoning/parse_function_call HTTP calls.
- If opencode_post_message_s - proxy_llm_chat_total_s is large, optimize opencode-side tool loop/sandbox wait.
