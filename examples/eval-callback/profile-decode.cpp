// profile-decode: time each decode step (wall clock + GPU timeline via CUDA
// driver-API events) to decompose per-token latency into CPU-active vs
// GPU-active vs gap. Uses dlopen(libcuda) so it builds as plain C++.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include <dlfcn.h>
#include <chrono>
#include <vector>
#include <sys/resource.h>

using clk = std::chrono::steady_clock;

typedef int CUresult;
typedef void* CUevent;
typedef void* CUstream;
static CUresult (*cuEventCreate)(CUevent*, unsigned int);
static CUresult (*cuEventRecord)(CUevent, CUstream);
static CUresult (*cuEventSynchronize)(CUevent);
static CUresult (*cuEventElapsedTime)(float*, CUevent, CUevent);
static CUresult (*cuEventDestroy)(CUevent);

int main(int argc, char ** argv) {
    void * h = dlopen("libcuda.so.1", RTLD_NOW);
    if (!h) { LOG_ERR("dlopen libcuda failed\n"); return 1; }
    *(void**)&cuEventCreate = dlsym(h, "cuEventCreate");
    *(void**)&cuEventRecord = dlsym(h, "cuEventRecord");
    *(void**)&cuEventSynchronize = dlsym(h, "cuEventSynchronize");
    *(void**)&cuEventElapsedTime = dlsym(h, "cuEventElapsedTime");
    *(void**)&cuEventDestroy = dlsym(h, "cuEventDestroy");
    if (!cuEventCreate || !cuEventRecord || !cuEventSynchronize || !cuEventElapsedTime) {
        LOG_ERR("missing driver API symbol\n"); return 1;
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    int n_gen = 256;
    if (const char * s = getenv("PROF_N_GEN")) n_gen = atoi(s);

    llama_backend_init();
    llama_numa_init(params.numa);
    params.warmup = false;

    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx   = init->context();
    if (!model || !ctx) { LOG_ERR("failed to init\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens =
        common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);

    llama_batch batch = llama_batch_init(2048, 0, 1);
    batch.n_tokens = (int32_t) tokens.size();
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = (int32_t) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    llama_decode(ctx, batch);

    CUevent e0, e1;
    cuEventCreate(&e0, 0);
    cuEventCreate(&e1, 0);

    llama_sampler * smpl = llama_sampler_init_greedy();
    int32_t pos = (int32_t) tokens.size();
    double wall_sum = 0, gpu_sum = 0, cpu_sum = 0;
    std::vector<double> wall_per;
    int cnt = 0;
    struct rusage ru0, ru1;
    getrusage(RUSAGE_SELF, &ru0);
    for (int g = 0; g < n_gen; g++) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        batch.n_tokens = 1;
        batch.token[0] = id;
        batch.pos[0] = pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        getrusage(RUSAGE_SELF, &ru0);
        cuEventRecord(e0, 0);
        auto t0 = clk::now();
        llama_decode(ctx, batch);
        auto t1 = clk::now();
        cuEventRecord(e1, 0);
        cuEventSynchronize(e1);
        getrusage(RUSAGE_SELF, &ru1);

        double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
        float gms = 0;
        cuEventElapsedTime(&gms, e0, e1);
        double cpu = (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec) * 1000.0
                   + (ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) / 1000.0
                   + (ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec) * 1000.0
                   + (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec) / 1000.0;
        wall_sum += wall; gpu_sum += gms; cpu_sum += cpu;
        wall_per.push_back(wall);
        cnt++;
        pos++;
    }
    llama_sampler_free(smpl);
    llama_batch_free(batch);
    cuEventDestroy(e0);
    cuEventDestroy(e1);

    double wall = wall_sum/cnt, gpu = gpu_sum/cnt, cpu = cpu_sum/cnt, gap = (wall_sum - gpu_sum)/cnt;
    LOG_INF("decode steps = %d\n", cnt);
    LOG_INF("mean wall = %.3f ms/token  (%.1f tok/s)\n", wall, 1000.0/wall);
    LOG_INF("mean GPU  = %.3f ms/token  (GPU stream busy incl. gaps waiting for CPU submit)\n", gpu);
    LOG_INF("mean CPU  = %.3f ms/token  (sum of thread busy time; /6 = avg busy cores)\n", cpu);
    LOG_INF("avg busy CPU cores = %.2f of 6\n", cpu/wall);
    LOG_INF("GPU/total  = %.1f%%\n", 100.0*gpu/wall);
    // chunked rate: report first and last 1000 tokens to detect degradation
    if (cnt > 2000) {
        double w0 = 0, w1 = 0;
        for (int i = 0; i < 1000; i++) w0 += wall_per[i];
        for (int i = cnt - 1000; i < cnt; i++) w1 += wall_per[i];
        LOG_INF("first 1000: %.2f tok/s, last 1000: %.2f tok/s\n", 1000000.0/w0, 1000000.0/w1);
    }
    llama_backend_free();
    return 0;
}
