// moe-trace-gen: capture MoE router expert selections for prompt (prefill) AND
// autoregressive generation (decode) tokens. Greedy sampling. Output format is
// identical to moe-trace.cpp: repeated (int32 layer, int32 n_tokens, int32 k,
// then n_tokens*k int32 expert ids) records.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct moe_trace_state {
    FILE * out = nullptr;
    size_t records = 0;
};

static bool moe_trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    const bool match = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    if (ask) return match;
    if (!match) return true;
    auto * st = (moe_trace_state *) user_data;

    const int32_t layer    = atoi(t->name + 13);
    const int32_t k        = (int32_t) t->ne[0];
    const int32_t n_tokens = (int32_t) t->ne[1];

    std::vector<int32_t> ids((size_t) k * n_tokens);
    for (int32_t i = 0; i < n_tokens; i++) {
        ggml_backend_tensor_get(t, ids.data() + (size_t) i * k,
                                (size_t) i * t->nb[1], k * sizeof(int32_t));
    }
    fwrite(&layer, sizeof(int32_t), 1, st->out);
    fwrite(&n_tokens, sizeof(int32_t), 1, st->out);
    fwrite(&k, sizeof(int32_t), 1, st->out);
    fwrite(ids.data(), sizeof(int32_t), ids.size(), st->out);
    st->records++;
    return true;
}

int main(int argc, char ** argv) {
    moe_trace_state st;

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;

    const char * out_path = getenv("MOE_TRACE_OUT");
    if (!out_path) out_path = "moe_trace_gen.bin";
    st.out = fopen(out_path, "wb");
    if (!st.out) { LOG_ERR("cannot open %s\n", out_path); return 1; }

    int n_gen = 256;
    if (const char * s = getenv("MOE_TRACE_N_GEN")) n_gen = atoi(s);

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = moe_trace_cb;
    params.cb_eval_user_data = &st;
    params.warmup            = false;
    params.n_batch           = 2048;

    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx   = init->context();
    if (!model || !ctx) { LOG_ERR("failed to init\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens =
        common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);

    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
    // prefill
    batch.n_tokens = (int32_t) tokens.size();
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = (int32_t) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = true;
    }
    if (llama_decode(ctx, batch)) { LOG_ERR("prefill failed\n"); return 1; }

    // generation: greedy by default, or temperature/top-p sampling via env vars
    float temp = 0.0f;
    if (const char * s = getenv("MOE_TRACE_TEMP")) temp = atof(s);
    uint32_t seed = 42;
    if (const char * s = getenv("MOE_TRACE_SEED")) seed = (uint32_t) atol(s);
    llama_sampler * smpl;
    if (temp <= 0.0f) {
        smpl = llama_sampler_init_greedy();
    } else {
        float topp = 1.0f;
        if (const char * s = getenv("MOE_TRACE_TOPP")) topp = atof(s);
        smpl = llama_sampler_chain_init({});
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topp, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    }
    int32_t pos = (int32_t) tokens.size();
    for (int g = 0; g < n_gen; g++) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        batch.n_tokens = 1;
        batch.token[0]    = id;
        batch.pos[0]      = pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]   = true;
        if (llama_decode(ctx, batch)) { LOG_ERR("decode failed at %d\n", g); break; }
        pos++;
    }
    llama_sampler_free(smpl);
    llama_batch_free(batch);
    fclose(st.out);
    LOG_INF("wrote %zu records to %s\n", st.records, out_path);

    llama_backend_free();
    return 0;
}
