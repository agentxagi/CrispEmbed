#include "pplx_contextual.h"
#include "core/env_gate.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

static bool tensor_to_f32(const ggml_tensor * src, std::vector<float> & dst) {
    if (!src || !src->data) return false;
    const int64_t n = ggml_nelements(src);
    dst.resize((size_t)n);
    if (src->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(src, dst.data(), 0, (size_t)n * sizeof(float));
        return true;
    }
    const auto * traits = ggml_get_type_traits(src->type);
    if (!traits || !traits->to_float) return false;
    std::vector<uint8_t> raw(ggml_nbytes(src));
    ggml_backend_tensor_get(src, raw.data(), 0, raw.size());
    const int64_t row = src->ne[0], rows = n / row;
    for (int64_t r = 0; r < rows; ++r) {
        traits->to_float(raw.data() + (size_t)r * src->nb[1], dst.data() + (size_t)r * row, row);
    }
    return true;
}

struct layer_weights {
    bool full_attn = false;
    ggml_tensor * in_norm = nullptr;
    ggml_tensor * post_norm = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * down = nullptr;
    // full_attention
    ggml_tensor * q_proj = nullptr;
    ggml_tensor * k_proj = nullptr;
    ggml_tensor * v_proj = nullptr;
    ggml_tensor * o_proj = nullptr;
    ggml_tensor * q_norm = nullptr;
    ggml_tensor * k_norm = nullptr;
    // linear_attention (Gated DeltaNet)
    ggml_tensor * qkv_proj = nullptr;
    ggml_tensor * z_proj = nullptr;
    ggml_tensor * a_proj = nullptr;
    ggml_tensor * b_proj = nullptr;
    ggml_tensor * out_proj = nullptr;
    ggml_tensor * conv = nullptr;
    ggml_tensor * neg_exp_alog = nullptr;
    ggml_tensor * dt_bias = nullptr;
    ggml_tensor * gated_norm = nullptr;
};

} // namespace

struct pplx_contextual_ctx {
    int emb_dim = 1024;
    int hidden_size = 1024;
    int intermediate_size = 3584;
    int n_layers = 12;
    int q_heads = 8;
    int kv_heads = 2;
    int head_dim = 256;
    int rotary_dim = 64;
    int lin_heads = 16;
    int lin_head_dim = 128;
    int lin_conv_k = 4;
    int vocab_size = 248320;
    float eps = 1e-6f;
    float rope_theta = 10000000.0f;
    int n_threads = 4;

    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context * weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    ggml_context * const_ctx = nullptr;
    ggml_backend_buffer_t const_buf = nullptr;

    ggml_tensor * embed_tokens = nullptr;
    ggml_tensor * final_norm = nullptr;
    ggml_tensor * l2_eps = nullptr;
    std::vector<layer_weights> layers;
    std::vector<float> proj_w; // [emb_dim, hidden_size] row-major F32 on host for exact float64 dot
    std::vector<uint8_t> graph_meta;
    std::mutex mu;

    ~pplx_contextual_ctx() {
        if (sched) ggml_backend_sched_free(sched);
        if (const_buf) ggml_backend_buffer_free(const_buf);
        if (const_ctx) ggml_free(const_ctx);
        if (weight_buf) ggml_backend_buffer_free(weight_buf);
        if (weight_ctx) ggml_free(weight_ctx);
        if (cpu && cpu != backend) ggml_backend_free(cpu);
        if (backend) ggml_backend_free(backend);
    }
};

extern "C" pplx_contextual_ctx * pplx_contextual_load(const char * model_path, int n_threads) {
    if (!model_path) return nullptr;
    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) return nullptr;
    const std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    if (arch != "pplx_contextual_qwen3_5") {
        core_gguf::free_metadata(meta);
        return nullptr;
    }

    auto * c = new pplx_contextual_ctx();
    c->n_threads = std::max(1, n_threads);
    c->emb_dim = (int)core_gguf::kv_u32(meta, "pplx_contextual.embedding_dim", 1024);
    c->hidden_size = (int)core_gguf::kv_u32(meta, "pplx_contextual.hidden_size", 1024);
    c->intermediate_size = (int)core_gguf::kv_u32(meta, "pplx_contextual.intermediate_size", 3584);
    c->n_layers = (int)core_gguf::kv_u32(meta, "pplx_contextual.num_hidden_layers", 12);
    c->q_heads = (int)core_gguf::kv_u32(meta, "pplx_contextual.num_attention_heads", 8);
    c->kv_heads = (int)core_gguf::kv_u32(meta, "pplx_contextual.num_key_value_heads", 2);
    c->head_dim = (int)core_gguf::kv_u32(meta, "pplx_contextual.head_dim", 256);
    c->lin_heads = (int)core_gguf::kv_u32(meta, "pplx_contextual.linear_num_heads", 16);
    c->lin_head_dim = (int)core_gguf::kv_u32(meta, "pplx_contextual.linear_head_dim", 128);
    c->lin_conv_k = (int)core_gguf::kv_u32(meta, "pplx_contextual.linear_conv_kernel", 4);
    c->vocab_size = (int)core_gguf::kv_u32(meta, "pplx_contextual.vocab_size", 248320);
    c->eps = core_gguf::kv_f32(meta, "pplx_contextual.rms_norm_eps", 1e-6f);
    c->rope_theta = core_gguf::kv_f32(meta, "pplx_contextual.rope_theta", 10000000.0f);
    const float prf = core_gguf::kv_f32(meta, "pplx_contextual.partial_rotary_factor", 0.25f);
    c->rotary_dim = (int)std::lround(c->head_dim * prf);
    core_gguf::free_metadata(meta);

    c->backend = crispasr_init_gpu_backend();
    if (!c->backend) c->backend = ggml_backend_cpu_init();
    if (!c->backend) { delete c; return nullptr; }
    c->cpu = ggml_backend_is_cpu(c->backend) ? c->backend : ggml_backend_cpu_init();
    if (!c->cpu) { delete c; return nullptr; }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, c->backend, "pplx_contextual", wl)) {
        delete c;
        return nullptr;
    }
    c->weight_ctx = wl.ctx;
    c->weight_buf = wl.buf;

    auto req = [&](const std::string & name) -> ggml_tensor * {
        return core_gguf::try_get(wl.tensors, name.c_str());
    };

    c->embed_tokens = req("l.embed_tokens.weight");
    ggml_tensor * raw_final_norm = req("l.norm.weight");
    ggml_tensor * raw_proj = req("contextual_projection.weight");
    if (!c->embed_tokens || !raw_final_norm || !raw_proj || !tensor_to_f32(raw_proj, c->proj_w)) {
        delete c;
        return nullptr;
    }

    const size_t n_const = (size_t)c->n_layers * 12 + 8;
    c->const_ctx = ggml_init({ n_const * ggml_tensor_overhead(), nullptr, true });
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> uploads;
    bool ok = true;

    auto make_vec = [&](ggml_tensor * src, float (*fn)(float) = nullptr) -> ggml_tensor * {
        if (!src) { ok = false; return nullptr; }
        std::vector<float> v;
        if (!tensor_to_f32(src, v)) { ok = false; return nullptr; }
        if (fn) for (float & x : v) x = fn(x);
        ggml_tensor * t = ggml_new_tensor_1d(c->const_ctx, GGML_TYPE_F32, (int64_t)v.size());
        uploads.emplace_back(t, std::move(v));
        return t;
    };
    auto one_plus = [](float w) { return 1.0f + w; };
    auto neg_exp = [](float a) { return -std::exp(a); };

    c->final_norm = make_vec(raw_final_norm, one_plus);
    c->l2_eps = ggml_new_tensor_1d(c->const_ctx, GGML_TYPE_F32, 1);
    uploads.emplace_back(c->l2_eps, std::vector<float>{ 1e-6f });

    const int64_t lin_channels = 3 * (int64_t)c->lin_heads * c->lin_head_dim;
    c->layers.resize((size_t)c->n_layers);
    for (int i = 0; i < c->n_layers; ++i) {
        const std::string pfx = "l.layers." + std::to_string(i) + ".";
        auto & lw = c->layers[(size_t)i];
        lw.full_attn = (req(pfx + "self_attn.q_proj.weight") != nullptr);
        lw.in_norm = make_vec(req(pfx + "input_layernorm.weight"), one_plus);
        lw.post_norm = make_vec(req(pfx + "post_attention_layernorm.weight"), one_plus);
        lw.gate = req(pfx + "mlp.gate_proj.weight");
        lw.up = req(pfx + "mlp.up_proj.weight");
        lw.down = req(pfx + "mlp.down_proj.weight");
        if (!lw.gate || !lw.up || !lw.down) ok = false;
        if (lw.full_attn) {
            lw.q_proj = req(pfx + "self_attn.q_proj.weight");
            lw.k_proj = req(pfx + "self_attn.k_proj.weight");
            lw.v_proj = req(pfx + "self_attn.v_proj.weight");
            lw.o_proj = req(pfx + "self_attn.o_proj.weight");
            lw.q_norm = make_vec(req(pfx + "self_attn.q_norm.weight"), one_plus);
            lw.k_norm = make_vec(req(pfx + "self_attn.k_norm.weight"), one_plus);
            if (!lw.q_proj || !lw.k_proj || !lw.v_proj || !lw.o_proj) ok = false;
        } else {
            lw.qkv_proj = req(pfx + "linear_attn.in_proj_qkv.weight");
            lw.z_proj = req(pfx + "linear_attn.in_proj_z.weight");
            lw.a_proj = req(pfx + "linear_attn.in_proj_a.weight");
            lw.b_proj = req(pfx + "linear_attn.in_proj_b.weight");
            lw.out_proj = req(pfx + "linear_attn.out_proj.weight");
            lw.neg_exp_alog = make_vec(req(pfx + "linear_attn.A_log"), neg_exp);
            lw.dt_bias = make_vec(req(pfx + "linear_attn.dt_bias"));
            lw.gated_norm = make_vec(req(pfx + "linear_attn.norm.weight"));
            ggml_tensor * raw_conv = req(pfx + "linear_attn.conv1d.weight");
            std::vector<float> cv;
            if (!raw_conv || !tensor_to_f32(raw_conv, cv) || (int64_t)cv.size() != c->lin_conv_k * lin_channels) {
                ok = false;
            } else {
                lw.conv = ggml_new_tensor_2d(c->const_ctx, GGML_TYPE_F32, c->lin_conv_k, lin_channels);
                uploads.emplace_back(lw.conv, std::move(cv));
            }
            if (!lw.qkv_proj || !lw.z_proj || !lw.a_proj || !lw.b_proj || !lw.out_proj) ok = false;
        }
    }

    if (!ok) { delete c; return nullptr; }
    c->const_buf = ggml_backend_alloc_ctx_tensors(c->const_ctx, c->backend);
    if (!c->const_buf) { delete c; return nullptr; }
    for (auto & u : uploads) {
        ggml_backend_tensor_set(u.first, u.second.data(), 0, u.second.size() * sizeof(float));
    }

    std::vector<ggml_backend_t> backends = { c->backend };
    if (c->cpu != c->backend) backends.push_back(c->cpu);
    constexpr int kMaxNodes = 16384;
    c->sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), kMaxNodes, false, false);
    c->graph_meta.resize(ggml_tensor_overhead() * kMaxNodes + ggml_graph_overhead_custom(kMaxNodes, false));
    if (!c->sched) { delete c; return nullptr; }

    std::fprintf(stderr, "[pplx_contextual] loaded %d layers (%s), dim=%d, vocab=%d\n",
                 c->n_layers, ggml_backend_name(c->backend), c->emb_dim, c->vocab_size);
    return c;
}

extern "C" void pplx_contextual_free(pplx_contextual_ctx * ctx) {
    delete ctx;
}

extern "C" int pplx_contextual_dim(const pplx_contextual_ctx * ctx) {
    return ctx ? ctx->emb_dim : 0;
}

extern "C" int pplx_contextual_encode_tokens(
    pplx_contextual_ctx * c,
    const int32_t * token_ids,
    int T,
    const int32_t * span_starts,
    const int32_t * span_ends,
    int n_spans,
    int is_query,
    int normalize,
    float * out_vecs) {
    if (!c || !token_ids || T < 1 || T > 32768 || n_spans < 1 || !out_vecs) return 0;
    if (!is_query && (!span_starts || !span_ends)) return 0;
    for (int i = 0; i < T; ++i) {
        if (token_ids[i] < 0 || token_ids[i] >= c->vocab_size) return 0;
    }

    std::lock_guard<std::mutex> lock(c->mu);
    const int D = c->hidden_size, d = c->head_dim, qh = c->q_heads, kh = c->kv_heads;
    const int H = c->lin_heads, W = c->lin_head_dim, K = c->lin_conv_k, group = H * W;
    const int64_t channels = 3 * (int64_t)group;

    constexpr int kMaxNodes = 16384;
    ggml_context * g = ggml_init({ c->graph_meta.size(), c->graph_meta.data(), true });
    if (!g) return 0;
    struct ctx_guard { ggml_context * g; ~ctx_guard() { ggml_free(g); } } guard{ g };
    ggml_cgraph * gf = ggml_new_graph_custom(g, kMaxNodes, false);

    ggml_tensor * tok_in = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_tensor * pos_in = ggml_new_tensor_1d(g, GGML_TYPE_I32, 4 * (int64_t)T);
    ggml_tensor * zero_conv = ggml_new_tensor_2d(g, GGML_TYPE_F32, K - 1, channels);
    ggml_tensor * zero_state = ggml_new_tensor_4d(g, GGML_TYPE_F32, W, W, H, 1);
    ggml_set_input(tok_in);
    ggml_set_input(pos_in);
    ggml_set_input(zero_conv);
    ggml_set_input(zero_state);

    auto mm = [&](ggml_tensor * w, ggml_tensor * x) {
        ggml_tensor * y = ggml_mul_mat(g, w, x);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        return y;
    };
    auto rms = [&](ggml_tensor * t, ggml_tensor * w1) {
        return ggml_mul(g, ggml_rms_norm(g, t, c->eps), w1);
    };
    int sections[GGML_MROPE_SECTIONS] = { 11, 11, 10, 0 };
    auto rope = [&](ggml_tensor * t) {
        return ggml_rope_multi(g, t, pos_in, nullptr, c->rotary_dim, sections, GGML_ROPE_TYPE_IMROPE, 0, c->rope_theta,
                               1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    };
    const float scale = 1.0f / std::sqrt((float)d);

    ggml_tensor * x = ggml_get_rows(g, c->embed_tokens, tok_in); // [D, T] F32
    for (int li = 0; li < c->n_layers; ++li) {
        const auto & l = c->layers[(size_t)li];
        ggml_tensor * xn = rms(x, l.in_norm);
        ggml_tensor * mixed = nullptr;
        if (l.full_attn) {
            ggml_tensor * qfull = mm(l.q_proj, xn); // [2*d*qh, T]
            const size_t fs = sizeof(float);
            ggml_tensor * query = ggml_cont(g, ggml_view_3d(g, qfull, d, qh, T, 2 * (size_t)d * fs, qfull->nb[1], 0));
            ggml_tensor * gate = ggml_cont(g, ggml_view_3d(g, qfull, d, qh, T, 2 * (size_t)d * fs, qfull->nb[1], (size_t)d * fs));
            ggml_tensor * qr = rope(rms(query, l.q_norm));
            ggml_tensor * kr = rope(rms(ggml_reshape_3d(g, mm(l.k_proj, xn), d, kh, T), l.k_norm));
            ggml_tensor * v = ggml_reshape_3d(g, mm(l.v_proj, xn), d, kh, T);

            ggml_tensor * q = ggml_cont(g, ggml_permute(g, qr, 0, 2, 1, 3));     // [d, T, qh]
            ggml_tensor * kall = ggml_cont(g, ggml_permute(g, kr, 0, 2, 1, 3));  // [d, T, kh]
            ggml_tensor * vt = ggml_cont(g, ggml_permute(g, v, 1, 2, 0, 3));     // [T, d, kh]
            ggml_tensor * kq = ggml_mul_mat(g, kall, q);                         // [T, T, qh]
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            // Bidirectional full attention (is_causal = false -> mask = nullptr)
            kq = ggml_soft_max_ext(g, kq, nullptr, scale, 0.0f);
            ggml_tensor * o = ggml_mul_mat(g, vt, kq);                           // [d, T, qh]
            ggml_mul_mat_set_prec(o, GGML_PREC_F32);
            o = ggml_cont(g, ggml_permute(g, o, 0, 2, 1, 3));                    // [d, qh, T]
            ggml_tensor * gated = ggml_mul(g, o, ggml_sigmoid(g, gate));
            mixed = mm(l.o_proj, ggml_reshape_2d(g, gated, qh * d, T));
        } else {
            ggml_tensor * qkv = mm(l.qkv_proj, xn);
            ggml_tensor * z = mm(l.z_proj, xn);
            ggml_tensor * av = mm(l.a_proj, xn);
            ggml_tensor * bv = mm(l.b_proj, xn);
            ggml_tensor * sx = ggml_concat(g, zero_conv, ggml_cont(g, ggml_transpose(g, qkv)), 0);
            ggml_tensor * conv = ggml_ssm_conv(g, ggml_reshape_3d(g, sx, K - 1 + T, channels, 1), l.conv);
            conv = ggml_silu(g, ggml_reshape_2d(g, conv, channels, T));
            const size_t cs = sizeof(float);
            auto part = [&](int idx) {
                return ggml_cont(g, ggml_view_4d(g, conv, W, H, T, 1, (size_t)W * cs, conv->nb[1], conv->nb[1] * T,
                                                 (size_t)idx * group * cs));
            };
            auto l2 = [&](ggml_tensor * t) {
                return ggml_div(g, t, ggml_sqrt(g, ggml_add(g, ggml_sum_rows(g, ggml_sqr(g, t)), c->l2_eps)));
            };
            ggml_tensor * decay = ggml_reshape_4d(
                g, ggml_mul(g, ggml_softplus(g, ggml_add(g, av, l.dt_bias)), l.neg_exp_alog), 1, H, T, 1);
            ggml_tensor * beta = ggml_reshape_4d(g, ggml_sigmoid(g, bv), 1, H, T, 1);
            ggml_tensor * delta = ggml_gated_delta_net(g, l2(part(0)), l2(part(1)), part(2), decay, beta, zero_state, 1);
            ggml_tensor * core = ggml_view_2d(g, delta, W, (int64_t)H * T, (size_t)W * sizeof(float), 0);
            ggml_tensor * zr = ggml_reshape_2d(g, z, W, (int64_t)H * T);
            ggml_tensor * normed = ggml_mul(
                g, ggml_mul(g, ggml_rms_norm(g, ggml_cont(g, core), c->eps), l.gated_norm), ggml_silu(g, zr));
            mixed = mm(l.out_proj, ggml_reshape_2d(g, normed, group, T));
        }
        x = ggml_add(g, x, mixed);
        ggml_tensor * hn = rms(x, l.post_norm);
        ggml_tensor * act = ggml_swiglu_split(g, mm(l.gate, hn), mm(l.up, hn));
        x = ggml_add(g, x, mm(l.down, act));
    }

    ggml_tensor * out_hidden = rms(x, c->final_norm); // [D, T]
    ggml_set_output(out_hidden);
    ggml_build_forward_expand(gf, out_hidden);

    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) return 0;

    ggml_backend_tensor_set(tok_in, token_ids, 0, (size_t)T * sizeof(int32_t));
    std::vector<int32_t> pos_data(4 * (size_t)T);
    for (int axis = 0; axis < 4; ++axis) {
        for (int t = 0; t < T; ++t) pos_data[(size_t)axis * T + t] = t;
    }
    ggml_backend_tensor_set(pos_in, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));
    std::vector<float> zc((size_t)(K - 1) * channels, 0.0f);
    ggml_backend_tensor_set(zero_conv, zc.data(), 0, zc.size() * sizeof(float));
    std::vector<float> zs((size_t)W * W * H, 0.0f);
    ggml_backend_tensor_set(zero_state, zs.data(), 0, zs.size() * sizeof(float));

    if (ggml_backend_is_cpu(c->backend)) ggml_backend_cpu_set_n_threads(c->backend, c->n_threads);
    if (c->cpu && c->cpu != c->backend) ggml_backend_cpu_set_n_threads(c->cpu, c->n_threads);

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) return 0;

    std::vector<float> hidden((size_t)T * D);
    ggml_backend_tensor_get(out_hidden, hidden.data(), 0, hidden.size() * sizeof(float));

    const int E = c->emb_dim;
    std::vector<double> mean((size_t)D, 0.0);
    for (int s = 0; s < n_spans; ++s) {
        const int st = is_query ? 0 : std::max(0, std::min(T, (int)span_starts[s]));
        const int en = is_query ? T : std::max(st, std::min(T, (int)span_ends[s]));
        const int count = std::max(1, en - st);
        std::fill(mean.begin(), mean.end(), 0.0);
        // Float64 accumulation eliminates the float32 cumsum cancellation defect on long documents.
        for (int t = st; t < en; ++t) {
            const float * hrow = hidden.data() + (size_t)t * D;
            for (int j = 0; j < D; ++j) mean[(size_t)j] += (double)hrow[j];
        }
        const double inv = 1.0 / (double)count;
        for (int j = 0; j < D; ++j) mean[(size_t)j] *= inv;

        float * out_row = out_vecs + (size_t)s * E;
        double norm_sq = 0.0;
        for (int r = 0; r < E; ++r) {
            const float * prow = c->proj_w.data() + (size_t)r * D;
            double dot = 0.0;
            for (int j = 0; j < D; ++j) dot += (double)prow[j] * mean[(size_t)j];
            const float qv = std::clamp(std::round(std::tanh((float)dot) * 127.0f), -128.0f, 127.0f);
            out_row[r] = qv;
            norm_sq += (double)qv * (double)qv;
        }
        if (normalize && norm_sq > 1e-24) {
            const double inv_norm = 1.0 / std::sqrt(norm_sq);
            for (int r = 0; r < E; ++r) out_row[r] = (float)((double)out_row[r] * inv_norm);
        }
    }
    return n_spans;
}
