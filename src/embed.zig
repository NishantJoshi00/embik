const llama = @cImport({
    @cInclude("llama.h");
});

const std = @import("std");

// const fn (c_uint, [*c]const u8, ?*anyopaque) void
fn empty(_: c_uint, _: [*c]const u8, _: ?*anyopaque) callconv(.c) void {} 

pub const EmbeddingEngine = struct {
    model: ?*llama.llama_model,
    context: ?*llama.llama_context,
    vocab: ?*const llama.llama_vocab,
    n_embd: i32,

    const Self = @This();

    pub fn init(model_path: []const u8) !Self {
        // Load the model

        llama.llama_log_set(empty, null);
        const model_params = llama.llama_model_default_params();
        const model = llama.llama_load_model_from_file(model_path.ptr, model_params);

        if (model == null) {
            return error.ModelLoadFailed;
        }

        const n_embd = llama.llama_n_embd(model);

        // Create context parameters
        var ctx_params = llama.llama_context_default_params();
        ctx_params.embeddings = true;
        ctx_params.pooling_type = llama.LLAMA_POOLING_TYPE_MEAN;
        const context = llama.llama_new_context_with_model(model, ctx_params);

        if (context == null) {
            llama.llama_free_model(model);
            return error.ContextCreationFailed;
        }

        // Get vocab
        const vocab = llama.llama_model_get_vocab(model);

        return Self{
            .model = model,
            .context = context,
            .vocab = vocab,
            .n_embd = n_embd,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.context != null) {
            llama.llama_free(self.context);
        }
        if (self.model != null) {
            llama.llama_free_model(self.model);
        }
    }

    pub fn embed(self: *Self, allocator: std.mem.Allocator, text: []const u8) ![]f32 {
        // Tokenize the text
        const max_tokens = 512;
        var tokens: [max_tokens]llama.llama_token = undefined;
        const n_tokens = llama.llama_tokenize(self.vocab, text.ptr, @intCast(text.len), tokens[0..].ptr, max_tokens, true, true);

        if (n_tokens < 0) {
            return error.TokenizationFailed;
        }

        // Create batch for processing
        var batch = llama.llama_batch_init(@intCast(n_tokens), 0, 1);
        defer llama.llama_batch_free(batch);

        // Add tokens to batch
        batch.n_tokens = @intCast(n_tokens);
        for (0..@intCast(n_tokens)) |i| {
            batch.token[i] = tokens[i];
            batch.pos[i] = @intCast(i);
            batch.seq_id[i][0] = 0;
            batch.n_seq_id[i] = 1;
            batch.logits[i] = 1;  // Enable logits for all tokens for embeddings
        }

        // Decode tokens
        if (llama.llama_decode(self.context, batch) != 0) {
            return error.DecodeFailed;
        }

        // Get embeddings
        const embeddings = llama.llama_get_embeddings_seq(self.context, 0);
        if (embeddings == null) {
            return error.EmbeddingExtractionFailed;
        }

        // Copy embeddings to owned slice
        const result = try allocator.alloc(f32, @intCast(self.n_embd));
        for (0..@intCast(self.n_embd)) |i| {
            result[i] = embeddings[i];
        }

        return result;
    }
};
