const std = @import("std");
const embik = @import("embik");

const llama = @cImport({
    @cInclude("llama.h");
});

const faiss = @cImport({
    @cInclude("faiss/c_api/IndexFlat_c.h");
    @cInclude("faiss/c_api/Index_c.h");
    @cInclude("faiss/c_api/MetricType_c.h");
});

const rocksdb = @cImport({
    @cInclude("rocksdb/db.h");
});

const EmbeddingEngine = struct {
    model: ?*llama.llama_model,
    context: ?*llama.llama_context,
    vocab: ?*const llama.llama_vocab,
    n_embd: i32,

    const Self = @This();

    pub fn init(model_path: []const u8) !Self {
        // Load the model
        const model_params = llama.llama_model_default_params();
        const model = llama.llama_load_model_from_file(model_path.ptr, model_params);

        if (model == null) {
            return error.ModelLoadFailed;
        }

        const n_embd = llama.llama_n_embd(model);
        std.debug.print("Model loaded with embedding dimension: {d}\n", .{n_embd});

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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize the embedding engine
    var engine = EmbeddingEngine.init("deps/nomic-embed-text-v2-gguf/nomic-embed-text-v2-moe-q8_0.gguf") catch |err| {
        std.debug.print("Failed to initialize embedding engine: {}\n", .{err});
        return;
    };
    defer engine.deinit();

    // List of texts to embed
    const texts = [_][]const u8{
        "Hello, World!",
        "This is a test sentence.",
        "Zig is a systems programming language.",
    };

    // Embed each text
    for (texts, 0..) |text, i| {
        std.debug.print("\n--- Embedding text {d}: '{s}' ---\n", .{ i + 1, text });

        const embedding = engine.embed(allocator, text) catch |err| {
            std.debug.print("Failed to embed text: {}\n", .{err});
            continue;
        };
        defer allocator.free(embedding);

        // Print first few dimensions for brevity
        std.debug.print("Embedding (first 10 dimensions): ", .{});
        for (0..@min(10, embedding.len)) |j| {
            std.debug.print("{d:.6}", .{embedding[j]});
            if (j < @min(10, embedding.len) - 1) {
                std.debug.print(", ", .{});
            }
        }
        std.debug.print("\n", .{});

        std.debug.print("Full embedding dimensions: {d}\n", .{embedding.len});
        std.debug.print("Successfully generated embedding for '{s}'\n", .{text});
    }

    std.debug.print("\nAll embeddings completed successfully!\n", .{});
}
