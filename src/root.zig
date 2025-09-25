//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

pub const embed = @import("embed.zig");
pub const storage = @import("storage.zig");






test "embed init and deinit" {
    const model_path = "deps/nomic-embed-text-v2-gguf/nomic-embed-text-v2-moe-q8_0.gguf";
    var engine = try embed.EmbeddingEngine.init(model_path);
    defer engine.deinit();

    try std.testing.expect(engine.model != null);
}

test "embed text" {
    const alloc = std.testing.allocator;
    const model_path = "deps/nomic-embed-text-v2-gguf/nomic-embed-text-v2-moe-q8_0.gguf";
    var engine = try embed.EmbeddingEngine.init(model_path);
    defer engine.deinit();

    const text = "Hello, world!";
    const embedding = try engine.embed(alloc, text);
    defer alloc.free(embedding);

    try std.testing.expect(embedding.len == engine.n_embd);
}

test "embed idempotency" {
    const alloc = std.testing.allocator;
    const model_path = "deps/nomic-embed-text-v2-gguf/nomic-embed-text-v2-moe-q8_0.gguf";
    var engine = try embed.EmbeddingEngine.init(model_path);
    defer engine.deinit();

    const text = "The quick brown fox jumps over the lazy dog.";
    const embedding1 = try engine.embed(alloc, text);
    defer alloc.free(embedding1);

    const embedding2 = try engine.embed(alloc, text);
    defer alloc.free(embedding2);

    try std.testing.expect(embedding1.len == embedding2.len);
    for (embedding1, embedding2) |val1, val2| {
        try std.testing.expect(@abs(val1 - val2) < 0.0001);
    }
}

test "embed perf small text" {
    const alloc = std.testing.allocator;
    const model_path = "deps/nomic-embed-text-v2-gguf/nomic-embed-text-v2-moe-q8_0.gguf";
    var engine = try embed.EmbeddingEngine.init(model_path);
    defer engine.deinit();

    const text = "Hello, world!";
    const iterations = 100;

    const start = std.time.milliTimestamp();
    for (0..iterations) |_| {
        const embedding = try engine.embed(alloc, text);
        defer alloc.free(embedding);
    }
    const end = std.time.milliTimestamp();
    const duration = end - start;

    // expected operation cost <5ms
    try std.testing.expect(@divFloor(duration, iterations) < 5);
}

test "embed perf large text" {
    const alloc = std.testing.allocator;
    const model_path = "deps/nomic-embed-text-v2-gguf/nomic-embed-text-v2-moe-q8_0.gguf";
    var engine = try embed.EmbeddingEngine.init(model_path);
    defer engine.deinit();

    const text = "The quick brown fox jumps over the lazy dog. ";
    const iterations = 20;

    const start = std.time.milliTimestamp();
    for (0..iterations) |_| {
        const embedding = try engine.embed(alloc, text);
        defer alloc.free(embedding);
    }

    const end = std.time.milliTimestamp();
    const duration = end - start;

    // expected operation cost <8ms
    try std.testing.expect(@divFloor(duration, iterations) < 8);
}

test "embed perf extra large text" {
    const alloc = std.testing.allocator;
    const model_path = "deps/nomic-embed-text-v2-gguf/nomic-embed-text-v2-moe-q8_0.gguf";
    var engine = try embed.EmbeddingEngine.init(model_path);
    defer engine.deinit();

    const text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";
    const iterations = 10;

    const start = std.time.milliTimestamp();
    for (0..iterations) |_| {
        const embedding = try engine.embed(alloc, text);
        defer alloc.free(embedding);
    }

    const end = std.time.milliTimestamp();
    const duration = end - start;

    // expected operation cost <13ms
    try std.testing.expect(@divFloor(duration, iterations) < 13);
}
