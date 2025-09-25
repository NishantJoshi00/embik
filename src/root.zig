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

    // expected operation cost <10ms
    try std.testing.expect(@divFloor(duration, iterations) < 10);
}

test "embed perf extra large text" {
    const alloc = std.testing.allocator;
    const model_path = "deps/nomic-embed-text-v2-gguf/nomic-embed-text-v2-moe-q8_0.gguf";
    var engine = try embed.EmbeddingEngine.init(model_path);
    defer engine.deinit();

    const text = "In a village of La Mancha, the name of which I have no desire to call to mind, there lived not long since one of those gentlemen that keep a lance in the lance-rack, an old buckler, a lean hack, and a greyhound for coursing. An olla of rather more beef than mutton, a salad on most nights, some lean chorizo, and a pigeon or so constituted his whole diet. He was thin and gaunt, with a face that was weathered and wrinkled from years of outdoor life. His eyes were sharp and bright, reflecting a keen intellect and a restless spirit. He had a habit of reading books of chivalry and adventure, which he devoured with great enthusiasm. This habit had given him a somewhat romantic view of the world, and he often dreamed of embarking on his own adventures. One day, after finishing yet another book on knights and their exploits, he decided to take action. He donned an old suit of armor that had belonged to his great-grandfather, mounted his trusty steed Rocinante, and set out to seek his fortune as a knight-errant. His first challenge came when he encountered a group of windmills on the horizon. Mistaking them for giants, he charged at them with all his might, only to be knocked off his horse by the force of the windmill's blades. Undeterred by this setback, he continued on his journey, determined to prove his valor and win glory in the eyes of the world.";
    const iterations = 10;

    const start = std.time.milliTimestamp();
    for (0..iterations) |_| {
        const embedding = try engine.embed(alloc, text);
        defer alloc.free(embedding);
    }

    const end = std.time.milliTimestamp();
    const duration = end - start;

    // expected operation cost <40ms
    try std.testing.expect(@divFloor(duration, iterations) < 40);
}
